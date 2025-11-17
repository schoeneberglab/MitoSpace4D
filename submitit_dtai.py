"""
submitit training script for MitoSpace4D SimCLR training on DeltaAI
Author: Eric Arkfeld

Updates:
- Maps one Slurm task per GPU with 72 CPU cores per task (Grace 72-core CPU per H100).
- Uses SLURM_* env vars for binding (since --cpu-bind/--gpu-bind are srun-only):
  SLURM_CPU_BIND=cores and SLURM_GPU_BIND=map_gpu:<ids>.
- Adds lightweight GPU performance logging via `nvidia-smi` (CSV).
- Adds lightweight host CPU/memory logging via /proc sampler (CSV).
- Optional PyTorch Lightning PerfCallback toggle via env flag.
- Sane NCCL diagnostics for multi-GPU/multinode.
"""
import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from utils.utils import load_config

import submitit

logger = logging.getLogger(__name__)


# --------------------- GPU perf logger (nvidia-smi) ---------------------
def start_nsmi_logger(log_dir: Path, interval_s: int = 1) -> callable:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "gpu_stats.csv"
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.total,pstate",
        "--format=csv",
        "-l", str(interval_s),
    ]
    f = open(out, "w")
    try:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        f.close()
        raise

    def _stop():
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            try:
                f.close()
            except Exception:
                pass

    atexit.register(_stop)
    return _stop


# --------------------- CPU & memory logger (/proc sampler) ---------------------
def _read_meminfo_kb() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            parts = line.split()
            key = parts[0].rstrip(":")
            val_kb = int(parts[1])
            out[key] = val_kb
    return out


def _read_loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg") as fh:
        a, b, c, *_ = fh.read().split()
    return float(a), float(b), float(c)


def _read_proc_stat() -> tuple[int, int]:
    with open("/proc/stat") as fh:
        fields = fh.readline().split()
    vals = list(map(int, fields[1:]))  # user nice system idle iowait irq softirq steal guest guest_nice
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)
    return idle, total


def _read_self_rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        parts = fh.read().split()
    resident_pages = int(parts[1])
    page_size = os.sysconf("SC_PAGE_SIZE")  # bytes
    return resident_pages * page_size / (1024 ** 2)


def start_system_logger(log_dir: Path, interval_s: int = 1) -> callable:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "cpu_mem_stats.csv"
    stop_ev = threading.Event()
    lock = threading.Lock()

    prev_idle, prev_total = _read_proc_stat()

    def loop():
        nonlocal prev_idle, prev_total
        with open(out_path, "w") as f:
            f.write("timestamp_iso,cpu_util_pct,load1,load5,load15,mem_used_mb,mem_avail_mb,mem_total_mb,proc_rss_mb\n")
            f.flush()
            while not stop_ev.wait(interval_s):
                try:
                    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                    idle, total = _read_proc_stat()
                    didle = idle - prev_idle
                    dtotal = total - prev_total
                    prev_idle, prev_total = idle, total
                    cpu_util = 0.0 if dtotal <= 0 else 100.0 * (1.0 - (didle / dtotal))

                    mem = _read_meminfo_kb()
                    mem_total_mb = mem.get("MemTotal", 0) / 1024.0
                    mem_avail_mb = mem.get("MemAvailable", 0) / 1024.0
                    mem_used_mb = max(0.0, mem_total_mb - mem_avail_mb)

                    l1, l5, l15 = _read_loadavg()
                    rss_mb = _read_self_rss_mb()

                    line = f"{now},{cpu_util:.2f},{l1:.2f},{l5:.2f},{l15:.2f},{mem_used_mb:.1f},{mem_avail_mb:.1f},{mem_total_mb:.1f},{rss_mb:.1f}\n"
                    with lock:
                        f.write(line)
                        f.flush()
                except Exception as e:
                    print(f"[system_logger] warning: {e}", file=sys.stderr, flush=True)
                    continue

    t = threading.Thread(target=loop, name="cpu-mem-logger", daemon=True)
    t.start()

    def _stop():
        stop_ev.set()
        t.join(timeout=3)

    atexit.register(_stop)
    return _stop


# --------------------- Argparse ---------------------
def parse_args():
    parser = argparse.ArgumentParser("Submitit for MitoSpace4D SimCLR training")
    parser.add_argument("--ngpus", default=4, type=int, help="GPUs per node")
    parser.add_argument("--nodes", default=8, type=int, help="Number of nodes")
    parser.add_argument("--timeout", default=1440, type=int, help="Duration (minutes), 24h")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--shared_dir", default="/work/nvme/begq/", type=str,
                        help="Shared directory; USER/experiments will be created under this.")
    parser.add_argument("--partition", default="ghx4", type=str, help="Partition to submit to (GH200/H100)")
    parser.add_argument("--constraint", default="", type=str, help="Optional Slurm constraint")
    parser.add_argument("--comment", default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="begq-dtai-gh", type=str, help="Slurm account (GH200)")
    parser.add_argument("--exclude", default="", type=str, help="Exclude hosts")

    parser.add_argument("--config", default="/u/earkfeld/MitoSpace4D/simclr/config.yaml", type=str,
                        help="Path to SimCLR config.yaml")
    parser.add_argument("--log-every-n-steps", default=None, type=int,
                        help="Optional override for Lightning logging frequency")

    parser.add_argument("--perf-log", action="store_true", help="Enable GPU + CPU/memory CSV logging")
    parser.add_argument("--nsmi-interval", type=int, default=30, help="nvidia-smi sampling interval (s)")
    parser.add_argument("--sys-interval", type=int, default=30, help="CPU/memory sampling interval (s)")
    parser.add_argument("--use-pl-callback", action="store_true",
                        help="Set USE_PL_PERF_CALLBACK=1 for Lightning-side PerfCallback")
    
    return parser.parse_args()


# --------------------- Paths ---------------------
def get_shared_folder(shared_dir: str) -> Path:
    user = os.getenv("USER") or "user"
    if Path(shared_dir).is_dir():
        p = Path(shared_dir) / user / "experiments"
        p.mkdir(exist_ok=True, parents=True)
        return p
    raise RuntimeError(f"No shared folder available at {shared_dir}")


def get_init_file(shared_dir: str):
    os.makedirs(str(get_shared_folder(shared_dir)), exist_ok=True)
    init_file = get_shared_folder(shared_dir) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def _coerce_paths_to_str(ns: SimpleNamespace) -> SimpleNamespace:
    d = vars(ns).copy()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return SimpleNamespace(**d)


# --------------------- Job object ---------------------
class Trainer(object):
    def __init__(self, args, cfg):
        d = vars(args).copy()
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        self.args = SimpleNamespace(**d)
        self.cfg = cfg

    def __call__(self):
        from train_simclr import run_from_cfg  # local module

        job_env = submitit.JobEnvironment()
        logger.info(
            f"Process group: {job_env.num_tasks} tasks, "
            f"rank: {job_env.global_rank}, node: {job_env.hostnames}"
        )

        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        ngpus = int(self.args.ngpus)
        gpu_map = ",".join(str(i) for i in range(ngpus))
        os.environ.setdefault("SLURM_CPU_BIND", "cores")
        if ngpus > 0:
            os.environ.setdefault("SLURM_GPU_BIND", f"map_gpu:{gpu_map}")

        cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "72")
        os.environ.setdefault("OMP_NUM_THREADS", cpus_per_task)
        os.environ.setdefault("MKL_NUM_THREADS", cpus_per_task)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", cpus_per_task)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", cpus_per_task)

        if self.args.use_pl_callback:
            os.environ["USE_PL_PERF_CALLBACK"] = "1"

        stop_funcs = []
        try:
            if self.args.perf_log:
                perf_dir = Path(self.args.job_dir) / "perf_logs"
                try:
                    stop_funcs.append(start_nsmi_logger(perf_dir, interval_s=int(self.args.nsmi_interval)))
                    logger.info(f"Started nvidia-smi logging at {perf_dir/'gpu_stats.csv'}")
                except FileNotFoundError:
                    logger.warning("nvidia-smi not found; skipping GPU perf logging.")
                stop_funcs.append(start_system_logger(perf_dir, interval_s=int(self.args.sys_interval)))
                logger.info(f"Started CPU/memory logging at {perf_dir/'cpu_mem_stats.csv'}")
        except Exception as e:
            logger.warning(f"Failed to start perf loggers: {e}")

        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

        try:
            log_every = self.args.log_every_n_steps if self.args.log_every_n_steps is not None else 100
            run_from_cfg(self.cfg, log_every_n_steps=log_every)
        finally:
            for s in stop_funcs[::-1]:
                try:
                    s()
                except Exception:
                    pass
            if self.args.perf_log:
                logger.info(f"Wrote perf logs under {Path(self.args.job_dir) / 'perf_logs'}")

    def checkpoint(self):
        logger.info("Requeuing job with same arguments and cfg.")
        return submitit.helpers.DelayedSubmission(type(self)(self.args, self.cfg))


# --------------------- Launcher ---------------------
def main():
    args = parse_args()

    if args.job_dir == "":
        job_dir_path = "./runs/%j"
        args.job_dir = str(job_dir_path)
    elif isinstance(args.job_dir, Path):
        args.job_dir = str(args.job_dir)

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition
    exclude = args.exclude

    kwargs = {}
    if len(args.constraint):
        kwargs["slurm_constraint"] = args.constraint
    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.qos:
        kwargs["slurm_qos"] = args.qos
    if args.account:
        kwargs["slurm_account"] = args.account
    if exclude:
        kwargs["slurm_exclude"] = exclude

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=72,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_exclusive=True,
        mem_gb=0,
        **kwargs,
    )

    _ = get_init_file(args.shared_dir)

    base_cfg = load_config(args.config)
    args = _coerce_paths_to_str(SimpleNamespace(**vars(args)))

    executor.update_parameters(name=base_cfg["experiment_name"])

    trainer = Trainer(args, base_cfg)
    job = executor.submit(trainer)
    logger.info(f"Submitted job {job.job_id}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()