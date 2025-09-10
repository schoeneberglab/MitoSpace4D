"""
submitit training script for MitoSpace4D SimCLR training on DeltaAI
Author: Eric Arkfeld
"""
import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
import submitit

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Submitit for MitoSpace4D SimCLR training")
    #-- Cluster params
    parser.add_argument("--ngpus", default=4, type=int, help="GPUs per node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--timeout", default=1440, type=int, help="Duration (minutes), 24h")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument(
        "--shared_dir",
        default="/work/nvme/begq/",
        type=str,
        help="Shared directory; USER/experiments will be created under this.",
    )
    parser.add_argument("--partition", default="ghx4", type=str, help="Partition to submit to (GH200/H100)")
    parser.add_argument("--constraint", default="", type=str, help="Optional Slurm constraint")
    parser.add_argument("--comment", default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="begq-dtai-gh", type=str, help="Slurm account (GH200)")
    parser.add_argument("--exclude", default="", type=str, help="Exclude hosts")
    
    #-- train_simclr.py passthroughs
    parser.add_argument(
        "--config",
        default="/u/earkfeld/MitoSpace4D/simclr/config.yaml",
        type=str,
        help="Path to SimCLR config.yaml passed to train_simclr.py",
    )
    parser.add_argument(
        "--log-every-n-steps",
        default=None,
        type=int,
        help="Optional override for Lightning logging frequency",
    )
    return parser.parse_args()


def get_shared_folder(shared_dir: str) -> Path:
    user = os.getenv("USER")
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


class Trainer(object):
    def __init__(self, args):
        d = vars(args).copy()
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        self.args = SimpleNamespace(**d)

    def __call__(self):
        import submitit
        import train_simclr

        job_env = submitit.JobEnvironment()
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

        argv = ["train_simclr.py", "--config", str(self.args.config)]
        if self.args.log_every_n_steps is not None:
            argv += ["--log-every-n-steps", str(self.args.log_every_n_steps)]

        old_argv = sys.argv
        try:
            sys.argv = argv
            train_simclr.main()
        finally:
            sys.argv = old_argv

    def checkpoint(self):
        import submitit
        logger.info("Requeuing job with same arguments.")
        return submitit.helpers.DelayedSubmission(type(self)(self.args))


def main():
    args = parse_args()

    # Compute job_dir
    if args.job_dir == "":
        job_dir_path = get_shared_folder(args.shared_dir) / "%j"
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

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,          # --gres=gpu:<ngpus>
        tasks_per_node=num_gpus_per_node,         # --ntasks-per-node=<ngpus>
        cpus_per_task=72,                         # --cpus-per-task=72
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_exclusive=True,
        mem_gb=0,                                 # 0 => all available node memory
        **kwargs,
    )

    # Match the sbatch job-name
    executor.update_parameters(name="mitospace_autoencoded_normal_run")

    # Keep parity with prior flow by creating a shared init file
    _ = get_init_file(args.shared_dir)

    args = _coerce_paths_to_str(SimpleNamespace(**vars(args)))

    trainer = Trainer(args)
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