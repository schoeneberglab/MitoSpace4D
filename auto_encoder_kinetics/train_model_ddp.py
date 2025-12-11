# import os
# import torch
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning import loggers as pl_loggers

# from autoencoder.autoencoder_models import MitoSpace3DAutoencoder
# from autoencoder.autoencoder_runner import AutoEncoderRunner
# from autoencoder.autoencoder_dataset import MitoSpaceAutoEncoderDataset

# def env_int(name: str, default: int) -> int:
#     v = os.environ.get(name)
#     try:
#         return int(v) if v is not None else default
#     except ValueError:
#         return default

# def train_model(model, train_loader, experiment_path: str):
#     # Resolve cluster-driven settings
#     num_nodes = env_int("SLURM_NNODES", 1)
#     devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

#     # Loggers / callbacks
#     logger = pl_loggers.TensorBoardLogger(
#         save_dir=experiment_path, name="", version=None, default_hp_metric=False
#     )
#     ckpt_cb = ModelCheckpoint(
#         dirpath=os.path.join(experiment_path, "checkpoints"),
#         filename="{epoch:03d}-{step:06d}-{loss:.4f}",
#         save_top_k=3,
#         monitor="Train/total_loss",  # ensure your LightningModule logs this
#         mode="min",
#         save_last=True,
#         auto_insert_metric_name=False,
#     )
#     lr_cb = LearningRateMonitor(logging_interval="epoch")

#     try:
#         torch.set_float32_matmul_precision("high")
#     except Exception:
#         pass

#     trainer = pl.Trainer(
#         default_root_dir="runs",
#         accelerator="gpu",
#         devices=devices if devices > 0 else None,
#         num_nodes=num_nodes,
#         strategy="ddp",                # ‘ddp’ pairs with torchrun
#         sync_batchnorm=True,
#         precision="16-mixed",
#         max_epochs=100,
#         log_every_n_steps=50,
#         callbacks=[ckpt_cb, lr_cb],
#         logger=logger,
#         detect_anomaly=False,
#         gradient_clip_val=0.5,         # optional but helpful
#         enable_progress_bar=True,
#     )

#     # Optional graph logging (only works on rank 0; Lightning will gate the logger)
#     trainer.logger._log_graph = True

#     trainer.fit(model, train_loader)
#     result = trainer.test(model, dataloaders=train_loader, verbose=False)
#     return model, result

# if __name__ == "__main__":
#     pl.seed_everything(42, workers=True)

#     data_dirs = ["/work/nvme/begq/MitoSpace4D/data/2025_data"]
#     dataset = MitoSpaceAutoEncoderDataset(root_dirs=data_dirs)

#     print("Total samples in dataset:", len(dataset))

#     experiment_path = "/home/earkfeld/Projects/MitoSpace4D/runs/lightning_logs/kinetics_autoencoder"
#     os.makedirs(experiment_path, exist_ok=True)

#     # Tune workers per rank: start with 8–12; bump if input pipeline is the bottleneck
#     workers_default = 24
#     num_workers = env_int("DATALOADER_WORKERS", workers_default)

#     train_loader = DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=True,              # Lightning will replace with DistributedSampler under DDP
#         drop_last=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         prefetch_factor=4,
#         persistent_workers=True,
#     )

#     model = MitoSpace3DAutoencoder()
#     runner = AutoEncoderRunner(model=model)

#     train_model(runner, train_loader, experiment_path)

import os
import pathlib
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

try:
    from .autoencoder_models_resnet import MitoSpace3DAutoencoder
    from .autoencoder_runner import AutoEncoderRunner
    from .autoencoder_dataset import MitoSpaceAutoEncoderDataset
except ImportError:
    from autoencoder_models_resnet import MitoSpace3DAutoencoder
    from autoencoder_runner import AutoEncoderRunner
    from autoencoder_dataset import MitoSpaceAutoEncoderDataset

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

def is_global_zero() -> bool:
    # torchrun sets RANK/WORLD_SIZE; fallback to single-process
    return int(os.environ.get("RANK", "0")) == 0

def get_shared_run_root() -> pathlib.Path:
    """
    Prefer a shared per-job directory from the jobscript (RUN_DIR).
    Otherwise use SLURM_SUBMIT_DIR (shared on the allocation).
    If neither is set (local dev), fall back to CWD.
    """
    base = (
        os.environ.get("RUN_DIR")
        or os.environ.get("SLURM_SUBMIT_DIR")
        or os.getcwd()
    )
    # Keep things grouped under autoencoder/runs/<job_id or 'local'>
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return pathlib.Path(base) / "autoencoder" / "runs" / job_id

def barrier_if_possible():
    # Only attempt a barrier if distributed is initialized
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        # Don't let sync failures crash local/single-node cases
        pass

def train_model(model, train_loader, experiment_path: str, ckpt_path: str = None):
    # Resolve cluster-driven settings
    num_nodes = env_int("SLURM_NNODES", 1)
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Loggers / callbacks
    logger = pl_loggers.TensorBoardLogger(
        save_dir=experiment_path, name="", version=None, default_hp_metric=False
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(experiment_path, "checkpoints"),
        filename="{epoch:03d}-{step:06d}-{loss:.4f}",
        save_top_k=3,
        monitor="Train/total_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_cb = LearningRateMonitor(logging_interval="epoch")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    trainer = pl.Trainer(
        default_root_dir=experiment_path,
        accelerator="gpu",
        devices=devices if devices > 0 else None,
        num_nodes=num_nodes,
        strategy="ddp",
        sync_batchnorm=True,
        precision="16-mixed",
        max_epochs=15,
        log_every_n_steps=50,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        detect_anomaly=False,
        enable_progress_bar=True,
    )

    # Optional graph logging (rank 0 gated by Lightning internally)
    trainer.logger._log_graph = True

    trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    result = trainer.test(model, dataloaders=train_loader, verbose=False)
    return model, result


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    # ----- Shared, per-job experiment directory -----
    shared_root = get_shared_run_root()
    experiment_path = str(shared_root / "lightning_logs" / "kinetics_autoencoder")

    # Create directories on rank 0 only, then let others continue
    if is_global_zero():
        (shared_root / "lightning_logs" / "kinetics_autoencoder" / "checkpoints").mkdir(
            parents=True, exist_ok=True
        )
    barrier_if_possible()

    # ----- Dataset / DataLoader -----
    data_dirs = ["/work/nvme/begq/MitoSpace4D/data/2025_data"]
    dataset = MitoSpaceAutoEncoderDataset(root_dir=data_dirs[0]) # Clean this up later

    print("Total samples in dataset:", len(dataset))

    # Tune workers per rank: start with 8–12; env override allowed
    workers_default = 24
    num_workers = env_int("DATALOADER_WORKERS", workers_default)

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,              # Lightning will swap in DistributedSampler under DDP
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True,
    )

    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner(model=model)

    # Experiment 0
    # resume_ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1023134/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    # resume_ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1023462/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    # resume_ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1023524/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    
    # Experiment 1
    # resume_ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1023795/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"

    # Experiment 2
    # resume_ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/2_kinetics_ae_do0.2_mse+l1/1029930/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    resume_ckpt_path = None
    
    train_model(runner, train_loader, experiment_path, ckpt_path=resume_ckpt_path)