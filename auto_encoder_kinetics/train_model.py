import argparse
import time
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from typing import Any, Dict, List, Tuple

try:
    from .autoencoder_models_resnet import MitoSpace3DAutoencoder
    from .autoencoder_runner import AutoEncoderRunner
    from .autoencoder_dataset import MitoSpaceAutoEncoderDataset, NormalizeChannelsByPath
except ImportError:
    from autoencoder_models_resnet import MitoSpace3DAutoencoder
    from autoencoder_runner import AutoEncoderRunner
    from autoencoder_dataset import MitoSpaceAutoEncoderDataset, NormalizeChannelsByPath

def train_model(model, train_loader):
    trainer = pl.Trainer(
        default_root_dir="runs",   # use your experiment path
        accelerator="gpu",
        devices=4,                     # or an int per node
        strategy="ddp",                     # explicit DDP
        sync_batchnorm=True,                # good practice across GPUs
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                save_top_k=3,
                monitor="Train/total_loss",
                mode="min",
                save_last=True,
            ),
            LearningRateMonitor("epoch"),
        ],
        precision="16-mixed",
        log_every_n_steps=50,
    )

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    trainer.fit(model, train_loader)
    result = trainer.test(model, dataloaders=train_loader, verbose=False)
    return model, result

# def train_model(model, train_loader):
#     trainer = pl.Trainer(
#         default_root_dir="runs",
#         accelerator="gpu",
#         num_nodes=2,
#         strategy="ddp",
#         sync_batchnorm=True,
#         max_epochs=100,
#         callbacks=[
#             ModelCheckpoint(save_top_k=3, monitor="Train/total_loss", mode="min", save_last=True),
#             LearningRateMonitor("epoch"),
#         ],
#         precision="16-mixed",
#         log_every_n_steps=50,
#     )
#     torch.set_float32_matmul_precision("high")
#     trainer.logger._log_graph = True
#     trainer.logger._default_hp_metric = None
#     trainer.fit(model, train_loader)
#     result = trainer.test(model, dataloaders=train_loader, verbose=False)
#     return model, result


if __name__ == '__main__':
    print("Started Training Loop")
    data_dirs = [
        "/work/nvme/begq/MitoSpace4D/data/2025_data"   # Summer 2025
    ]

    # ds_transform = NormalizeChannelsByPath(
    #     path_substr='2024',
    #     max_values=[25000, 10000]  # TMRM (ch0), MTG (ch1)
    # )

    # Create a dataset object
    dataset = MitoSpaceAutoEncoderDataset(
        root_dirs=data_dirs,
        # transform=ds_transform
    )
    
    print("Total samples in dataset:", len(dataset))

    # checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)
    # # load the model state dict
    # model_state_dict = checkpoint['state_dict']
    # print(model_state_dict)

    # Create DataLoader for training
    train_loader = DataLoader(dataset, 
                              batch_size=4,
                              shuffle=True,
                              drop_last=True,
                              num_workers=24,
                              pin_memory=True,
                              prefetch_factor=4,
                              persistent_workers=True
                              )

    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner(model=model)
    train_model(runner, train_loader)