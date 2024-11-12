import argparse
import time

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
from pytorch_lightning.callbacks import LearningRateMonitor
from typing import Any, Dict, List, Tuple
from resnet_model import MitoSpace3ResNetAutoEncoder
from models import MitoSpace3DAutoencoder
from autoencoder import AutoEncoderRunner
from autoencoder_dataset import MitoSpaceAutoEncoderDataset

print("Imported packages")

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def train_model(model, train_loader):
    trainer = pl.Trainer(
        default_root_dir='lightning_logs/resnet_model_25k_10k_clipped_gamma_1e-1',
        accelerator="gpu",
        devices=-1,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                save_top_k=3,  # Keep only the best checkpoint
                monitor="Train/total_loss",  # Monitor cumulative training loss
                mode="min"  # Minimize cumulative training loss
            ),
            LearningRateMonitor("epoch"),
        ],
        precision=16,
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    trainer.fit(model, train_loader)

    result = trainer.test(model, dataloaders=train_loader, verbose=False)
    return model, result


if __name__ == '__main__':
    print("Started Training Loop")
    dataset = MitoSpaceAutoEncoderDataset(
        root_dir='/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data/')
    print("Total samples in dataset:", len(dataset))

    # checkpoint_path = "/home/dhruvagarwal/projects/MitoSpace4D/autoencoder/runs/autoencoder/lightning_logs/version_2/checkpoints/epoch=31-step=43136.ckpt"
    # checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)

    # Create DataLoader for training
    train_loader = DataLoader(dataset, batch_size=1,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True
                              )

    model = MitoSpace3ResNetAutoEncoder()
    runner = AutoEncoderRunner(model=model)
    print(model)
    train_model(runner, train_loader)
