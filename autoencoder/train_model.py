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
from autoencoder import AutoEncoderRunner
from mitospace_dataset import MitoSpaceAutoEncoderDataset
from models import MitoSpace3DAutoencoder


def train_model(model, train_loader):
    trainer = pl.Trainer(
        default_root_dir='/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/runs/autoencoder/',
        accelerator="gpu",
        devices=-1,
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
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
    dataset = MitoSpaceAutoEncoderDataset(
        root_dir='data')
    print("Total samples in dataset:", len(dataset))

    # Create DataLoader for training
    train_loader = DataLoader(dataset, batch_size=4,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True
                              )

    # pbar = tqdm(len(train_loader))
    # for batch in train_loader:
    #     print(batch.shape)
    #     pbar.update(1)

    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner(model)
    print(model)
    train_model(runner, train_loader)