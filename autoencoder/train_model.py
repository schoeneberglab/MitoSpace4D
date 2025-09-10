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
from autoencoder.autoencoder_models import MitoSpace3DAutoencoder
from archive.model_params import MitoSpace3DAutoencoderHigherParams
from autoencoder.autoencoder_runner import AutoEncoderRunner
from autoencoder_dataset import MitoSpaceAutoEncoderDataset, NormalizeChannelsByPath

print("Imported packages")

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def train_model(model, train_loader):
    trainer = pl.Trainer(
        default_root_dir='lightning_logs/sanity_check',
        accelerator="gpu",
        devices=-1,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                save_top_k=3,  # Keep only the best checkpoint
                monitor="Train/total_loss",  # Monitor cumulative training loss
                mode="min",
                save_last=True,  # Minimize cumulative training loss
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
    data_dirs = [
        "/mnt/aquila0/others/MitoSpace4D/data/aligned",                     # Summer 2024
        "/mnt/aquila0/ssd_processing/Others/MitoSpace4D/summer_2025_new/"   # Summer 2025
    ]

    ds_transform = NormalizeChannelsByPath(
        path_substr='2024',
        max_values=[25000, 10000]  # TMRM (ch0), MTG (ch1)
    )

    # Create a dataset object
    dataset = MitoSpaceAutoEncoderDataset(
        root_dirs=data_dirs,
        transform=ds_transform
    )
    
    print("Total samples in dataset:", len(dataset))

    # checkpoint_path = "/home/earkfeld/Projects/MitoSpace4D/autoencoder/runs/autoencoder/lightning_logs/"
    # checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)
    # # load the model state dict
    # model_state_dict = checkpoint['state_dict']
    # print(model_state_dict)


    # Create DataLoader for training
    train_loader = DataLoader(dataset, 
                              batch_size=1,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True
                              )

    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner(model=model)
    print(model)
    train_model(runner, train_loader)