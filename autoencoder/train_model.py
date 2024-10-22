import argparse
import torch
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

def train_model(model, train_loader, val_loader, test_loader):
    trainer = pl.Trainer(
        default_root_dir='./logs/',
        accelerator="gpu",
        devices=1,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    trainer.fit(model, train_loader, val_loader)
                
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

if __name__ == '__main__':
    dataset = MitoSpaceAutoEncoderDataset(root_dir='data')
    print("Total samples in dataset:", len(dataset))
    
    # Create DataLoader for training
    train_loader = dataset.get_dataloader(batch_size=32, split='train', train_split=0.8, val_split=0.1, shuffle=True)
    val_loader = dataset.get_dataloader(batch_size=32, split='val', train_split=0.8, val_split=0.1, shuffle=True)
    test_loader = dataset.get_dataloader(batch_size=32, split='test', train_split=0.8, val_split=0.1, shuffle=True)
    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner(model)
    print(len(train_loader), len(val_loader), len(test_loader))
    print(model)
    train_model(runner, train_loader, val_loader, test_loader)