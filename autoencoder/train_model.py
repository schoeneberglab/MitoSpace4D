import argparse
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import torch.backends.cudnn as cudnn
from utils import load_config
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from typing import Any, Dict, List, Tuple
from models.resnet_model import MitoSpace3ResNetAutoEncoder
from models.model import MitoSpace3DAutoencoder
from autoencoder import AutoEncoderRunner
from autoencoder_dataset import MitoSpaceAutoEncoderDataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


parser = argparse.ArgumentParser(description='MitoSpace4D')
parser.add_argument('--config', default='/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml', type=str,
                    help='Config path.')

def main():
    print("Starting Training")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(cfg)

    dataset = MitoSpaceAutoEncoderDataset(cfg = cfg)
    print("Total samples in dataset:", len(dataset))
    model = MitoSpace3DAutoencoder()
    if cfg['training']['continue_from_ckpt_wo_opt'] != 'None':
        print("Loaded from checkpoint path: ", cfg['training']['continue_from_ckpt_wo_opt'])
        runner = AutoEncoderRunner.load_from_checkpoint(cfg['training']['continue_from_ckpt_wo_opt'], model = model, cfg = cfg)

    else:
        runner = AutoEncoderRunner(model = model, cfg = cfg)        

    train_loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'],
                              shuffle=cfg['training']['shuffle'],
                              drop_last=True,
                              num_workers=cfg['training']['num_workers'],
                              pin_memory=cfg['training']['pin_memory'],
                              prefetch_factor=cfg['training']['prefetch_factor'],
                              persistent_workers=cfg['training']['persistent_workers']
                              )

    trainer = pl.Trainer(
        default_root_dir=cfg['logging_params']['save_path'],
        accelerator=cfg['distributed']['accelerator'],
        devices=cfg['distributed']['num_gpus'],
        max_epochs=cfg['training']['max_epochs'],
        callbacks=[
            ModelCheckpoint(
                save_last=cfg['training']['ckpt_callback']['save_last'],
                save_top_k=cfg['training']['ckpt_callback']['save_top_k'],  # Keep only the best checkpoint
                monitor=cfg['training']['ckpt_callback']['monitor'],  # Monitor cumulative training loss
                mode=cfg['training']['ckpt_callback']['mode']  # Minimize cumulative training loss
            ),
            LearningRateMonitor("epoch"),
        ],
        precision=16,
    )

    if trainer.logger:
        log_dir = trainer.logger.log_dir + '/'
        os.makedirs(log_dir, exist_ok=True)
        dest_path = os.path.join(log_dir, "config.yaml")
        shutil.copy(args.config, dest_path)
        print(f"Current Lightning logging folder: {log_dir}")

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    trainer.fit(runner, train_loader)

if __name__ == '__main__':
    main()
    # dataset = MitoSpaceAutoEncoderDataset(
    #     root_dir='/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data/')

    # checkpoint_path = "/home/dhruvagarwal/projects/MitoSpace4D/autoencoder/runs/autoencoder/lightning_logs/version_2/checkpoints/epoch=31-step=43136.ckpt"
    # checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)

    # Create DataLoader for training
    # train_loader = DataLoader(dataset, batch_size=1,
    #                           shuffle=True,
    #                           drop_last=True,
    #                           num_workers=4,
    #                           pin_memory=True,
    #                           prefetch_factor=2,
    #                           persistent_workers=True
    #                           )

    # model = MitoSpace3ResNetAutoEncoder()
    # runner = AutoEncoderRunner(model=model)
    # print(model)
    # train_model(runner, train_loader)
