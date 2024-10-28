import argparse
import torch
import numpy as np
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
import matplotlib.pyplot as plt
import io
from PIL import Image

class AutoEncoderRunner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.intermediate_outputs = []
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.data_bank = {"Train": [], "Val": []}

        print(f"###################### Using MSE Loss For Training ##################")

        self.criterion = nn.L1Loss()

    def flush_bank(self):
        self.data_bank = {"Train": [], "Val": []}

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def batch_step(self, batch: Dict[str, Any]):
        z = self.model(batch)
        loss = self.criterion(batch, z)

        return loss, z

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, z = self.batch_step(batch)

        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', learning_rate, on_step=True, on_epoch=False)
        self.log('Train/loss', loss)

        if self.global_step % 100 == 0:
            self.log_images(batch, z)

        return loss

    def log_images(self, batch, z):
        y = batch
        reconstructed_images = z

        batch_size, timesteps, channels, depth, height, width = y.shape

        # Randomly select batch_idx, timestep, and depth
        batch_idx = np.random.randint(batch_size)
        timestep = np.random.randint(timesteps)
        depth_idx = np.random.randint(depth)

        # Convert tensors to numpy arrays for plotting
        original_img_channel_1 = y[batch_idx, timestep, 0, depth_idx, :, :].detach().cpu().numpy()
        original_img_channel_2 = y[batch_idx, timestep, 1, depth_idx, :, :].detach().cpu().numpy()

        reconstructed_img_channel_1 = reconstructed_images[batch_idx, timestep, 0, depth_idx, :, :].detach().cpu().numpy()
        reconstructed_img_channel_2 = reconstructed_images[batch_idx, timestep, 1, depth_idx, :, :].detach().cpu().numpy()

        sep = 20
        channel_1 = np.zeros((original_img_channel_1.shape[-2], original_img_channel_1.shape[-1] * 2 + sep))
        channel_2 = np.zeros((original_img_channel_2.shape[-2], original_img_channel_2.shape[-1] * 2 + sep))

        channel_1[:, :original_img_channel_1.shape[1]] = original_img_channel_1
        channel_1[:, original_img_channel_1.shape[1]+sep:] = reconstructed_img_channel_1

        channel_2[:, :original_img_channel_1.shape[1]] = original_img_channel_2
        channel_2[:, original_img_channel_1.shape[1]+sep:] = reconstructed_img_channel_2

        cm = plt.get_cmap('viridis')
        channel_1 = cm(channel_1)[:, :, :3]
        channel_2 = cm(channel_2)[:, :, :3]

        self.logger.experiment.add_image('Train/TMRM', channel_1, self.global_step, dataformats='HWC')
        self.logger.experiment.add_image('Train/MitoTracker', channel_2, self.global_step, dataformats='HWC')