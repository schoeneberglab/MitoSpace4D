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
from autoencoder.models.models import MitoSpace3DAutoencoder
import io
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
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
        # self.ssim_loss = MS_SSIM(
        #                     data_range=1.0,      # Assuming the random tensors are between 0 and 1
        #                     size_average=True,
        #                     win_size=11,
        #                     win_sigma=1.5,
        #                     channel=2,           # Number of channels is 2
        #                     spatial_dims=3       # Set spatial dimensions to 3 for 3D images
        #                 )
        # Initialize cumulative loss and step counter for monitoring total loss
        self.cumulative_loss = 0.0
        self.step_counter = 0

    def flush_bank(self):
        self.data_bank = {"Train": [], "Val": []}

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def batch_step(self, batch: Dict[str, Any]):
        z = self.model(batch)
        loss = self.criterion(batch, z)
        # b, t, c, d, x, y = batch.shape
        # batch = batch.view(b*t, c, d, x, y)
        # z = z.view(b*t, c, d, x, y)
        # print(batch.shape, z.shape)
        # loss += self.ssim_loss(batch, z)
        # z = z.view(b, t, c, d, x, y)
        return loss, z

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, z = self.batch_step(batch)

        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', learning_rate, on_step=True, on_epoch=False)
        self.log('Train/loss', loss)

        # Accumulate total loss and increment step counter
        self.cumulative_loss += loss.item()
        self.step_counter += 1

        # Log total loss every 2000 steps
        if self.step_counter == 2000:
            self.log('Train/total_loss', self.cumulative_loss)
            self.cumulative_loss = 0.0  # Reset cumulative loss
            self.step_counter = 0       # Reset step counter

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


if __name__ == '__main__':
    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner(model=model)
    
    # Create a dummy batch with shape (batch, t, c, z, x, y)
    batch_size = 1
    t, c, z, x, y = 20, 2, 60, 256, 256
    dummy_data = torch.rand(batch_size, t, c, z, x, y)  # Values between 0 and 1

    # Run batch_step and print output shapes
    loss, output = runner.batch_step(dummy_data)
    print("Loss:", loss.item())
    print("Output shape:", output.shape)