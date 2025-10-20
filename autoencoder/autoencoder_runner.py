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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from PIL import Image

from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
from autoencoder.ae_loss import ReconstructionLoss

class AutoEncoderRunner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.intermediate_outputs = []

        # Optimizer: AdamW tends to be more stable for ResNet-style AEs
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,            # safer default for 3D ResNet AE
            betas=(0.9, 0.95),
            weight_decay=0.05
        )
        # Scheduler will be constructed in configure_optimizers once total steps are known
        self.scheduler = None

        print("###################### Using MSE/L1/SSIM ReconstructionLoss ##################")
        self.recon_loss = ReconstructionLoss()

        # Cumulative loss bookkeeping
        self.cumulative_loss = 0.0
        self.step_counter = 0

        # Simple container for any cached items (kept from original)
        self.data_bank = {"Train": [], "Val": []}

    def flush_bank(self):
        self.data_bank = {"Train": [], "Val": []}

    def configure_optimizers(self):
        optimizer = self.optimizer

        # Build warm-up -> cosine schedule *by step* once Trainer knows total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.05 * total_steps))
        cosine_steps = max(1, total_steps - warmup_steps)

        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )

        # Step scheduler every optimization step
        return [optimizer], [{
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }]

    def batch_step(self, batch: Dict[str, Any]):
        z = self.model(batch)
        loss, metrics = self.recon_loss(batch, z)

        # Log the loss components
        for key, value in metrics.items():
            self.log(f'Train/{key}', value, on_step=True, on_epoch=True)

        return loss, z

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, z = self.batch_step(batch)

        # Log current LR
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', learning_rate, on_step=True, on_epoch=False)

        # Log training loss
        self.log('Train/loss', loss, on_epoch=True, on_step=True)

        # Accumulate total loss and increment step counter
        self.cumulative_loss += loss.item()
        self.step_counter += 1

        # Log total loss every 2000 steps
        if self.step_counter == 2000:
            self.log('Train/total_loss', self.cumulative_loss)
            self.cumulative_loss = 0.0
            self.step_counter = 0

        # Periodic image logging
        if self.global_step % 100 == 0:
            self.log_images(batch, z)

        return loss

    def log_images(self, batch, z, as_mip=True):
        y = batch
        reconstructed_images = z

        batch_size, timesteps, channels, depth, height, width = y.shape

        # Randomly select indices
        batch_idx = np.random.randint(batch_size)
        timestep = np.random.randint(timesteps)
        depth_idx = np.random.randint(depth)

        if as_mip:
            # Maximum Intensity Projection along depth
            original_img_channel_1 = torch.max(y[batch_idx, timestep, 0, :, :, :], dim=0).values.detach().cpu().numpy()
            original_img_channel_2 = torch.max(y[batch_idx, timestep, 1, :, :, :], dim=0).values.detach().cpu().numpy()

            reconstructed_img_channel_1 = torch.max(reconstructed_images[batch_idx, timestep, 0, :, :, :], dim=0).values.detach().cpu().numpy()
            reconstructed_img_channel_2 = torch.max(reconstructed_images[batch_idx, timestep, 1, :, :, :], dim=0).values.detach().cpu().numpy()
        else:
            # Single-slice visualization
            original_img_channel_1 = y[batch_idx, timestep, 0, depth_idx, :, :].detach().cpu().numpy()
            original_img_channel_2 = y[batch_idx, timestep, 1, depth_idx, :, :].detach().cpu().numpy()

            reconstructed_img_channel_1 = reconstructed_images[batch_idx, timestep, 0, depth_idx, :, :].detach().cpu().numpy()
            reconstructed_img_channel_2 = reconstructed_images[batch_idx, timestep, 1, depth_idx, :, :].detach().cpu().numpy()

        sep = 20
        channel_1 = np.zeros((original_img_channel_1.shape[-2], original_img_channel_1.shape[-1] * 2 + sep))
        channel_2 = np.zeros((original_img_channel_2.shape[-2], original_img_channel_2.shape[-1] * 2 + sep))

        channel_1[:, :original_img_channel_1.shape[1]] = original_img_channel_1
        channel_1[:, original_img_channel_1.shape[1] + sep:] = reconstructed_img_channel_1

        channel_2[:, :original_img_channel_1.shape[1]] = original_img_channel_2
        channel_2[:, original_img_channel_1.shape[1] + sep:] = reconstructed_img_channel_2

        cm = plt.get_cmap('viridis')
        channel_1 = cm(channel_1)[:, :, :3]
        channel_2 = cm(channel_2)[:, :, :3]

        self.logger.experiment.add_image('Train/TMRM', channel_1, self.global_step, dataformats='HWC')
        self.logger.experiment.add_image('Train/MitoTracker', channel_2, self.global_step, dataformats='HWC')


if __name__ == '__main__':
    model = MitoSpace3DAutoencoder()
    ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1081149/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    runner = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
    print("Loaded model from checkpoint.")

    # Create a dummy batch with shape (batch, t, c, z, x, y)
    batch_size = 1
    t, c, z, x, y = 20, 2, 60, 256, 256
    dummy_data = torch.rand(batch_size, t, c, z, x, y).cuda()  # Values between 0 and 1

    # Run batch_step and print output shapes
    loss, output = runner.batch_step(dummy_data)
    print("Loss:", loss.item())
    print("Output shape:", output.shape)