import pytorch_lightning as pl
import os
import torch
import numpy as np
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
from models.resnet_model import MitoSpace3ResNetAutoEncoder
from models.model import MitoSpace3DAutoencoder
from autoencoder import AutoEncoderRunner
from autoencoder_dataset import MitoSpaceAutoEncoderDataset
from utils import load_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Path to your saved checkpoint file
def encode_data(runner, x):
    enc = runner.model.encoder(x)
    return enc


if __name__ == '__main__':
    checkpoint_path = "/home/dhruvagarwal/projects/Manav_MitoSpace/MitoSpace4D/autoencoder/lightning_logs/final_training_sdsc_16_nodes_low_lr_low_gamma/lightning_logs/version_3178623/checkpoints/epoch=8-step=6462.ckpt"
    cfg = load_config('/home/dhruvagarwal/projects/Manav_MitoSpace/MitoSpace4D/autoencoder/config.yaml')
    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner.load_from_checkpoint(checkpoint_path, model=model, cfg=cfg)
    runner.model.eval()

    print(runner.model)
    print("Number of params", count_parameters(runner.model))

    # Path to your data
    data_root = "/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data"

    dataset = MitoSpaceAutoEncoderDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1, pin_memory=True, persistent_workers=True,
                            shuffle=False)

    for batch in dataloader:
        print(batch)
        x, fpath = batch['image'], batch['fpath']
        enc = encode_data(runner, x.to(runner.device))

        enc = enc.cpu().detach().numpy()

        # for i in range(enc.shape[0]):
        #     os.makedirs(os.path.dirname(fpath[i].replace("processed_data", "encoded_data")), exist_ok=True)
        #     np.save(fpath[i].replace("processed_data", "encoded_data"), enc[i])
        #     print(f"Saved {fpath[i].replace('processed_data', 'encoded_data')}")
