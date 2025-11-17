import argparse
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import AdvancedProfiler
import warnings
from tqdm import tqdm
import os.path as osp

from simclr.simclr import SimCLRRunner
from simclr.models import MitoSpace4DConvLSTM
from simclr.models_transformer import MitoSpace4DTransformer

from autoencoder.autoencoder_runner import AutoEncoderRunner
from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder

from simclr.augmentations import DataAugmentation
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

from utils.utils import load_config

cudnn.benchmark = True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MitoSpace4D')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--config', default='/u/earkfeld/MitoSpace4D/simclr/config_debug.yaml', type=str,
                    help='Config path.')

def save_views_side_by_side_with_time(x, filename, n_views=3):
    """
    Creates a side-by-side MIP of the two views across time and save to an image for each channel.
    
    Horizontally: time
    Vertically: views
    Separate image per channel
    x: (b*n_views, t, c, z, h, w)
    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    b_nviews, t, c, z, h, w = x.shape
    b = b_nviews // n_views

    x_np = x.cpu().numpy()
    for batch_idx in range(b):
        for channel_idx in range(c):
            fig, ax = plt.subplots(n_views, t, figsize=(t*3, n_views*3))
            for view_idx in range(n_views):
                for time_idx in range(t):
                    mip = np.max(x_np[batch_idx*n_views + view_idx, time_idx, channel_idx], axis=0)
                    ax[view_idx, time_idx].imshow(mip)
                    ax[view_idx, time_idx].axis('off')
                    if view_idx == 0:
                        ax[view_idx, time_idx].set_title(f'Time {time_idx}')
            plt.tight_layout()
            plt.savefig(filename if filename.endswith('.png') else f'{filename}.png')
            plt.close()

def save_views_side_by_side(x_orig, x_aug, time_idx=0, save_path=None):
    """
    Compare a single time point using MIPs (maximum intensity projections) over Z for:
      - Row 0: Original
      - Row 1: Augmented (first view only)
    Columns are channels (e.g., Channel 0, Channel 1).

    Args:
        x_orig: (b, t, c, z, h, w)  original decoded tensor
        x_aug:  (b*n_views, t, c, z, h, w) augmented tensor with n_views=2 concatenated along batch
        time_idx: int time index to visualize (default 0)
        save_path: directory to save figures into (created if needed)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    n_views = 2
    b, t, c, z, h, w = x_orig.shape
    b_nviews, t2, c2, z2, h2, w2 = x_aug.shape
    assert t2 == t and c2 == c and z2 == z and h2 == h and w2 == w, "x_orig and x_aug shapes must match except batch"
    assert b_nviews == b * n_views, "x_aug must have b*n_views along batch dimension"
    assert 0 <= time_idx < t, f"time_idx {time_idx} out of range [0, {t-1}]"
    
    if save_path is None:
        save_path = "."
    else:
        os.makedirs(save_path, exist_ok=True)

    x_orig_np = x_orig.detach().cpu().numpy()
    x_aug_np  = x_aug.detach().cpu().numpy()

    n_rows = 2  # original + one augmented view
    n_cols = c
    row_labels = ["Original", "Augmented"]

    for batch_idx in range(b):
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3), squeeze=False)

        # Column headers for channels
        for ch in range(c):
            ax[0, ch].set_title(f'Channel {ch}', fontsize=12)

        # Row 0: original at single time point
        for ch in range(c):
            print("original mean: ", np.mean(x_orig_np[batch_idx, time_idx, ch]))
            mip_orig = np.max(x_orig_np[batch_idx, time_idx, ch], axis=0)
            ax[0, ch].imshow(mip_orig)
            ax[0, ch].axis('off')

        # Row 1: first augmented view only
        aug_bi = batch_idx * n_views + 0  # first view
        for ch in range(c):
            print("augmented mean: ", np.mean(x_aug_np[aug_bi, time_idx, ch]))
            mip_aug = np.max(x_aug_np[aug_bi, time_idx, ch], axis=0)
            ax[1, ch].imshow(mip_aug)
            ax[1, ch].axis('off')

        # Tight vertical row labels next to first column
        for row_idx, label in enumerate(row_labels):
            ax[row_idx, 0].text(
                -0.05, 0.5, label,
                fontsize=12,
                rotation=90,
                va='center',
                ha='right',
                transform=ax[row_idx, 0].transAxes
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'batch{batch_idx}_t{time_idx}.png'))
        plt.close()

def main():
    warnings.filterwarnings("ignore")  # supress all warnings
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg_aug = cfg['data_params']['transforms']
    device = torch.device('cuda')

    outdir = "./dumps/"

    assert cfg['training']['n_views'] == 2, "Only two view training is supported. Please use --n-views 2."

    torch.set_float32_matmul_precision('medium')

    dataset = ContrastiveLearningDataset(cfg['data_params']['data_path'], cfg)
    
    train_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                        cfg['training']['n_views'],
                                        flag='train', 
                                        seed=None,
                                        pick_labels=None,
                                        samples_per_drug=cfg['data_params']['samples_per_drug'],
                                        timesteps=cfg['data_params']['timesteps'],
                                        zstacks=cfg['data_params']['zstacks'])
    
    val_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                      cfg['training']['n_views'],
                                      flag='val', 
                                      seed=None,
                                      pick_labels=None,
                                      samples_per_drug=cfg['data_params']['samples_per_drug'],
                                      timesteps=cfg['data_params']['timesteps'],
                                      zstacks=cfg['data_params']['zstacks'])

    train_loader = DataLoader(train_dataset, 
                            #   batch_size=cfg['training']['batch_size'],
                              batch_size=1,
                              shuffle=True,
                              num_workers=cfg['training']['workers'], 
                              pin_memory=True, 
                              drop_last=True,
                              persistent_workers=cfg['training']['persistent_workers'])

    val_loader = DataLoader(val_dataset, 
                            # batch_size=cfg['training']['batch_size'], 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=cfg['training']['workers'], 
                            pin_memory=True, 
                            drop_last=True,
                            persistent_workers=cfg['training']['persistent_workers'])
    
    #-- Decoder
    dec_checkpoint_path = "/u/earkfeld/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
    decoder_model = MitoSpace3DAutoencoder()
    decoder = AutoEncoderRunner.load_from_checkpoint(dec_checkpoint_path, model=decoder_model)
    decoder = decoder.model.decoder
    decoder.eval()
    for param in decoder.parameters(): # Just to be sure
        param.requires_grad = False
    decoder.to(device)

    #-- Augmentation Pipeline
    augmentation_pipeline = DataAugmentation(cfg_aug, zero_mean_norm=True)
    augmentation_pipeline.to(device)

    pbar = tqdm(len(train_loader))
    n_max = 10
    n=0
    for i, batch in enumerate(train_loader):
        x, lbl = batch["images"], batch["classes"]
        x = x.to(device, non_blocking=True)
        filename = osp.join(outdir, f'train_batch{i}_lbl{lbl[0]}.png')
        with torch.no_grad():
            x = decoder(x)              # (b, t, c, z, h, w) original decoded
            x_aug = augmentation_pipeline(x)  # (b*n_views, t, c, z, h, w)

        x_combined = torch.cat([x, x_aug], dim=0)  # (b*(n_views+1), t, c, z, h, w)

        save_views_side_by_side_with_time(x_combined, filename, n_views=3)
        pbar.update(1)
        n += 1
        if n >= n_max:
            break
        # break
    pbar.close()
    print("Done")

    # pbar = tqdm(len(val_loader))
    # for batch in val_loader:
    #     x, lbl = batch["images"], batch["classes"]
    #     x = x.to(device, non_blocking=True)
    #     with torch.no_grad():
    #         x = decoder(x)  # (b*n_views, t, c, z, h, w)
    #         x = augmentation_pipeline(x)  # (b*n_views, t, c, z, h, w)
    #     pbar.update(1)

if __name__ == "__main__":
    main()