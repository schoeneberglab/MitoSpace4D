"""
Adding custom coloring by time and instance + tvn support

To-Do's
- Filter legends so only conditions plotted are shown.
- Move image path and time handling to make_mitospace?
"""

import torch
import numpy as np
import os
import random
import yaml
import tqdm

# from umap import UMAP
from cuml.manifold import UMAP
from cuml.metrics import trustworthiness

import argparse
import os.path as osp
import time
import einops

from sklearn.decomposition import PCA

from utils.vis import make_mitospace
# from utils.colormaps import create_colormap # TODO: set up color map generation
from data_aug.dataset_utils import get_mitospace_data_loaders
from train_simclr import SimCLRRunner
import torch.nn.functional as F
from utils.utils import normalize, load_config
from torch.utils.data import DataLoader
from utils.utils import get_drug_label_maps, increase_contrast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from simclr.models_simple import Lightweight3DResNet
# from simclr.models import MitoSpace4DConvLSTM
# from simclr.models_simple_attn import Lightweight3DResNet
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = load_config('/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml')

decoder_ckpt = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
model_ckpt = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitoSpace4D_resnetbilstm_encoded_normal_eps287.ckpt"
# model_ckpt = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_reproc2024_coslr_decoupled-tmrm_r20251027.ckpt"

data_dir = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_summer_new/"

# condition1 = "20240809-1" # oligomycin
# condition2 = "20240809-1" # oligomycin

condition1 = "20240729-1" # control
condition2 = "20240729-1" # control
# condition1 = "20240805-1" # h2o2
# condition2 = "20240805-1" # h2o2
# condition_dir = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_summer_new/20240729-1" # control
# diff_condition_dir = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_summer_new/20240805-1" # mitoq

# Number of random samples to draw from each condition
n_samples = 50

# Build and load model
model = Lightweight3DResNet(embedding_size=2048, 
                            cfg_aug=cfg['data_params']['transforms'], 
                            apply_aug=False, 
                            decoder_checkpoint_path=None
                            # decoder_checkpoint_path=decoder_ckpt
                            )
model = SimCLRRunner.load_from_checkpoint(model_ckpt, model=model, cfg=cfg, strict=False).model
model.eval().to(device)

for param in model.parameters():
    param.requires_grad = False

# Collect distances across samples
all_cosine_dist = []
all_cosine_dist_single = []
all_cosine_dist_diff = []
all_cosine_dist_swap = []
all_cosine_dist_swap_other = []
all_cosine_dist_half_swap = []
all_cosine_dist_half_swap_other = []
all_cosine_dist_single_repeat = []

for i in range(n_samples):
    # Pick two random files each condition dir
    indir1 = osp.join(data_dir, condition1)
    indir2 = osp.join(data_dir, condition2)
    infile = osp.join(indir1, np.random.choice(os.listdir(indir1)))
    infile_diff = osp.join(indir2, np.random.choice(os.listdir(indir2)))

    print(infile)
    print(infile_diff)

    im = np.load(infile)
    im_diff = np.load(infile_diff)

    # Convert to tensors
    im = torch.from_numpy(im).unsqueeze(0)  # (1, C, T, D, H, W)
    im_diff = torch.from_numpy(im_diff).unsqueeze(0)  # (1, C, T, D, H, W)

    im_normal = im.permute(0, 2, 1, 3, 4, 5).contiguous() # (B, T, C, D, H, W)
    im_diff = im_diff.permute(0, 2, 1, 3, 4, 5).contiguous() # (B, T, C, D, H, W)

    # Create a new version with the last frame repeated over the time dimension
    im_repeated = im_normal.clone()
    im_repeated[:, 1:, ...] = im_normal[:, -1:, ...].repeat(1, im_normal.shape[1]-1, 1, 1, 1, 1) 

    # Create a single frame version with the last frame only
    im_single = im_normal[:, -1:, ...]  # (B, 1, C, D, H, W)

    # Create versions where the last frame is swapped between samples (both directions)
    im_swapped_last = im_normal.clone()
    im_swapped_last[:, -1, ...] = im_diff[:, -1, ...]  # normal gets diff's last frame

    im_swapped_last_other = im_diff.clone()
    im_swapped_last_other[:, -1, ...] = im_normal[:, -1, ...]  # diff gets normal's last frame

    # Create versions where HALF of the time steps are swapped between samples (double swap on half)
    T = im_normal.shape[1]
    half = T // 2
    im_half_swap = im_normal.clone()
    im_half_swap_other = im_diff.clone()
    im_half_swap[:, :half, ...] = im_diff[:, :half, ...]            # normal's first half <- diff's first half
    im_half_swap_other[:, :half, ...] = im_normal[:, :half, ...]    # diff's first half   <- normal's first half

    ims = [im_normal, im_repeated, im_single, im_diff, im_swapped_last, im_swapped_last_other, im_half_swap, im_half_swap_other]
    im_features = []
    # Get embeddings
    with torch.no_grad():
        for im_batch in ims:
            features, _ = model(im_batch.to(device)) # model expects (B, T, C, D, H, W)
            features = F.normalize(features, dim=-1)  # (B, 2048) or (B, T, 2048)
            features = features[:, -1, :]  # Take last time step only
            im_features.append(features)

    # Calculate the cosine distance between the two sets of embeddings
    cosine_dist = 1 - F.cosine_similarity(im_features[0], im_features[1])
    cosine_dist_single = 1 - F.cosine_similarity(im_features[0], im_features[2])
    cosine_dist_diff = 1 - F.cosine_similarity(im_features[0], im_features[3])
    cosine_dist_swap = 1 - F.cosine_similarity(im_features[0], im_features[4])
    cosine_dist_swap_other = 1 - F.cosine_similarity(im_features[3], im_features[5])
    cosine_dist_half_swap = 1 - F.cosine_similarity(im_features[0], im_features[6])
    cosine_dist_half_swap_other = 1 - F.cosine_similarity(im_features[3], im_features[7])
    cosine_dist_single_repeat = 1 - F.cosine_similarity(im_features[1], im_features[2])

    # Each is shape (B=1,), store scalars
    all_cosine_dist.append(cosine_dist.detach().cpu().item())
    all_cosine_dist_single.append(cosine_dist_single.detach().cpu().item())
    all_cosine_dist_diff.append(cosine_dist_diff.detach().cpu().item())
    all_cosine_dist_swap.append(cosine_dist_swap.detach().cpu().item())
    all_cosine_dist_swap_other.append(cosine_dist_swap_other.detach().cpu().item())
    all_cosine_dist_half_swap.append(cosine_dist_half_swap.detach().cpu().item())
    all_cosine_dist_half_swap_other.append(cosine_dist_half_swap_other.detach().cpu().item())
    all_cosine_dist_single_repeat.append(cosine_dist_single_repeat.detach().cpu().item())

# Convert to tensors for stats
all_cosine_dist_t = torch.tensor(all_cosine_dist)
all_cosine_dist_single_t = torch.tensor(all_cosine_dist_single)
all_cosine_dist_diff_t = torch.tensor(all_cosine_dist_diff)
all_cosine_dist_swap_t = torch.tensor(all_cosine_dist_swap)
all_cosine_dist_swap_other_t = torch.tensor(all_cosine_dist_swap_other)
all_cosine_dist_half_swap_t = torch.tensor(all_cosine_dist_half_swap)
all_cosine_dist_half_swap_other_t = torch.tensor(all_cosine_dist_half_swap_other)
all_cosine_dist_single_repeat_t = torch.tensor(all_cosine_dist_single_repeat)

mean_cd = all_cosine_dist_t.mean().item()
std_cd = all_cosine_dist_t.std(unbiased=False).item()

mean_cd_single = all_cosine_dist_single_t.mean().item()
std_cd_single = all_cosine_dist_single_t.std(unbiased=False).item()

mean_cd_diff = all_cosine_dist_diff_t.mean().item()
std_cd_diff = all_cosine_dist_diff_t.std(unbiased=False).item()

mean_cd_swap = all_cosine_dist_swap_t.mean().item()
std_cd_swap = all_cosine_dist_swap_t.std(unbiased=False).item()

mean_cd_swap_other = all_cosine_dist_swap_other_t.mean().item()
std_cd_swap_other = all_cosine_dist_swap_other_t.std(unbiased=False).item()

mean_cd_half_swap = all_cosine_dist_half_swap_t.mean().item()
std_cd_half_swap = all_cosine_dist_half_swap_t.std(unbiased=False).item()

mean_cd_half_swap_other = all_cosine_dist_half_swap_other_t.mean().item()
std_cd_half_swap_other = all_cosine_dist_half_swap_other_t.std(unbiased=False).item()

mean_cd_single_repeat = all_cosine_dist_single_repeat_t.mean().item()
std_cd_single_repeat = all_cosine_dist_single_repeat_t.std(unbiased=False).item()

print(f"    original and repeat last: {mean_cd:.6f} ± {std_cd:.6f}")
print(f"   original and single frame: {mean_cd_single:.6f} ± {std_cd_single:.6f}")
print(f"original and diff embeddings: {mean_cd_diff:.6f} ± {std_cd_diff:.6f}")
print(f" original and swapped last T: {mean_cd_swap:.6f} ± {std_cd_swap:.6f}")
print(f"      diff and swapped last T: {mean_cd_swap_other:.6f} ± {std_cd_swap_other:.6f}")
print(f" original and half-time swap: {mean_cd_half_swap:.6f} ± {std_cd_half_swap:.6f}")
print(f"      diff and half-time swap: {mean_cd_half_swap_other:.6f} ± {std_cd_half_swap_other:.6f}")
print(f"      original and single repeat: {mean_cd_single_repeat:.6f} ± {std_cd_single_repeat:.6f}")

