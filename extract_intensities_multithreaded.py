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
import pandas as pd

from umap import UMAP
# from cuml.manifold import UMAP
# from cuml.metrics import trustworthiness

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
from utils.utils import normalize, load_config, get_fpaths
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# from simclr.models_simple_exp import Lightweight3DResNet # Getting resnet feats too
# from simclr.models_simple import Lightweight3DResNet # newly trained model (some dict changes)
from simclr.models_simple_3d import Lightweight3DResNet # 3D resnet model
# from simclr.models_simple_original import Lightweight3DResNet
#
# from simclr.models import MitoSpace4DConvLSTM
# from simclr.models_simple_attn import Lightweight3DResNet
from utils.tvn import *
import joblib

import os.path as osp
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from autoencoder.ae_util import AEUtil


np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('--checkpoint_path', help='Checkpoint path', 
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitospaceResnetBiLSTM_Summer2024.ckpt"
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_decoupled-tmrm_r20251115_epoch=145-step=25988-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_ablated-tmrm_r20251115_epoch=161-step=28836-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_decoupled-tmrm_r20251115_epoch=256-step=41120-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_ablated-tmrm_r20251115_epoch=291-step=46720-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2-161eps-ckpt_kinetics_ablated-tmrm_r20260105.ckpt",
                    default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnet3d_encoded_kinetics_ablated-tmrm_singleframe_r20260108.ckpt"
                    )

parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml', type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict',
                    # default="/home/earkfeld/Projects/MitoSpace4D/data/2024v2_encoded_data",
                    # default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v2_data/processed_data", # 2024v2 Dataset
                    # default="/mnt/DATA_02/2024_data_encoded", # 2024v2 Dataset Encoded
                    default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/2025_kinetics_data/processed_data", # Kinetics Dataset
                    # default="/home/earkfeld/Projects/MitoSpace4D/data/2025_kinetics_encoded_data", # Kinetics Dataset Encoded
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data"
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_3b"
                    # default="/run/user/1002/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/leukemia_drug_resistance_data")
                    # default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_4",
                    # default="/home/earkfeld/Projects/MitoSpace4D/data/2025_kinetics_encoded_data"
                    )

parser.add_argument('--decoder_ckpt', help='Path to decoder checkpoint',
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
                    default=None,
                    )

parser.add_argument('--embeddings_dir', help='Directory to save/load embeddings', default=None)

parser.add_argument('--visualize', default=False, action='store_true', help="Visualize UMAP'd MitoSpace embeddings")
parser.add_argument('--save_embeddings', default=False, action='store_true', help='Save embeddings')
parser.add_argument('--save_pcd', default=None, help='Path to save the point cloud')

parser.add_argument('--single_frames', default=False, action='store_true', help='Generates frame embeddings independently.')
parser.add_argument('--labels', default=None, type=int, nargs='+', help='Labels to pick. Default is all labels.')
parser.add_argument('--datasets', default=None, type=str, nargs='+', help='Datasets to use. Default is all datasets.')
parser.add_argument('--cmap', default='label', help='Color map to use.', choices=['label', 'temporal', 'region', 'dataset', 'tmrm'])
parser.add_argument('--to_load', default='all', type=str, choices=["all", "train", "val"], help='Which splits to load. Default is "all".')
# parser.add_argument('--channels', default=[0, 1], type=int, nargs='+', help='Channels to use. Default is [0, 1].')

parser.add_argument('--reproject', default=False, action='store_true', help='Reproject embeddings')
parser.add_argument('--load_transform', default=False, action='store_true', help='Load UMAP transform from disk instead of fitting new one.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dataloaders')
parser.add_argument('--use_pca', default=False, action='store_true', help='Use PCA for dimensionality reduction before UMAP. Default is False.')
parser.add_argument('--densmap', default=False, action='store_true', help='Use densMAP instead of UMAP. Default is False.')
parser.add_argument('--tvn', default=False, action='store_true', help='Apply Typical Variation Normalization (TVN) using controls before UMAP.')  # <-- added flag
parser.add_argument('--control_label', default='control', type=str, help='Label name for the control samples for performing typical variation normalization (TVN). Default is "control".')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.multiprocessing.set_sharing_strategy('file_system')

# def get_otsu_intensity(img, mask_ch=0, measure_ch=0):
#     # (B, T, C, Z, Y, Z)
#     img = img.cpu().numpy()
#     raw_mean_intensities = np.zeros(img.shape[1])
#     otsu_mean_intensities = np.zeros(img.shape[1])
#
#     for t in range(img.shape[1]):
#         # Otsu threshold the morphology channel get a mask
#         thr = threshold_otsu(img[0, t, mask_ch, ...])
#         mask = img[0, t, 1, ...] > thr
#
#         # Raw mean intensity of the tmrm channel
#         raw_mean_intensities[t] = img[0, t, measure_ch, ...].mean()
#
#         # Masked mean intensity of the tmrm channel
#         img[0, t, measure_ch, ...] = img[0, t, measure_ch, ...] * mask
#         otsu_mean_intensities[t] = img[0, t, measure_ch, ...].mean()
#
#     return raw_mean_intensities, otsu_mean_intensities

def get_intensities(img, mask_ch=1):
    # (T, C, Z, Y, Z)
    morph_intensities = []
    tmrm_intensities = []

    for t in range(img.shape[0]):
        # Otsu threshold the mask_ch get a mask
        thr = threshold_otsu(img[t, mask_ch, ...])
        mask = img[t, 1, ...] > thr

        img[t, 0, ...] = img[t, 0, ...] * mask
        img[t, 1, ...] = img[t, 1, ...] * mask

        morph_intensities.append(img[t, 0, ...].mean())
        tmrm_intensities.append(img[t, 1, ...].mean())

    return morph_intensities, tmrm_intensities

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Update the cfg with the args (
    cfg.update(vars(args))

    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"

    embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_252eps'
    os.makedirs(embeddings_dir, exist_ok=True)

    img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')

    # Set up image paths
    df = pd.read_csv(img_pathfile, header=None)
    # name the column "morph_path"
    df.columns = ['morph_path']

    # Add a new column "tmrm_path" by replacing the morph_path with the tmrm path
    df['tmrm_path'] = df['morph_path'].apply(lambda x: x.replace('-0-1.npy', '-0-0.npy'))

    # Initialize the columns for the intensities as series objects
    df['morph_intensities'] = pd.Series(object)
    df['tmrm_intensities'] = pd.Series(object)

    morph_intensity_path = osp.join(embeddings_dir, 'morph_intensities.npy') # (N, T) float32
    tmrm_intensity_path = osp.join(embeddings_dir, 'tmrm_intensities.npy') # (N, T) float32

    for i, row in df.iterrows():
        morph_path = row['morph_path']
        tmrm_path = row['tmrm_path']

        morph_img = np.load(morph_path) # (T, Z, Y, X)
        tmrm_img = np.load(tmrm_path) # (T, Z, Y, X)

        # Concatenate along new channel dimension to get (T, C, Z, Y, X)
        morph_img = np.expand_dims(morph_img, axis=1) # (T, 1, Z, Y, X)
        tmrm_img = np.expand_dims(tmrm_img, axis=1) # (T, 1, Z, Y, X)
        img = np.concatenate([tmrm_img, morph_img], axis=1) # (T, 2, Z, Y, X); consistent with original channel ordering

        morph_intensities, tmrm_intensities = get_intensities(morph_img, mask_ch=1)

        df.at[i, 'morph_intensities'] = morph_intensities
        df.at[i, 'tmrm_intensities'] = tmrm_intensities

    # Save the intensities as numpy arrays
    morph_intensities_array = np.stack(df['morph_intensities'].values)
    tmrm_intensities_array = np.stack(df['tmrm_intensities'].values)

    np.save(morph_intensity_path, morph_intensities_array)
    np.save(tmrm_intensity_path, tmrm_intensities_array)

    # Save the dataframe with the paths and intensities as a parquet file for easier loading later
    df.to_parquet(osp.join(embeddings_dir, 'mean_intensities.parquet'))