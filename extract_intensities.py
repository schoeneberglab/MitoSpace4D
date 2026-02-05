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

def get_label_colormap(proj_dir):
    colors = {}
    with open(f"{proj_dir}/extraction_utils/colors.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:
                date, label, index, r, g, b = parts
                if float(r) >= 1 or float(g) >= 1 or float(b) >= 1:
                    colors[int(index)] = [float(r) / 255, float(g) / 255, float(b) / 255]
                else:
                    colors[int(index)] = [float(r), float(g), float(b)]
            else:
                print("Invalid line format:", line)
    return colors

def get_temporal_colormap(embeddings_dir):
    print("Loading temporal colormap")
    # colors = np.load(f"{embeddings_dir}/temporal_colormap.npy")
    colors = np.load(f"{embeddings_dir}/cmap_temporal.npy")
    return colors

def get_region_colormap(embeddings_dir):
    print("Loading region colormap")
    colors = np.load(f"{embeddings_dir}/cmap_region.npy")
    return colors

def get_dataset_colormap(embeddings_dir):
    print("Loading dataset colormap")
    colors = np.load(f"{embeddings_dir}/cmap_dataset.npy")
    return colors

def get_tmrm_colormap(embeddings_dir):
    print("Loading TMRM colormap")
    colors = np.load(f"{embeddings_dir}/cmap_tmrm.npy")

    # set up a viridis colormap for visualization from the normalized intensities
    cmap = plt.get_cmap('viridis')
    colors = cmap(colors)[:, :3]  # get RGB values only, 0-1 range
    return colors

def perform_tvn(embeddings, labels, img_names, label_names, control_label='wt_cal27', scope="global", eps=1e-6):
    assert scope in ['global', 'batch'], "Scope must be 'global' or 'batch'"
    # Setting up dataframe for metadata
    plates = [x.split('/')[-2].split('-')[0] for x in img_names]
    print("Unique plates:", np.unique(plates))
    df_data = pd.DataFrame()
    df_data['img_name'] = img_names
    df_data['plate'] = plates
    df_data['label'] = labels
    df_data['condition'] = [label_names[int(x)] for x in labels]

    control_mask_arr = np.array(df_data['condition'] == control_label)

    # Run TVN
    if scope == "global":
        print(f"Performing global TVN using {control_mask_arr.sum()} control samples ({control_label})...")
        normalized_embeddings = tvn_global(X=embeddings,
                                          controls_mask=control_mask_arr,
                                          eps=eps,
                                          ledoit_wolf=True)
    else:  # per-batch
        print(f"Performing per-batch TVN using {control_mask_arr.sum()} control samples ({control_label})...")
        normalized_embeddings = tvn_per_batch(X=embeddings,
                                            meta=df_data,
                                            batch_col='plate',
                                            controls_mask=control_mask_arr,
                                            eps=eps,
                                            ledoit_wolf=True)
    return normalized_embeddings

def get_intensity(img, mask_ch=0, measure_ch=0):

    # (B, T, C, Z, Y, Z)
    img = img.cpu().numpy()
    raw_mean_intensities = np.zeros(img.shape[1])
    otsu_mean_intensities = np.zeros(img.shape[1])

    for t in range(img.shape[1]):
        # Otsu threshold the morphology channel get a mask
        thr = threshold_otsu(img[0, t, mask_ch, ...])
        mask = img[0, t, 1, ...] > thr

        # Raw mean intensity of the tmrm channel
        raw_mean_intensities[t] = img[0, t, measure_ch, ...].mean()

        # Masked mean intensity of the tmrm channel
        img[0, t, measure_ch, ...] = img[0, t, measure_ch, ...] * mask
        otsu_mean_intensities[t] = img[0, t, measure_ch, ...].mean()

    return raw_mean_intensities, otsu_mean_intensities

if __name__ == '__main__':
    # TODO: Set up to read mostly from config, optionally overwrite w/ args, add specific visualization entries, and copy updated to the embedding dir
    # Goal: be able to just provide embeddings dir and use previous settings stored there in a copy of the config file
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Update the cfg with the args (
    cfg.update(vars(args))

    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    save_dir = f"{proj_dir}/runs/"

    # data_path = cfg['data_params']['data_path']
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_20250828')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_combined_r20250905')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_20250811')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_r20250920')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_r20250929_10frames')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_r20251002_single_frames') 
    # embeddings_dir = osp.join(save_dir, 'embeddings_leukemia_r20251014_all_frames')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_ds20251009-20251010_r20251015_all_frames')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_r20251016_10frames')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_r20251016_10frames_mtg-only')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_r20251025_ft-kinetics-37eps_10frames')
    
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_ft-kinetics-50eps_decoupled-tmrm_r20251028')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_ft-kinetics-50eps_decoupled-tmrm_single-frames_r20251027')

    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_model2024v2-decoupled-tmrm_r20251104')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_debug_eps149_r20251109')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_pten_trial3_ft-kinetics-50eps_decoupled-tmrm_r20251112')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_debug_no-geometric-augs_10eps_r20251114')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_debug_all-augs_10eps_r20251114')

    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_decoupled-tmrm_r20251117')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics_ablated-tmrm_r20251117')

    # embeddings_dir = osp.join(save_dir, 'embeddings_2024v2_decoupled-tmrm_eps145_r20251119')
    # embeddings_dir = osp.join(save_dir, 'embeddings_2024v2-encoded_ablated-tmrm_eps162_r20251120')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics-encoded_ablated-tmrm_eps291_r20251120')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics-encoded_decoupled-tmrm_eps256_r20251120')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics-encoded_2024v2-model_ablated-tmrm_eps162_r20251124')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer-resistance-trial3b_2024v2-model_ablated-tmrm_eps162_r20251204')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220')
    # embeddings_dir = osp.join(save_dir, "exp0_modified_embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220")
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260108_kinetics_morphology_resnet_embeddings_val-set"
    # embeddings_dir = osp.join(save_dir, '20260108_kinetics_morphology_resnet_embeddings_all')
    # embeddings_dir = osp.join(save_dir, '20260110_kinetics_morphology_resnet_embeddings_val-set_tscrambled')
    # embeddings_dir = osp.join(save_dir, 'tmp')

    # embeddings_dir = osp.join(save_dir, '20260108_kinetics_embeddings_tscrambled_2024v2-model_morphology_only')
    # embeddings_dir = osp.join(save_dir, '20260108_kinetics_morphology_resnet_embeddings_all')

    # embeddings_dir = osp.join("/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classifier/deepprofiler_features/PTEN_deepprofiler_pooled-clones")
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classifier/deepprofiler_features/2024v2/2024v2_deepprofiler"

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-val-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-all-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm"

    embeddings_dir = osp.join(save_dir, '20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm')
    # embeddings_dir = osp.join(save_dir, '20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm')
    os.makedirs(embeddings_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    print(f"Ckpt Path: {args.checkpoint_path}")
    print(f"Data Path: {args.data_path}")

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(osp.join(proj_dir, "extraction_utils/drugs_to_labels.txt"), 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    labels = [args.labels] if args.labels else [list(drug_labels_dict.values())]  # Default to all conditions in the drug label dict

    batch_size = args.batch_size

    morph_intensity_path = osp.join(embeddings_dir, 'morph_intensities.npy')  # (N, T) float32

    data_paths = [args.data_path]
    loaders = []
    for data_path, pick_label in zip(data_paths, labels):
        loaders.append(
            get_mitospace_data_loaders(
                data_path,
                shuffle=False,
                batch_size=batch_size,
                to_load=[args.to_load],
                seed=None,
                pick_labels=pick_label,
                samples_per_drug=None,
                num_workers=8,
            )[args.to_load]
        )


    morph_intensities_list = []
    for loader_idx, loader in enumerate(loaders):
        pbar = tqdm.tqdm(total=len(loader))
        for i, batch in enumerate(loader):
            if isinstance(batch, list):
                im, lbl, img_pth = batch[0], batch[1], batch[2]
            else:
                im, lbl, img_pth = batch["images"], batch["classes"], batch["image_paths"]

            _, otsu_intensity = get_intensity(im, mask_ch=1, measure_ch=1) # morph, morph
            morph_intensities_list.append(otsu_intensity)
            pbar.update(1)
    np.save(morph_intensity_path, np.asarray(morph_intensities_list, dtype=np.float32))