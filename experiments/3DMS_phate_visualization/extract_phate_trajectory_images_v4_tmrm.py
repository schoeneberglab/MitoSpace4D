"""
Created 202620117 by Eric Arkfeld

No Model eval; visualization only using phate for down projection

set up for visualizing 3D embeddings across all frames.
consolidated visualization routines from vis.py

extracting images using the global time index
"""

import torch
import random
import open3d as o3d
import imageio

from cuml.manifold import umap
# from cuml.metrics import trustworthiness

import os.path as osp

# from utils.colormaps import create_colormap # TODO: set up color map generation
from utils.utils import load_config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.feature_trajectory import generate_phate_trajectories

# from simclr.models_simple_original import Lightweight3DResNet

# from simclr.models import MitoSpace4DConvLSTM
# from simclr.models_simple_attn import Lightweight3DResNet
from utils.tvn import *
import phate
from skimage.filters import threshold_otsu

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('--checkpoint_path', help='Checkpoint path',
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitospaceResnetBiLSTM_Summer2024.ckpt"
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_decoupled-tmrm_r20251115_epoch=145-step=25988-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_ablated-tmrm_r20251115_epoch=161-step=28836-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_decoupled-tmrm_r20251115_epoch=256-step=41120-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_ablated-tmrm_r20251115_epoch=291-step=46720-val_loss=0.00.ckpt",
                    default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2-161eps-ckpt_kinetics_ablated-tmrm_r20260105.ckpt"
                    )

parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml', type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict',
                    # default="/home/earkfeld/Projects/MitoSpace4D/data/2024v2_encoded_data",
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_summer_new/", # 2024v2 Dataset
                    # default="/mnt/DATA_02/2024_data_encoded", # 2024v2 Dataset Encoded
                    default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/2025_kinetics_data/processed_data", # Kinetics Dataset
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_data_encoded/", # Kinetics Dataset Encoded
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data"
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_3b"
                    # default="/run/user/1002/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/leukemia_drug_resistance_data")
                    #     default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_4",
                    # default="/home/earkfeld/Projects/MitoSpace4D/data/2025_kinetics_encoded_data"
                    )

parser.add_argument('--decoder_ckpt', help='Path to decoder checkpoint',
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
                    default=None,
                    )

parser.add_argument('--embeddings_dir', help='Directory to save/load embeddings', default=None)

parser.add_argument('--visualize', default=False, action='store_true', help="Visualize UMAP'd MitoSpace embeddings")
parser.add_argument('--labels', default=None, type=int, nargs='+', help='Labels to pick. Default is all labels.')
parser.add_argument('--cmap', default='label', help='Color map to use.', choices=['label', 'time', 'region', 'dataset', 'tmrm'])

parser.add_argument('--reproject', default=False, action='store_true', help='Reproject embeddings')
parser.add_argument('--load_transform', default=False, action='store_true', help='Load UMAP transform from disk instead of fitting new one.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dataloaders')
parser.add_argument('--use_pca', default=False, action='store_true', help='Use PCA for dimensionality reduction before UMAP. Default is False.')
parser.add_argument('--densmap', default=False, action='store_true', help='Use densMAP instead of UMAP. Default is False.')
parser.add_argument('--tvn', default=False, action='store_true', help='Apply Typical Variation Normalization (TVN) using controls before UMAP.')  # <-- added flag
parser.add_argument('--control_label', default='control', type=str, help='Label name for the control samples for performing typical variation normalization (TVN). Default is "control".')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # TODO: Set up to read mostly from config, optionally overwrite w/ args, add specific visualization entries, and copy updated to the embedding dir
    # Goal: be able to just provide embeddings dir and use previous settings stored there in a copy of the config file
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.update(vars(args))

    proj_dir = "//"

    # Kinetics 3D
    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm/"

    # Kinetics 4D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_kinetics-4D-embeddings_2024v2-161eps_ablated-tmrm"

    # 2024v2 3D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"

    # 2024v2 4D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-4D-embeddings_2024v2-161eps_ablated-tmrm"

    # metadata_file = "/home/earkfeld/Projects/MitoSpace4D/experiments/3DMS_phate_visualization/phate_2024v2_metadata.csv"
    metadata_file = "/experiments/3DMS_phate_visualization/metadata/phate_kinetics_metadata.csv"

    # 4: tbhp, 5: h2o2
    pick_labels = list(range(0,26))
    # pick_labels = [5]

    print(f'Using pick labels: {pick_labels}')
    sample_interval = 5
    n_images = 10
    seed=11235

    save_dir = osp.join(proj_dir, f"kinetics_phate_trajectory_images+tmrm_globalnorm_n{n_images}_stride{sample_interval}")
    os.makedirs(save_dir, exist_ok=True)

    with open(osp.join(save_dir, "sampling_info.txt"), "w") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"sample_interval: {sample_interval}\n")
        f.write(f"n_images: {n_images}\n")
        f.write(f"pick_labels: {pick_labels}\n")

    frames_per_region = 60 if "kinetics" in metadata_file else 20
    frames_per_movie = 20

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(osp.join(proj_dir, "../../extraction_utils/drugs_to_labels.txt"), 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    print(f"Loading phate data from {osp.join(embeddings_dir, 'phate_visualization.parquet')}...")

    print(f"Generating phate data for {embeddings_dir}...")
    embeddings = np.load(osp.join(embeddings_dir, 'embeddings_raw.npy'))   # (N, 2048) float32
    labels = np.load(osp.join(embeddings_dir, 'labels.npy'))           # (N,) int32
    img_paths = np.loadtxt(osp.join(embeddings_dir, 'image_paths.csv'), dtype=str).tolist()      # (N,) object
    tmrm_intensities = np.load(osp.join(embeddings_dir, 'tmrm_intensities.npy'))

    df_data = pd.DataFrame({
        "embedding": list(embeddings),
        "label": labels,
        "img_path": img_paths,
        "tmrm": list(tmrm_intensities)
    })
    df_data['path_id'] = df_data['img_path'].apply(lambda x: x.split("ed_data/")[-1])

    df_metadata = pd.read_csv(metadata_file)
    meta_subset = df_metadata[['path', 'region id', 'movie id']].rename(
        columns={'region id': 'region_id', 'movie id': 'movie_id'}
    )

    df_data = df_data.merge(
        meta_subset,
        left_on='path_id',
        right_on='path',
        how='left'
    )

    # df_data = df_data.explode("embedding")
    df_data = df_data.explode(["embedding", "tmrm"])
    df_data = df_data.reset_index(drop=True)

    df_data['frame_id'] = df_data.groupby("img_path").cumcount()
    df_data['time'] = (df_data['region_id'] * frames_per_region) + (df_data['movie_id'] * frames_per_movie) + df_data['frame_id']
    df_data['label_name'] = df_data['label'].map(label_drug_dict)

    sample_times = list(np.arange(0, n_images * sample_interval, sample_interval))
    print(f"Sample times: {sample_times}")

    data_by_label = {}

    # Calculate global max from the loaded TMRM intensities provided in the .npy file
    global_tmrm_max = df_data['tmrm'].max()
    print(f"Global TMRM normalization factor (from file): {global_tmrm_max}")

    print("Collecting images...")
    for label in pick_labels:
        df_label = df_data[df_data['label'] == label]

        # Sort by time
        df_label = df_label.sort_values(by=['time'])

        label_data = []

        for sample_time in sample_times:
            candidates = df_label[df_label['time'] == sample_time]

            if candidates.empty:
                print(f"Skipping label {label}: No frame {sample_time} found.")
                continue

            # Pick a random row from the sampled group
            row = candidates.sample(n=1, random_state=seed).iloc[0]

            img_path = row['img_path']
            label_name = row['label_name']

            print(f"Processing Label {label}, Time {row['time']}: {img_path}")

            # Load data (ensure string path)
            img_4d = np.load(str(img_path))

            # Use specific frame from the 4D/5D array
            frame_idx = row['frame_id']

            # Channel 1: Structure (MIP) - Local Normalization
            img_struct = img_4d[1, frame_idx, ...].max(axis=0)  # MIP: Channel 1, Time frame_idx, project Z

            # Channel 0: TMRM (MIP) - Global Normalization
            img_tmrm = img_4d[0, frame_idx, ...].max(axis=0)    # MIP: Channel 0

            # Set up mask using otsu threshold
            # mask = img_struct > threshold_otsu(img_struct)

            # max_tmrm = img_tmrm[mask].mean()
            # max_tmrm = img_tmrm[mask].max()

            # global_tmrm_max = max(global_tmrm_max, max_tmrm)

            label_data.append({
                'struct': img_struct,
                'tmrm': img_tmrm,
                'label_name': label_name
            })

        data_by_label[label] = label_data

    print(f"Global TMRM Max: {global_tmrm_max}")

    print("Generating images...")
    for label, items in data_by_label.items():
        if not items:
            continue

        label_name = items[0]['label_name']
        struct_images = []
        tmrm_images = []

        for item in items:
            # Structure Channel (Viridis, Local Norm)
            img = item['struct']
            img_max = img.max()
            if img_max > 0:
                img_norm = img / img_max
            else:
                img_norm = img

            img_color = plt.cm.viridis(img_norm)
            img_color = (img_color[:, :, :3] * 255).astype(np.uint8)
            struct_images.append(img_color)

            # TMRM Channel (Hot, Global Norm)
            img_t = item['tmrm']
            img_t_norm = img_t

            img_t_color = plt.cm.hot(img_t_norm)
            img_t_color = (img_t_color[:, :, :3] * 255).astype(np.uint8)
            tmrm_images.append(img_t_color)

        if struct_images:
            # First row: Structure
            row1 = np.hstack(struct_images)
            # Second row: TMRM
            row2 = np.hstack(tmrm_images)

            combined_img = np.vstack([row1, row2])

            outfile = f"{label}_{label_name}_combined.png"
            save_path = osp.join(save_dir, outfile)
            imageio.imwrite(save_path, combined_img)
            print(f"Saved {save_path}")
