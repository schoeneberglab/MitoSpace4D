"""
Created 202620117 by Eric Arkfeld

No Model eval; visualization only using phate for down projection

set up for visualizing 3D embeddings across all frames.
consolidated visualization routines from vis.py
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

    save_dir = osp.join(proj_dir, "../../kinetics_phate_trajectory_images2")
    os.makedirs(save_dir, exist_ok=True)

    # 4: tbhp, 5: h2o2
    pick_labels = list(range(0,26))
    print(f'Using pick labels: {pick_labels}')
    pick_frame = 0
    n_images = 1
    movie_id = 0

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

    for label in pick_labels:
        df_label = df_data[df_data['label'] == label]
        # Use inplace or reassignment for sort to stick
        df_label = df_label.sort_values(by=['region_id'])

        min_region = int(df_label['region_id'].min())
        max_region = int(df_label['region_id'].max())

        label_images = []
        label_name = ""

        for region in range(min_region, max_region + 1):
            df_region = df_label[df_label['region_id'] == region]

            # get the first movies only
            df_region = df_region[df_region['movie_id'] == movie_id]

            # Filter for the specific frame
            candidates = df_region[df_region['frame_id'] == pick_frame]

            if candidates.empty:
                print(f"Skipping region {region}: No frame {pick_frame} found.")
                continue

            # Sample exactly 1 row
            # .index[0] extracts the single scalar index integer
            idx = candidates.sample(n=n_images, random_state=1123).index[0]

            # Use .loc[idx] to get the Series for that specific row
            row = df_data.loc[idx]

            img_path = row['img_path']
            label_name = row['label_name']

            print(f"Processing Region {region}: {img_path}")

            # Load data (ensure string path)
            img_4d = np.load(str(img_path))
            print(f"Loaded image shape: {img_4d.shape}")  # Expecting (C, T, D, H, W)

            img = img_4d[1, 0, ...].max(axis=0)  # MIP: Channel 1, Time 0, project Z

            # Use a viridis colormap for the MIP
            img_color = plt.cm.viridis(img / img.max() + 1.e-12)  # Normalize and apply colormap
            img_color = (img_color[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 RGB

            label_images.append(img_color)

        if label_images:
            combined_img = np.hstack(label_images)
            outfile = f"{label}_{label_name}_combined.png"
            save_path = osp.join(save_dir, outfile)
            imageio.imwrite(save_path, combined_img)
