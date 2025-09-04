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
import umap
import argparse
import os.path as osp
import time
import einops

from sklearn.decomposition import PCA

from utils.vis import make_mitospace
from utils.setup_temporal_colormap import create_colormap
from data_aug.dataset_utils import get_mitospace_data_loaders
from train_simclr import SimCLRRunner
import torch.nn.functional as F
from utils.utils import normalize, load_config, get_fpaths
from torch.utils.data import DataLoader
from utils.utils import get_drug_label_maps, increase_contrast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from simclr.models_simple import Lightweight3DResNet
# from simclr.models import MitoSpace4DConvLSTM
# from simclr.models_simple_attn import Lightweight3DResNet
from utils.tvn import tvn_global

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--checkpoint_path', help='Checkpoint path', default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitoSpace4D_resnetbilstm_encoded_normal_eps287.ckpt")
parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml', type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict', default="/run/user/1002/gvfs/smb-share:server=aquila0.jslab.ucsd.edu,share=ssd_processing/Others/MitoSpace4D/2025_summer")
# parser.add_argument('--embeddings_dir', help='Directory to save/load embeddings', default=None)

parser.add_argument('--visualize', default=False, action='store_true', help="Visualize UMAP'd MitoSpace embeddings")
parser.add_argument('--save_embeddings', default=False, action='store_true', help='Save embeddings')
parser.add_argument('--save_pcd', default=None, help='Path to save the point cloud')
parser.add_argument('--single_frames', default=False, action='store_true', help='Generates frame embeddings independently.')

parser.add_argument('--labels', default=None, type=int, nargs='+', help='Labels to pick. Default is all labels.')
parser.add_argument('--datasets', default=None, type=str, nargs='+', help='Datasets to use. Default is all datasets.')
parser.add_argument('--cmap', default='label', help='Color map to use.', choices=['label', 'time', 'region', 'dataset'])

parser.add_argument('--reproject', default=False, action='store_true', help='Reproject embeddings')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dataloaders')
parser.add_argument('--use_pca', default=False, action='store_true', help='Use PCA for dimensionality reduction before UMAP. Default is False.')
parser.add_argument('--densmap', default=False, action='store_true', help='Use densMAP instead of UMAP. Default is False.')
parser.add_argument('--use_tvn', default=False, action='store_true', help='Apply Typical Variation Normalization (TVN) using controls before UMAP.')  # <-- added flag

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

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    
    # save_dir = f"{proj_dir}/runs/"
    save_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/"
    
    embeddings_dir = osp.join(save_dir, 'embeddings_cancer_20250828')
    # embeddings_dir = osp.join(save_dir, 'embeddings_cancer_20250811')
    # embeddings_dir = osp.join(save_dir, 'embeddings_kinetics')
    
    os.makedirs(embeddings_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    image_paths = None  # define upfront; populated only in visualize pass or left None

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(osp.join(proj_dir, "extraction_utils/drugs_to_labels.txt"), 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug
    
    labels = [args.labels] if args.labels else [list(drug_labels_dict.values())]  # Default to all conditions in the drug label dict
    # labels = [[27]]

    batch_size = args.batch_size
    t_slice = 0
    z_slice = 30

    if args.cmap == 'time':
        colors = get_temporal_colormap(embeddings_dir)
    elif args.cmap == 'region':
        colors = get_region_colormap(embeddings_dir)
    elif args.cmap == 'dataset':
        colors = get_dataset_colormap(embeddings_dir)
    else:
        colors = get_label_colormap(proj_dir)  # Default to label colormap

    emb_raw_path = osp.join(embeddings_dir, 'embeddings_raw.npy')   # (N, 2048) float32
    lbl_path     = osp.join(embeddings_dir, 'labels.npy')           # (N,) int32
    img_path     = osp.join(embeddings_dir, 'images.npy')           # (N, C, H, W) float16 (init later)
    img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')      # (N,) object
    img_times    = osp.join(embeddings_dir, 'image_times.npy')      # (N,) int32

    if args.save_embeddings:

        # Build and load model
        model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'], apply_aug=False)
        model = SimCLRRunner.load_from_checkpoint(checkpoint_path, model=model, cfg=cfg).model
        model.eval().to(device)

        for param in model.parameters():
            param.requires_grad = False

        data_paths = [args.data_path]
        loaders = []
        for data_path, pick_label in zip(data_paths, labels):
            loaders.append(
                get_mitospace_data_loaders(
                    data_path,
                    shuffle=False,
                    batch_size=batch_size,
                    to_load=["all"],
                    seed=None,
                    pick_labels=pick_label,
                    samples_per_drug=None,
                )['all']
            )

        # ---- Accumulate into Python lists; save once at the end ----
        embeddings_list = []
        labels_list = []
        image_times_list = [] if args.single_frames else None
        img_pth_list = []

        n_frames = None
        n_datasets = sum(len(ld.dataset) for ld in loaders)

        for loader_idx, loader in enumerate(loaders):
            pbar = tqdm.tqdm(total=len(loader))
            for i, batch in enumerate(loader):
                if isinstance(batch, list):
                    im, lbl, img_pth = batch[0], batch[1], batch[2]
                else:
                    im, lbl, img_pth = batch["images"], batch["classes"], batch["image_paths"]

                B = im.shape[0]

                if n_frames is None:
                    n_frames = im.shape[2]  # Get number of frames

                # im: (B, C, T, D, H, W) -> (B, T, C, D, H, W)
                im = im.permute(0, 2, 1, 3, 4, 5).contiguous()

                # ---- Generate spatial (single-frame) embeddings ----
                if args.single_frames:
                    # labels for each frame in the batch
                    frame_labels = np.repeat(
                        lbl.detach().cpu().numpy().reshape(-1).astype(np.int32),
                        n_frames
                    )

                    with torch.no_grad():
                        for t in range(n_frames):
                            img_pth_list.extend(img_pth)
                            # Select the t-th frame: (B, 1, C, D, H, W)
                            frame = im[:, t:t+1, :, :, :, :]

                            # Forward pass -> (B, 2048) after selecting last step
                            features, _ = model(frame.to(device))
                            features = F.normalize(features, dim=-1)   # (B, 2048) or (B, T, 2048)
                            features = features[:, -1, :]              # ensure (B, 2048)
                            feats_np = features.detach().cpu().numpy().astype(np.float32)

                            embeddings_list.extend(feats_np)  # add B rows
                            labels_list.extend(lbl.detach().cpu().numpy().reshape(-1).astype(np.int32))
                            image_times_list.extend([t] * B)

                # ---- Generate spatiotemporal (sequence) embeddings ----
                else:
                    img_pth_list.extend(img_pth)

                    with torch.no_grad():
                        # Forward pass (model expects (B, T, C, D, H, W) as given)
                        features, _ = model(im.to(device)) 
                        features = F.normalize(features, dim=-1)  # (B, 2048) or (B, T, 2048)

                    embeddings_list.extend(features.detach().cpu().numpy().astype(np.float32))
                    labels_list.extend(lbl.detach().cpu().numpy().reshape(-1).astype(np.int32))

                pbar.update(1)
            pbar.close()
        
        # Convert lists to arrays and save once
        embeddings_arr = np.asarray(embeddings_list, dtype=np.float32)
        labels_arr = np.asarray(labels_list, dtype=np.int32)
        np.save(emb_raw_path, embeddings_arr)
        np.save(lbl_path, labels_arr)
        if args.single_frames and image_times_list is not None:
            image_times_arr = np.asarray(image_times_list, dtype=np.int32)
            np.save(img_times, image_times_arr)

        # Saving image paths (text file with one path per line)
        with open(img_pathfile, 'w') as f:
            for pth in img_pth_list:
                f.write(f"{pth}\n")

        np.save(osp.join(embeddings_dir, 'label_names.npy'), np.array(list(drug_labels_dict.keys())))

        # Set up colormaps
        # cell_region_map = osp.join(args.data_path, "cell_to_region_new.csv")
        # create_colormap(img_pathfile, cell_region_map, embedding_dir=embeddings_dir)

    if args.reproject or args.save_embeddings:
        # ---- UMAP: fit on subsample, transform all at once; no memmap, no blocks ----
        feats = np.load(emb_raw_path)  # (N, T, D) fully loaded
        
        feats = feats[:, -1, :] # Get last frame
        # feats = einops.reduce(feats, 'n, t, d -> n d', 'mean') # Mean over time
        # feats = einops.rearrange(feats, 'n t d -> (n t) d') # Treat each frame independently

        # --- Apply TVN prior to PCA/UMAP if requested ---
        if args.use_tvn:
            print("Applying TVN...")
            try:
                labels_all = np.load(lbl_path)  # (N,)
                ctrl_mask = np.array([label_drug_dict.get(int(l)) == 'control' for l in labels_all], dtype=bool)
                if ctrl_mask.sum() > 0:
                    feats = tvn_global(feats, ctrl_mask, eps=1e-6, ledoit_wolf=False).astype(np.float32)
                    print(f"Applied TVN using {ctrl_mask.sum()} control samples (control).")
                else:
                    print("Warning: --use_tvn set but no control samples found in labels; skipping TVN.")
            except Exception as e:
                print(f"Warning: TVN failed with error: {e}. Proceeding without TVN.")

        N_total = feats.shape[0]

        pca_dim = 64
        subsample = min(30000, N_total)
        rng = np.random.default_rng(0)
        if subsample < N_total:
            idx = np.sort(rng.choice(N_total, size=subsample, replace=False))
        else:
            idx = np.arange(N_total)

        if args.use_pca:
            pca = PCA(n_components=pca_dim, svd_solver='randomized', whiten=False)
            pca.fit(feats[idx])
            sample_red = pca.transform(feats[idx])
            feats_red = pca.transform(feats)
        else:
            pca = None
            sample_red = feats[idx]
            feats_red = feats

        reducer = umap.UMAP(
            verbose=True,
            n_components=3,
            n_neighbors=25,
            min_dist=0.01,
            metric='cosine',
            low_memory=True,
        )
        emb3d = reducer.fit_transform(feats_red).astype(np.float32)
        emb_umap_path = osp.join(embeddings_dir, 'embeddings_umap.npy')
        np.save(emb_umap_path, emb3d)
        print(f"UMAP embeddings saved to: {emb_umap_path}")

    if args.visualize:
        if not osp.exists(osp.join(embeddings_dir, "embeddings_umap.npy")):
            print("Embeddings are not saved. Please run the script again without --visualize flag (or with --save_embeddings).")
            exit()

        # image_paths = np.loadtxt(osp.join(embeddings_dir, 'image_paths.csv'), dtype=str).tolist()

        # if args.single_frames:
        #     image_paths = get_single_frame_img_paths(image_paths)

        make_mitospace(embedding_dir=embeddings_dir,
                    pick_labels=labels,
                    color_palette=colors,
                    image_paths=image_paths,
                    single_frames=args.single_frames,
                    save_pcd=args.save_pcd,
                    label_drug_dict=label_drug_dict,
                    datasets=args.datasets
                   )
        exit()
