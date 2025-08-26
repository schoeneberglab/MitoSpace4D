"""
Adding custom coloring by time and instance.

To-Do's
- Implement custom coloring logic (add cli flags?)
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

from numpy.lib.format import open_memmap  # ensures .npy headers for memmapped files
from sklearn.decomposition import PCA

from utils.vis import make_mitospace
from data_aug.dataset_utils import get_mitospace_data_loaders
from train_simclr import SimCLRRunner
import torch.nn.functional as F
from utils.utils import normalize, load_config, get_fpaths
from torch.utils.data import DataLoader
from utils.utils import get_drug_labels, increase_contrast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from simclr.models_simple import Lightweight3DResNet
# from simclr.models import MitoSpace4DConvLSTM
# from simclr.models_simple_attn import Lightweight3DResNet

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--checkpoint_path', help='Checkpoint path', default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitoSpace4D_resnetbilstm_encoded_normal_eps287.ckpt")
parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml', type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict', default="/run/user/1002/gvfs/smb-share:server=aquila0.jslab.ucsd.edu,share=ssd_processing/Others/MitoSpace4D/2025_summer")
# parser.add_argument('--embeddings_dir', help='Directory to save/load embeddings', default=None)

parser.add_argument('--visualize_space', default=True, action='store_true', help='Visualize MitoSpace')
parser.add_argument('--save_embeddings', default=False, action='store_true', help='Save embeddings')
parser.add_argument('--single_frames', default=True, action='store_true', help='Embeds all frames individually.')

parser.add_argument('--labels', default=None, type=int, nargs='+', help='Labels to pick. Default is all labels.')
parser.add_argument('--color_by', default='label', help='Color map to use.', choices=['label', 'time', 'region'])

parser.add_argument('--reproject', default=False, action='store_true', help='Reproject embeddings')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dataloaders')
parser.add_argument('--use_pca', default=False, action='store_true', help='Use PCA for dimensionality reduction before UMAP. Default is False.')
parser.add_argument('--densmap', default=False, action='store_true', help='Use densMAP instead of UMAP. Default is False.')

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

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    # save_dir = f"{proj_dir}/runs/"
    save_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/"
    
    embeddings_dir = osp.join(save_dir, 'embeddings_test')
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

    batch_size = args.batch_size
    t_slice = 0
    z_slice = 30

    if args.color_by == 'time':
        colors = get_temporal_colormap(embeddings_dir)
    elif args.color_by == 'region':
        colors = get_region_colormap(embeddings_dir)
    else:
        colors = get_label_colormap(proj_dir)  # Default to label colormap

    emb_raw_path = osp.join(embeddings_dir, 'embeddings_raw.npy')   # (N, 2048) float32
    lbl_path     = osp.join(embeddings_dir, 'labels.npy')           # (N,) int32
    img_path     = osp.join(embeddings_dir, 'images.npy')           # (N, C, H, W) float16 (init later)
    img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')      # (N,) object
    img_times    = osp.join(embeddings_dir, 'image_times.npy')      # (N,) int32

    if args.visualize_space:
        if not osp.exists(osp.join(embeddings_dir, "embeddings_umap.npy")):
            print("Embeddings are not saved. Please run the script again without --visualize_space flag (or with --save_embeddings).")
            exit()

        # image_paths = np.loadtxt(osp.join(embeddings_dir, 'image_paths.csv'), dtype=str).tolist()

        # if args.single_frames:
        #     image_paths = get_single_frame_img_paths(image_paths)

        make_mitospace(embedding_dir=embeddings_dir,
                       pick_labels=labels,
                       color_palette=colors,
                       image_paths=image_paths,
                       single_frames=args.single_frames)
        exit()

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

        # ---- One-pass streaming with .npy headers (open_memmap) ----
        N_total = None
        n_frames = None
        n_datasets = sum(len(ld.dataset) for ld in loaders)

        # emb_raw_path = osp.join(embeddings_dir, 'embeddings_raw.npy')   # (N, 2048) float32
        # lbl_path     = osp.join(embeddings_dir, 'labels.npy')           # (N,) int32
        # img_path     = osp.join(embeddings_dir, 'images.npy')           # (N, C, H, W) float16 (init later)
        # img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')      # (N,) object
        # img_times    = osp.join(embeddings_dir, 'image_times.npy')      # (N,) int32

        embeddings_mm  = None  # created lazily when N_total is known
        labels_mm      = None
        image_times_mm = None

        write_idx = 0
        img_pth_list = []
        for loader_idx, loader in enumerate(loaders):
            pbar = tqdm.tqdm(total=len(loader))
            for i, batch in enumerate(loader):
                if isinstance(batch, list):
                    im, lbl, img_pth = batch[0], batch[1], batch[2]
                else:
                    im, lbl, img_pth = batch["images"], batch["classes"], batch["image_paths"]

                B = im.shape[0]

                if N_total is None:
                    n_frames = im.shape[2]  # Get number of frames
                    N_total = n_datasets * (n_frames if args.single_frames else 1)
                    embeddings_mm  = open_memmap(emb_raw_path, mode='w+', dtype=np.float32, shape=(N_total, 2048))
                    labels_mm      = open_memmap(lbl_path,     mode='w+', dtype=np.int32,   shape=(N_total,))
                    if args.single_frames:
                        image_times_mm = open_memmap(img_times,     mode='w+', dtype=np.int32,   shape=(N_total,))

                # im: (B, C, T, D, H, W) -> (B, T, C, D, H, W)
                im = im.permute(0, 2, 1, 3, 4, 5).contiguous()

                # ---- Generate spatial (single-frame) embeddings ----
                if args.single_frames:
                    end_idx = write_idx + (n_frames * B)

                    # labels for each frame in the batch
                    frame_labels = np.repeat(
                        lbl.detach().cpu().numpy().reshape(-1).astype(np.int32),
                        n_frames
                    )
                    labels_mm[write_idx:end_idx] = frame_labels

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

                            # Compute slice for this frame across the batch
                            start = write_idx + t * B
                            stop  = start + B

                            # Write embeddings and times; avoids deprecated array->scalar coercion
                            embeddings_mm[start:stop]  = feats_np            # (B, 2048)
                            image_times_mm[start:stop] = t                   # broadcast scalar t to slice

                    # advance the write pointer for the next dataloader batch
                    write_idx = end_idx

                # ---- Generate spatiotemporal (sequence) embeddings ----
                else:
                    img_pth_list.extend(img_pth)
                    end_idx = write_idx + B

                    # Labels
                    labels_mm[write_idx:end_idx] = lbl.detach().cpu().numpy().reshape(-1).astype(np.int32)

                    with torch.no_grad():
                        # Forward pass (model expects (B, T, C, D, H, W) as given)
                        features, _ = model(im.to(device))
                        features = F.normalize(features, dim=-1)  # (B, 2048) or (B, T, 2048)
                        features = features[:, -1, :]             # (B, 2048)

                    # Embeddings
                    embeddings_mm[write_idx:end_idx] = features.detach().cpu().numpy().astype(np.float32)

                    write_idx = end_idx

                pbar.update(1)
            pbar.close()
        
        # Write image paths to the .csv
        with open(img_pathfile, 'w') as f:
            for pth in img_pth_list:
                f.write(f"{pth}\n")

        # Flush to disk
        embeddings_mm.flush()
        labels_mm.flush()
        if image_times_mm is not None:
            image_times_mm.flush()

        # Save the label names
        np.save(osp.join(embeddings_dir, 'label_names.npy'), np.array(list(drug_labels_dict.keys())))

    # if args.reproject or args.save_embeddings:
    #     # ---- UMAP: fit on subsample, transform in chunks; read with np.load ----
    #     feats_mm = np.load(emb_raw_path, mmap_mode='r')  # (N_total, 2048) backed by .npy file
    #     N_total = feats_mm.shape[0]

    #     pca_dim = 64
    #     subsample = min(30000, feats_mm.shape[0])
    #     rng = np.random.default_rng(0)
    #     if subsample < feats_mm.shape[0]:
    #         idx = np.sort(rng.choice(feats_mm.shape[0], size=subsample, replace=False))
    #     else:
    #         idx = np.arange(feats_mm.shape[0])

    #     if args.use_pca:
    #         pca = PCA(n_components=pca_dim, svd_solver='randomized', whiten=False)
    #         pca.fit(feats_mm[idx])
    #         sample_red = pca.transform(feats_mm[idx])
    #     else:
    #         pca = None
    #         sample_red = feats_mm[idx]

    #     reducer = umap.UMAP(
    #         verbose=True,
    #         n_components=3,
    #         n_neighbors=25,
    #         min_dist=0.001,
    #         metric='cosine',
    #         low_memory=True,
    #     )
    #     reducer.fit(sample_red)

    #     emb_umap_path = osp.join(embeddings_dir, 'embeddings_umap.npy')
    #     emb3d_mm = open_memmap(emb_umap_path, dtype=np.float32, mode='w+', shape=(N_total, 3))
        
    #     bs = 8192
    #     for start in tqdm.tqdm(range(0, N_total, bs), desc="UMAP transform"):
    #         stop = min(start + bs, N_total)
    #         block = feats_mm[start:stop]
    #         if args.use_pca:
    #             block = pca.transform(block)
    #         emb3d_mm[start:stop] = reducer.transform(block).astype(np.float32)
    #     emb3d_mm.flush()

    # expects: args.densmap (bool), args.use_pca (bool), args.reproject / args.save_embeddings

    if args.reproject or args.save_embeddings:
        # ---- UMAP: fit on subsample, transform all at once; no memmap, no blocks ----
        feats = np.load(emb_raw_path)  # (N_total, D) fully loaded
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
            feats_red = pca.transform(feats)   # all-at-once
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
            densmap=args.densmap,
            # dens_lambda=0.1,
        )
        # reducer.fit(sample_red)
        # emb3d = reducer.transform(feats_red).astype(np.float32)  # all-at-once
        emb3d = reducer.fit_transform(feats_red).astype(np.float32)  # all-at-once

        emb_umap_path = osp.join(embeddings_dir, 'embeddings_umap.npy')
        np.save(emb_umap_path, emb3d)


        # ---- Align image_paths to embedding count to avoid mask indexing mismatch in make_mitospace ----
        # If you want to visualize with per-sample image paths, load the saved list here.
        # Otherwise, pass None and make_mitospace can render without thumbnails.
        try:
            image_paths = np.loadtxt(img_pathfile, dtype=str).tolist()
        except Exception:
            image_paths = None

        if image_paths is not None:
            if len(image_paths) >= N_total:
                image_paths_aligned = image_paths[:N_total]
            else:
                pad_n = N_total - len(image_paths)
                image_paths_aligned = image_paths + ([image_paths[-1]] * pad_n if len(image_paths) > 0 else [])
        else:
            image_paths_aligned = None

        # Final visualization
        make_mitospace(
            embedding_dir=embeddings_dir,
            pick_labels=labels,
            color_palette=colors,
            image_paths=image_paths_aligned,
            single_frames=args.single_frames
        )
