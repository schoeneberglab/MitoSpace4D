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
import tifffile
from math import ceil
import open3d as o3d

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
import matplotlib.pyplot as plt

from simclr.models_simple import Lightweight3DResNet  # newly trained model (some dict changes)
# from simclr.models_simple_3d import Lightweight3DResNet # 3D resnet model
# from simclr.models_simple_original import Lightweight3DResNet

from utils.tvn import *
import joblib

import os.path as osp
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('--checkpoint_path', help='Checkpoint path',
                    default="/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/checkpoints/ms4d_2024v3_epoch=300-step=32099.ckpt"
                    )

parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml', type=str,
                    help='Config path.')

parser.add_argument('--embeddings_dir', help='Directory to save/load embeddings', default=None)

parser.add_argument('--visualize', default=False, action='store_true', help="Visualize UMAP'd MitoSpace embeddings")
parser.add_argument('--save_embeddings', default=True, action='store_true', help='Save embeddings')
parser.add_argument('--save_pcd', default=None, help='Path to save the point cloud')

parser.add_argument('--single_frames', default=False, action='store_true',
                    help='Generates frame embeddings independently.')
parser.add_argument('--labels', default=None, type=int, nargs='+', help='Labels to pick. Default is all labels.')
parser.add_argument('--datasets', default=None, type=str, nargs='+', help='Datasets to use. Default is all datasets.')
parser.add_argument('--cmap', default='label', help='Color map to use.',
                    choices=['label', 'temporal', 'region', 'dataset', 'tmrm'])
parser.add_argument('--to_load', default='all', type=str, choices=["all", "train", "val"],
                    help='Which splits to load. Default is "all".')

# parser.add_argument('--channels', default=[0, 1], type=int, nargs='+', help='Channels to use. Default is [0, 1].')

parser.add_argument('--reproject', default=False, action='store_true', help='Reproject embeddings')
parser.add_argument('--load_transform', default=False, action='store_true',
                    help='Load UMAP transform from disk instead of fitting new one.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dataloaders')
parser.add_argument('--use_pca', default=False, action='store_true',
                    help='Use PCA for dimensionality reduction before UMAP. Default is False.')
parser.add_argument('--densmap', default=False, action='store_true',
                    help='Use densMAP instead of UMAP. Default is False.')
parser.add_argument('--tvn', default=False, action='store_true',
                    help='Apply Typical Variation Normalization (TVN) using controls before UMAP.')  # <-- added flag
parser.add_argument('--control_label', default='control', type=str,
                    help='Label name for the control samples for performing typical variation normalization (TVN). Default is "control".')

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

def normalize(x):
    """Normalize array or tensor to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

# def get_sample(frame_paths):
#     frames = [tifffile.imread(f) for f in frame_paths]
#     frames = np.stack(frames, axis=0).astype(np.float32)
#     return normalize(frames)

def get_sample(frame_paths):
    # frames = [tifffile.imread(f) for f in frame_paths]
    # frames = np.stack(frames, axis=0).astype(np.float32)
    frames = []
    for path in frame_paths:
        frame = tifffile.imread(path)
        # Z'ing the image
        z_stride = ceil(frame.shape[0] / 60)
        # select every z_stride-th slice
        frame = frame[::z_stride]
        # expand it by adding zeros to 60 slices
        frame = np.pad(frame, ((0, 60 - frame.shape[0]), (0, 0), (0, 0)), mode='constant')

        # X and Ying the image
        h, w = frame.shape[1], frame.shape[2]

        if h < 256:
            frame = np.pad(frame, ((0, 0), (0, 256 - h), (0, 0)), mode='constant')
        elif h > 256:
            # select the middle 256 slices
            frame = frame[:, h // 2 - 128: h // 2 + 128, :]
        if w < 256:
            frame = np.pad(frame, ((0, 0), (0, 0), (0, 256 - w)), mode='constant')
        elif w > 256:
            # select the middle 256 slices
            frame = frame[:, :, w // 2 - 128: w // 2 + 128]

        frame = np.expand_dims(normalize(frame), axis=[0, 1, 2]) # B=1, T=1, C=1, Z, Y, X
        frames.append(frame.astype(np.float32))
    frames = np.stack(frames, axis=1)  # (B=1, T, C=1, Z, Y, X)
    return normalize(frames)

def pick_points(df, reducer="phate", cmap="label", decoder=None, is_4d=False):

    def _pick_points():
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")

        if cmap == "label":
            legend_colors = []
            unique_label_vals = np.unique(df['label'])
            for i, l in enumerate(unique_label_vals):
                label_name = df['label_name'][df['label'] == l].iloc[0]
                color = df[f'cmap_{cmap}'][df['label'] == l].iloc[0]
                legend_colors.append((label_name, l, color))

            # Setup legend for pcd colors
            fig, ax = plt.subplots()

            for legend_entry in legend_colors:
                ax.scatter(
                    [],
                    [],
                    c=legend_entry[2],
                    label=f"{legend_entry[0]}: {legend_entry[1]}",
                    s=100,
                )
            ax.axis("off")
            legend = ax.legend(
                loc="center",
                frameon=False,
                fontsize=12,
                markerscale=1.5,
            )

            # resize figure to legend size
            fig.canvas.draw()  # needed to compute legend size
            bbox = legend.get_window_extent()
            width, height = bbox.width / fig.dpi, bbox.height / fig.dpi
            fig.set_size_inches(width, height)
            legend.set_bbox_to_anchor((0.5, 0.5), transform=fig.transFigure)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)

        # Set up the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack(df[f'{reducer}_embeddings'].values, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(df[f'cmap_{cmap}'].values)

        vis = o3d.visualization.VisualizerWithEditing()

        vis.create_window()
        vis.add_geometry(pcd)

        print(f"Number of Points: {len(pcd.points)}")
        vis.get_render_option().point_size = 5.0
        # vis.get_render_option().background_color = np.asarray([0, 0, 0]) # black

        vis.run()  # user picks points
        idxs = vis.get_picked_points()

        if len(idxs) == 0:
            print("No points picked, closing the visualizer.")
            vis.destroy_window()
            exit(0)

        # TODO: Fix napari visualization routines
        # napari_viewer = napari.Viewer()
        print(idxs)
        drug_names = df['label_name'].iloc[idxs]
        picked_image_paths = df['path'].iloc[idxs]

        imgs = []
        for idx in idxs:
            if picked_image_paths is not None:
                img_4d = np.load(df['path'].iloc[idx].replace("SSD_processing", "ssd_processing"))  # (t, c, d, h, w)
                imgs.append(img_4d[:, -1, ...].max(axis=1))  # MIP specified frame

         # napari_viewer.window.add_plugin_dock_widget(
        #     plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
        # )
        # napari.run()

        print(drug_names)
        print(picked_image_paths)

        colors = np.array(pcd.colors)[idxs]

        f, axarr = plt.subplots(1, len(imgs), figsize=(20, 20))
        f.suptitle("MIP", fontsize=50)
        vals = []
        for i in range(len(imgs)):

            axarr[i].imshow(imgs[i][0], vmin=0., vmax=1., cmap=plt.cm.viridis)
            # axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)

            vals.append(np.mean(imgs[i][:, :, 0]))
            axarr[i].set_xticks([])
            # for minor ticks
            axarr[i].set_yticks([])

            # axarr[1, i].set_xticks([])
            # for minor ticks
            # axarr[1, i].set_yticks([])

        print(vals)
        plt.xticks([]), plt.yticks([])
        for idx in idxs:
            print(idx)
            print(df['label'][idx])
            print(label_drug_dict[df['label'][idx]])
            print(df['path'][idx])
            print(df['frame_id'][idx])
            print("")

        patches = []
        for i, idx in enumerate(idxs):
            l = label_drug_dict[df['label'][idx]]
            ds_id = picked_image_paths[idx].split("/")[-2]
            sample_id = picked_image_paths[idx].split("/")[-1].split(".npy")[0]
            sample_caption = f"{ds_id}/{sample_id}"
            patches.append(mpatches.Patch(color=colors[i], label="{l} ({sc})".format(l=l, sc=sample_caption)))

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        print("")

        return vis.get_picked_points()

    while True:
        _pick_points()

if __name__ == '__main__':
    # TODO: Set up to read mostly from config, optionally overwrite w/ args, add specific visualization entries, and copy updated to the embedding dir
    # Goal: be able to just provide embeddings dir and use previous settings stored there in a copy of the config file
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Update the cfg with the args (
    cfg.update(vars(args))

    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    # save_dir = f"{proj_dir}/runs/"
    save_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data"

    index_file = "organoid_data.parquet"

    df = pd.read_parquet(osp.join(index_file))

    embeddings_dir = osp.join(save_dir, 'ms4d_organoid_embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    # save merged config and args to embeddings dir
    with open(osp.join(embeddings_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    with open(osp.join(embeddings_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    checkpoint_path = args.checkpoint_path
    print(f"Ckpt Path: {args.checkpoint_path}")
    print(f"Data Path: {args.data_path}")
    image_paths = None  # define upfront; populated only in visualize pass or left None

    visualize_embeddings = "embeddings"

    df["embeddings"] = pd.Series(dtype=object)  # placeholder column for embeddings
    df["resnet_embeddings"] = pd.Series(dtype=object)  # placeholder column for resnet embeddings

    model = Lightweight3DResNet(embedding_size=2048,
                                cfg=cfg,
                                apply_aug=False,
                                decoder_checkpoint_path=None,
                                )

    model = SimCLRRunner.load_from_checkpoint(checkpoint_path, model=model, cfg=cfg, strict=False).model
    model.eval().to(device)

    for param in model.parameters():
        param.requires_grad = False

    if args.save_embeddings:
        pbar = tqdm.tqdm(total=len(df))
        for i, row in df.iterrows():
            frame = get_sample(row['frame_paths'])
            frame = np.expand_dims(frame, axis=[0,2])  # (B, T, C=1, Z, Y, X)

            with torch.no_grad():
                # model expects (B, T, C, D, H, W)
                features, resnet_features, _ = model(torch.from_numpy(frame).to(device), get_resnet_feats=True)
                features = F.normalize(features, dim=-1)  # (B, 2048) or (B, T, 2048)
                resnet_features = F.normalize(resnet_features, dim=-1)  # (B, T, 512)

            df.at[i, 'embeddings'] = features.cpu().numpy()
            df.at[i, 'resnet_embeddings'] = resnet_features.cpu().numpy()

            pbar.update(1)
        pbar.close()

        # Save the dataframe with embeddings to a parquet file
        df.to_parquet(osp.join(embeddings_dir, 'organoid_embeddings.parquet'))

    if args.reproject or args.save_embeddings:
        df_embeddings = pd.read_parquet(osp.join(embeddings_dir, 'organoid_embeddings.parquet'))
        if "embeddings_umap" in df_embeddings.columns:
            print(f"Clearing existing UMAP embeddings...")
        df_embeddings["embeddings_umap"] = pd.Series(dtype=object)  # placeholder column for UMAP embeddings

        feats = np.stack(df_embeddings['embeddings'].values)  # (N, T, D)
        print(f"Features shape: {feats.shape}, dtype: {feats.dtype}")

        feats = feats[:, -1, :]  # take the last time point only

        reducer = UMAP(
            verbose=True,
            n_components=3,
            n_neighbors=25,
            min_dist=0.01,
            metric='cosine',
        )

        emb3d = reducer.fit_transform(feats).astype(np.float32)
        df_embeddings['embeddings_umap'] = pd.Series(data=emb3d, index=df.index)
        pd.to_parquet(osp.join(embeddings_dir, 'organoid_embeddings.parquet'), df_embeddings)

    if args.visualize:
        df_embeddings = pd.read_parquet(osp.join(embeddings_dir, 'organoid_embeddings.parquet'))
        pick_points(df_embeddings, reducer="umap")
