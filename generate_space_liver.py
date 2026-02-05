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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simclr.models_simple import Lightweight3DResNet # newly trained model (some dict changes)
# from simclr.models_simple_original import Lightweight3DResNet

# from simclr.models import MitoSpace4DConvLSTM
# from simclr.models_simple_attn import Lightweight3DResNet
from utils.tvn import *
import joblib

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('--checkpoint_path', help='Checkpoint path', 
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitospaceResnetBiLSTM_Summer2024.ckpt"
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_decoupled-tmrm_r20251115_epoch=145-step=25988-val_loss=0.00.ckpt",
                    default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_ablated-tmrm_r20251115_epoch=161-step=28836-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_decoupled-tmrm_r20251115_epoch=256-step=41120-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_ablated-tmrm_r20251115_epoch=291-step=46720-val_loss=0.00.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2-161eps-ckpt_kinetics_ablated-tmrm_r20260105.ckpt",
                    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2-161eps_ft-kinetics60_ablated-tmrm_r20260110_epoch=178-step=28819_best.ckpt"
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

        # if with_lines:
        #     # Create lines between consecutive points of the same label
        #     lines = []
        #     line_colors = []
        #     for idx, row in df.iterrows():
        #         next_point = row['next_point']
        #         if next_point != -1:
        #             lines.append([idx, next_point])
        #             line_colors.append(row[f'cmap_{cmap}'])
        #     if lines:
        #         line_set = o3d.geometry.LineSet(
        #             points=pcd.points,
        #             lines=o3d.utility.Vector2iVector(lines)
        #         )
        #         line_set.colors = o3d.utility.Vector3dVector(line_colors)

        vis = o3d.visualization.VisualizerWithEditing()

        vis.create_window()
        vis.add_geometry(pcd)
        # vis.add_geometry(line_set) if with_lines and lines else None

        print(f"Number of Points: {len(pcd.points)}")
        vis.get_render_option().point_size = 5.0

        vis.get_render_option().background_color = np.asarray([0, 0, 0]) # black

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
        time_indices = df['frame_id'].iloc[idxs]
        picked_image_paths = df['path'].iloc[idxs]

        imgs = []
        for idx in idxs:
            if picked_image_paths is not None:
                img_4d = np.load(df['path'].iloc[idx].replace("SSD_processing", "ssd_processing"))  # (t, c, d, h, w)
                imgs.append(img_4d[:, -1, ...].max(axis=1))  # MIP specified frame
                #
                # if decoder is not None:
                #     img_tensor = torch.from_numpy(img_4d).unsqueeze(0).cuda()  # (1, t, c, d, h, w)
                #     with torch.no_grad():
                #         img_tensor = decoder(img_tensor)
                #     img_4d = img_tensor.squeeze(0).cpu().numpy()
                #     img_4d = img_4d.astype(np.float32)
                #
                # if is_4d:
                #     imgs.append(img_4d[:, -1, ...].max(axis=1))  # MIP last frame
                # else:
                #     imgs.append(img_4d[:, df['frame_id'].iloc[idx], ...].max(axis=1)) # MIP specified frame
                #
                # skimage.io.imsave("/home/dhruvagarwal/Desktop/p110.tiff", img_4d[0, 0])

                # add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 0), channel=0, label=label_names[(labels[idx]%27)])
                # add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 256 + 10), channel=1, label=label_names[(labels[idx]%27)])

         # napari_viewer.window.add_plugin_dock_widget(
        #     plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
        # )
        # napari.run()

        print(drug_names)
        print(picked_image_paths)

        colors = np.array(pcd.colors)[idxs]

        # f, axarr = plt.subplots(2, len(imgs), figsize=(20, 20))
        # f.suptitle("MIP", fontsize=50)
        # vals = []
        # for i in range(len(imgs)):
        #     mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
        #     tmrm_idx = 0
        #
        #     axarr[0, i].imshow(imgs[i][tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
        #     axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)
        #
        #     vals.append(np.mean(imgs[i][:, :, 0]))
        #     axarr[0, i].set_xticks([])
        #     # for minor ticks
        #     axarr[0, i].set_yticks([])
        #
        #     # axarr[1, i].set_xticks([])
        #     # for minor ticks
        #     # axarr[1, i].set_yticks([])
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
    save_dir = f"{proj_dir}/runs/"

    reproject = False
    pick_labels = None
    # pick_labels = [0, 36, 37]
    # pick_labels = [0, 5, 10, 21, 24, 36, 37]
    pick_labels = [0, 36, 37, 10, 21, 22]


    embedding_dirs = [
        # 3D embeddings
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm",
        "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_3D-embeddings_Kinetics3D-model",
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm/",

        # 4D embeddings
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-4D-embeddings_2024v2-161eps_ablated-tmrm",
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_4D-embeddings_2024v2-model",
    ]

    # combined_embedding_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260203_liver+kinetics_combined_3D"
    combined_embedding_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/tmp"
    os.makedirs(combined_embedding_dir, exist_ok=True)

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
    colors = get_label_colormap(proj_dir)  # Default to label colormap

    df_data = pd.DataFrame(columns=["label", "label_name", "path", "embedding", "frame_id"])
    for dir in embedding_dirs:
        labels = np.load(osp.join(dir, "labels.npy"))
        paths = np.loadtxt(osp.join(dir, "image_paths.csv"), dtype=str)
        embeddings = np.load(osp.join(dir, "embeddings_raw.npy"))
        label_names = [label_drug_dict[l] for l in labels]
        frame_ids = [list(range(20)) for _ in range(len(labels))]

        # if len(embeddings.shape) == 3:
            # Take last frame embedding only
            # embeddings = embeddings[:, -1, :]

            # Take the mean embedding
            # embeddings = np.mean(embeddings, axis=1)

        df_data = pd.concat([df_data, pd.DataFrame({"label": labels, "label_name": label_names, "path": paths, "embedding": embeddings.tolist(), "frame_id": frame_ids})], ignore_index=True)

    # Explode the embeddings into individual rows
    df_data = df_data.explode('embedding')

    df_data = df_data.reset_index(drop=True)
    color_palette = get_label_colormap(proj_dir)
    df_data['cmap_label'] = df_data['label'].map(color_palette)

    if reproject:

        # reducer_2d = UMAP(
        #     verbose=True,
        #     n_components=2,
        #     n_neighbors=50,
        #     min_dist=0.01,
        #     metric='cosine',
        # )
        #
        # X_2d = reducer_2d.fit_transform(np.array(np.stack(df_data['embedding'].values, axis=0)))
        #
        # # Plot in 2d with points colored by label
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.scatter(X_2d[:, 0], X_2d[:, 1], c=df_data['cmap_label'].values, s=1)
        # # ax.set_title("Combined Embeddings UMAP 2D Projection Colored by Label")
        # plt.show()

        # reducer = UMAP(
        #     verbose=True,
        #     n_components=3,
        #     n_neighbors=25,
        #     min_dist=0.01,
        #     metric='cosine',
        # )

        reducer = UMAP(
            verbose=True,
            n_components=3,
            n_neighbors=50,
            # min_dist=0.01,
            metric='cosine',
        )

        X = np.stack(df_data['embedding'].values, axis=0)
        X_reduced = reducer.fit_transform(X)
        df_data['umap_embeddings'] = list(X_reduced)

        if pick_labels:
            # Filter the data to only include the specified labels
            print(f"Filtering embeddings to only include labels: {pick_labels}")
            df_data = df_data[df_data['label'].isin(pick_labels)].reset_index(drop=True)

        # Save the data to a parquet file for faster loading next time
        df_data.to_parquet(f"{combined_embedding_dir}/embeddings_combined.parquet")
    else:
        df_data = pd.read_parquet(f"{combined_embedding_dir}/embeddings_combined.parquet")

    if pick_labels:
        # Filter the data to only include the specified labels
        print(f"Filtering embeddings to only include labels: {pick_labels}")
        df_data = df_data[df_data['label'].isin(pick_labels)].reset_index(drop=True)

    pick_points(df_data, reducer="umap")