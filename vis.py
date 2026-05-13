import argparse
import os
import os.path as osp
import random
import time

import einops
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import napari
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import yaml
from sklearn.decomposition import PCA
from umap import UMAP

from data.dataset_utils import get_mitospace_data_loaders
from utils.utils import load_config


# from cuml.manifold import UMAP
# from cuml.metrics import trustworthiness


# from simclr.models_simple_original import Lightweight3DResNet


np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description="PyTorch SimCLR")

parser.add_argument(
    "--checkpoint_path",
    help="Checkpoint path",
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitospaceResnetBiLSTM_Summer2024.ckpt"
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_decoupled-tmrm_r20251115_epoch=145-step=25988-val_loss=0.00.ckpt",
    default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_ablated-tmrm_r20251115_epoch=161-step=28836-val_loss=0.00.ckpt",
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_decoupled-tmrm_r20251115_epoch=256-step=41120-val_loss=0.00.ckpt",
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_ablated-tmrm_r20251115_epoch=291-step=46720-val_loss=0.00.ckpt",
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2-161eps-ckpt_kinetics_ablated-tmrm_r20260105.ckpt",
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2-161eps_ft-kinetics60_ablated-tmrm_r20260110_epoch=178-step=28819_best.ckpt"
)

parser.add_argument(
    "--config", default="./simclr/config.yaml", type=str, help="Config path."
)
parser.add_argument(
    "--data_path",
    help="Data to predict",
    # default="/home/earkfeld/Projects/MitoSpace4D/data/2024v2_encoded_data",
    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_summer_new/", # 2024v2 Dataset
    # default="/mnt/DATA_02/2024_data_encoded", # 2024v2 Dataset Encoded
    default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/2025_kinetics_data/processed_data",
    # Kinetics Dataset
    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_data_encoded/", # Kinetics Dataset Encoded
    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data"
    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_3b"
    # default="/run/user/1002/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/leukemia_drug_resistance_data")
    #     default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_4",
    # default="/home/earkfeld/Projects/MitoSpace4D/data/2025_kinetics_encoded_data"
)

parser.add_argument(
    "--decoder_ckpt",
    help="Path to decoder checkpoint",
    # default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
    default=None,
)

parser.add_argument(
    "--embeddings_dir", help="Directory to save/load embeddings", default=None
)

parser.add_argument(
    "--visualize",
    default=True,
    action="store_true",
    help="Visualize UMAP'd MitoSpace embeddings",
)
parser.add_argument(
    "--save_embeddings", default=False, action="store_true", help="Save embeddings"
)
parser.add_argument(
    "--reproject", default=False, action="store_true", help="Reproject embeddings"
)
parser.add_argument("--save_pcd", default=None, help="Path to save the point cloud")

parser.add_argument(
    "--single_frames",
    default=False,
    action="store_true",
    help="Generates frame embeddings independently.",
)
parser.add_argument(
    "--labels",
    default=None,
    type=int,
    nargs="+",
    help="Labels to pick. Default is all labels.",
)
parser.add_argument(
    "--datasets",
    default=None,
    type=str,
    nargs="+",
    help="Datasets to use. Default is all datasets.",
)
parser.add_argument("--cmap", default="label", help="Color map to use.")
parser.add_argument(
    "--to_load",
    default="all",
    type=str,
    choices=["all", "train", "val"],
    help='Which splits to load. Default is "all".',
)

parser.add_argument(
    "--load_transform",
    default=False,
    action="store_true",
    help="Load UMAP transform from disk instead of fitting new one.",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for dataloaders"
)
parser.add_argument(
    "--use_pca",
    default=False,
    action="store_true",
    help="Use PCA for dimensionality reduction before UMAP. Default is False.",
)
parser.add_argument(
    "--densmap",
    default=False,
    action="store_true",
    help="Use densMAP instead of UMAP. Default is False.",
)
parser.add_argument(
    "--tvn",
    default=False,
    action="store_true",
    help="Apply Typical Variation Normalization (TVN) using controls before UMAP.",
)  # <-- added flag
parser.add_argument(
    "--control_label",
    default="control",
    type=str,
    help='Label name for the control samples for performing typical variation normalization (TVN). Default is "control".',
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.multiprocessing.set_sharing_strategy("file_system")


def get_label_colormap(proj_dir):
    colors = {}
    with open(f"{proj_dir}/extraction_utils/colors.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:
                date, label, index, r, g, b = parts
                if float(r) >= 1 or float(g) >= 1 or float(b) >= 1:
                    colors[int(index)] = [
                        float(r) / 255,
                        float(g) / 255,
                        float(b) / 255,
                    ]
                else:
                    colors[int(index)] = [float(r), float(g), float(b)]
            else:
                print("Invalid line format:", line)
    return colors


def pick_points(df, reducer="umap", cmap="label", decoder=None, is_4d=False):
    def _pick_points():
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")

        if cmap == "label":
            legend_colors = []
            unique_label_vals = np.unique(df["labels"])
            for i, l in enumerate(unique_label_vals):
                label_name = df["label_names"][df["labels"] == l].iloc[0]
                color = df[f"cmap_{cmap}"][df["labels"] == l].iloc[0]
                legend_colors.append((label_name, l, color))

            # Setup legend for pcd colors
            fig, ax = plt.subplots()

            for legend_entry in legend_colors:
                ax.scatter(
                    [],
                    [],
                    color=legend_entry[2],
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

            pcd_colors = df["cmap_label"].values
        elif cmap == "moa":
            legend_colors = []
            unique_label_vals = np.unique(df["labels_moa"])
            for i, l in enumerate(unique_label_vals):
                label_name = df["labels_moa"][df["labels_moa"] == l].iloc[0]
                color = df[f"cmap_moa"][df["labels_moa"] == l].iloc[0]
                legend_colors.append((label_name, l, color))

            # Setup legend for pcd colors
            fig, ax = plt.subplots()

            for legend_entry in legend_colors:
                ax.scatter(
                    [],
                    [],
                    color=legend_entry[2],
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

            pcd_colors = df["cmap_moa"].values

        elif cmap == "random":
            # create a discrete colormap and create legend entries
            random_values = np.random.uniform(low=0.0, high=1.0, size=len(df))
            pcd_colors = plt.get_cmap("plasma")(random_values)[
                :, :3
            ]  # get RGB values only, 0-1 range

        else:
            # Create a colormap from the values in the specified column
            col_values = df[cmap].values

            # check if the column values are categorical or continuous
            if col_values.dtype == object:
                # keep the last value in each array in col_values
                col_values = np.array([val[-1] for val in col_values])
            if len(np.unique(col_values)) < 1000 and (
                isinstance(col_values[0], str) or isinstance(col_values[0], int)
            ):  # arbitrary threshold for categorical vs continuous
                # create a discrete colormap and create legend entries
                unique_vals = np.unique(col_values)
                legend_colors = []
                for i, val in enumerate(unique_vals):
                    color = plt.get_cmap("tab10")(i % 10)  # cycle through tab10 colors
                    legend_colors.append((val, color))

                # Setup legend for pcd colors
                fig, ax = plt.subplots()
                for legend_entry in legend_colors:
                    ax.scatter(
                        [],
                        [],
                        color=legend_entry[1],
                        label=f"{legend_entry[0]}",
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
            else:
                # If continuous, create a continuous colormap and add a colorbar with the column name as label
                # plt.figure(figsize=(6, 1))
                # norm = plt.Normalize(vmin=col_values.min(), vmax=col_values.max())
                # sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                # sm.set_array([])
                # cbar = plt.colorbar(sm, orientation='horizontal')
                # cbar.set_label(cmap, fontsize=12)
                # plt.show(block=False)
                # plt.pause(0.5)
                # pcd_colors = plt.get_cmap('viridis')(norm(col_values))[:, :3]  # get RGB values only, 0-1 range

                # Log scale the values
                # col_values = np.log10(col_values + np.full_like(col_values, 1e-9))  # add small value to avoid log(0)
                # pcd_colors = plt.get_cmap('viridis')(col_values)[:, :3]
                # normalize the values to 0-1 range after log scaling

                if cmap == "tmrm_intensities":
                    # Clip b/c tails
                    vmax_val = np.percentile(col_values, 99)
                    vmin_val = np.percentile(col_values, 1)

                    # Print the min max vals
                    print(f"vmin: {vmin_val}, vmax: {vmax_val}")
                    col_values = np.clip(col_values, vmin_val, vmax_val)

                if cmap == "morph_intensities":
                    # Clip b/c tails
                    vmax_val = np.percentile(col_values, 99)
                    vmin_val = np.percentile(col_values, 1)

                    # Print the min max vals
                    print(f"vmin: {vmin_val}, vmax: {vmax_val}")
                    col_values = np.clip(col_values, vmin_val, vmax_val)

                if (
                    cmap == "fragment_diffusivity_mean"
                    or cmap == "segment_diffusivity_mean"
                ):
                    vmax_val = np.percentile(col_values, 99)
                    vmin_val = np.percentile(col_values, 1)
                    col_values = np.clip(col_values, vmin_val, vmax_val)
                    # col_values = np.log2(col_values + np.full_like(col_values, 1e-9))
                    # col_values = np.log10(col_values + np.full_like(col_values, 1e-9))

                if cmap == "segment_length_mean":
                    vmax_val = np.percentile(col_values, 99)
                    vmin_val = np.percentile(col_values, 1)
                    col_values = np.clip(col_values, vmin_val, vmax_val)

                col_values = (col_values - col_values.min()) / (
                    col_values.max() - col_values.min() + 1.0e-9
                )

                # col_values = np.log10(col_values)
                pcd_colors = plt.get_cmap("plasma")(col_values)[:, :3]

        # Set up the point cloud
        pcd = o3d.geometry.PointCloud()

        emb_col = f"embeddings_{reducer}"
        if emb_col not in df.columns:
            raise ValueError(
                f"Embedding column '{emb_col}' not found in dataframe. Available columns: {df.columns}"
            )

        pcd.points = o3d.utility.Vector3dVector(np.stack(df[emb_col].values, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)

        print(f"Number of Points: {len(pcd.points)}")
        vis.get_render_option().point_size = 5.0

        vis.run()

        idxs = vis.get_picked_points()

        if len(idxs) == 0:
            print("No points picked, closing the visualizer.")
            vis.destroy_window()
            plt.close("all")
            exit(0)

        print(idxs)
        drug_names = df["label_names"].iloc[idxs]
        picked_image_paths = df["image_paths"].iloc[idxs]
        print(drug_names)
        print(picked_image_paths)

        for idx in idxs:
            print(idx)
            print(df["labels"][idx])
            print(label_drug_dict[df["labels"][idx]])
            print(df["image_paths"][idx])
            print("")

        picked_points = vis.get_picked_points()
        vis.destroy_window()
        plt.close("all")

        viewer = napari.Viewer(ndisplay=3)
        added_any = False
        for i, idx in enumerate(idxs):
            path = df["image_paths"].iloc[idx]
            try:
                img = np.load(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

            drug = label_drug_dict[df["labels"].iloc[idx]]
            name = f"{i}: {drug} — {osp.basename(path)}"
            print(f"Picked {i}: shape={img.shape}, dtype={img.dtype}, path={path}")

            lo = float(img.min())
            hi = float(np.percentile(img, 99.9))
            if hi <= lo:
                hi = float(img.max())
            if hi <= lo:
                hi = lo + 1

            if img.ndim == 4:
                # (T, Z, Y, X) — 4D movie
                offset = i * (img.shape[-1] + 20)
                viewer.add_image(
                    img,
                    name=name,
                    colormap="green",
                    contrast_limits=(lo, hi),
                    rendering="mip",
                    translate=(0, 0, 0, offset),
                )
                added_any = True
            elif img.ndim == 3 and img.shape[0] == 2:
                # (C, Y, X) — legacy 2-channel 2D fallback; keep morph channel
                img2d = img[1]
                offset = i * (img2d.shape[-1] + 20)
                viewer.add_image(
                    img2d,
                    name=name,
                    colormap="green",
                    contrast_limits=(lo, hi),
                    rendering="mip",
                    translate=(0, offset),
                )
                added_any = True
            else:
                print(f"Unsupported image shape {img.shape} at {path}; skipping.")

        if added_any:
            viewer.reset_view()
            napari.run()
        else:
            print("No layers added to napari viewer; skipping display.")
            viewer.close()

        return picked_points

    while True:
        _pick_points()

def create_datafile(embeddings_dir, outfile="embeddings+metadata_vis.parquet"):
    intensities_infile = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/2024v3_channel_intensities.parquet"
    embeddings = np.load(osp.join(embeddings_dir, "embeddings.npy"))
    labels = np.load(osp.join(embeddings_dir, "labels.npy"))
    label_names = np.load(osp.join(embeddings_dir, "label_names.npy"))
    image_paths = np.loadtxt(
        osp.join(embeddings_dir, "image_paths.csv"), dtype=str
    ).tolist()

    # Get last frame embeddings only
    embeddings = embeddings[:, -1, :]

    df = pd.read_parquet(intensities_infile)
    df = df.rename(columns={"morph_path": "image_paths"})

    df_embeddings = pd.DataFrame(
        {
            "image_paths": image_paths,
            "labels": labels,
            "embeddings": embeddings.tolist(),
            "label_names": [label_names[lbl] for lbl in labels],
        }
    )

    df = df.merge(df_embeddings, on="image_paths", how="inner")
    df.to_parquet(osp.join(embeddings_dir, outfile), index=False)
    print("saved to", outfile)
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Update the cfg with the args (
    cfg.update(vars(args))

    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    save_dir = f"{proj_dir}/runs/"

    pick_labels = None
    args.load_reducer = False

    embeddings_root = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/"

    # embeddings_dir = 'ms2d_2024v3'
    # embeddings_dir = "ms3d_2024v3_225eps"
    embeddings_dir = "ms4d_2024v3_252eps"
    # embeddings_dir = "ms4d_2024v3_supcon_190eps"
    # embeddings_dir = "ms4d_2024v3_zero-shot_241eps"
    # embeddings_dir = 'ms4d_2024v3_resnet_252eps'
    # embeddings_dir = "ms4d_2024v3_tscrambled_284eps"
    # embeddings_dir = "ms4d_2024v3_252eps_tscrambled"
    # embeddings_dir = "ms4d_2024v3_supcon_210eps"

    # embeddings_dir = "ms4d_reproducibility_252eps"
    # embeddings_dir = "ms4d_2024v3_random_init"

    embeddings_dir = osp.join(embeddings_root, embeddings_dir)

    datafile = "embeddings+metadata_vis_joined.parquet"
    # datafile = "embeddings+metadata_vis.parquet"
    filter_infile = osp.join(embeddings_root, "2024v3_exclude_paths.parquet")

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(osp.join(proj_dir, "extraction_utils/drugs_to_labels.txt"), "r") as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    labels = (
        [args.labels] if args.labels else [list(drug_labels_dict.values())]
    )  # Default to all conditions in the drug label dict

    batch_size = args.batch_size
    colors = get_label_colormap(proj_dir)  # Default to label colormap

    if not osp.exists(osp.join(embeddings_dir, datafile)):

        df = create_datafile(embeddings_dir)

        print(
            "Visualization parquet file not found, creating from full parquet file..."
        )

        df = pd.read_parquet(osp.join(embeddings_dir, "embeddings+metadata.parquet"))

        embeddings = np.stack(df["embeddings"].values)
        if embeddings.ndim == 3:
            embeddings = embeddings[:, -1, :]
        elif embeddings.ndim == 2 and isinstance(embeddings[0, -1], (list, np.ndarray)):
            embeddings = np.stack([emb[-1] for emb in embeddings])

        # convert to list for easier handling in the dataframe
        df["embeddings"] = list(embeddings)

        # save to embeddings+metadata_vis.parquet
        # df.to_parquet(osp.join(embeddings_dir, datafile))
        print("Saved visualization parquet file.")

    else:
        print(f"Loading data from {datafile}...")
        df = pd.read_parquet(osp.join(embeddings_dir, datafile))

    df = df.reset_index(drop=True)

    df["cmap_label"] = df["labels"].map(get_label_colormap(proj_dir))

    df_filter = pd.read_parquet(filter_infile)
    n_init = len(df)
    df_data = df[~df["image_paths"].isin(df_filter["image_paths"])].reset_index(
        drop=True
    )
    print(f"Filtered out {n_init - len(df_data)} samples based on the filter file.")

    if args.reproject:
        if args.load_reducer:
            print("Loading UMAP reducer from disk...")
            reducer = joblib.load(osp.join(embeddings_dir, "umap_reducer.pkl"))
        else:
            reducer = UMAP(
                verbose=True,
                n_components=3,
                n_neighbors=25,
                min_dist=0.01,
                metric="cosine",
            )

        embeddings = np.stack(df_data["embeddings"].values)

        print("Original embedding shape:", embeddings.shape)

        if embeddings.ndim == 3:
            embeddings = embeddings[:, -1, :]
        elif embeddings.ndim == 2 and isinstance(embeddings[0, -1], (list, np.ndarray)):
            embeddings = np.stack([emb[-1] for emb in embeddings])

        reducer.random_state = 42

        X_reduced = reducer.fit_transform(embeddings)
        df_data["embeddings_umap"] = list(X_reduced)

        # Save the data with UMAP embeddings to the parquet file for faster loading next time
        df_data.to_parquet(osp.join(embeddings_dir, datafile))

    if args.labels:
        # Filter the data to only include the specified labels
        print(f"Filtering embeddings to only include labels: {args.labels}")
        df_data = df_data[df_data["labels"].isin(args.labels)].reset_index(drop=True)

    df_data = df_data.dropna(subset=args.cmap)

    pick_points(df_data, reducer="umap", cmap=args.cmap)
