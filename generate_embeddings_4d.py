"""
Created 202620117 by Eric Arkfeld

No Model eval; visualization only using phate for down projection

set up for visualizing 3D embeddings across all frames.
consolidated visualization routines from vis.py
"""

import torch
import torch.nn.functional as F

import random
import open3d as o3d

from cuml.manifold import umap
# from cuml.metrics import trustworthiness

import os.path as osp

# from utils.colormaps import create_colormap # TODO: set up color map generation
from utils.utils import load_config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.feature_trajectory import generate_phate_trajectories

from train_simclr import SimCLRRunner
from simclr.models_simple_exp import Lightweight3DResNet
from utils.tvn import *

from umap import UMAP
import phate

from tqdm import tqdm
import einops

np.random.seed(0)
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.multiprocessing.set_sharing_strategy('file_system')

def pick_points(df, reducer="phate", cmap=None):

    def _pick_points():
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")

        # Set up the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack(df[f'{reducer}_embeddings'].values, axis=0))

        if cmap is not None:
            print(f"Using {cmap} colormap")
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
        # drug_names = df['label_name'].iloc[idxs]
        picked_image_paths = df['path'].iloc[idxs]

        imgs = []
        for idx in idxs:
            if picked_image_paths is not None:
                # img_4d = np.load(df['img_path'].iloc[idx])  # (t, c, d, h, w)
                img = np.load(df['path'].iloc[idx])
                # Maximum intensity projection over over first axis
                imgs.append(img.max(axis=0))

        # napari_viewer.window.add_plugin_dock_widget(
        #     plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
        # )
        # napari.run()

        # print(drug_names)
        print(picked_image_paths)

        # colors = np.array(pcd.colors)[idxs]

        # f, axarr = plt.subplots(1, len(imgs), figsize=(20, 20))
        # f.suptitle("MIP", fontsize=50)
        # vals = []
        # for i in range(len(imgs)):
        #     mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
        #     tmrm_idx = 0
        #
        #     # axarr[0, i].imshow(imgs[i][tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
        #     # axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)
        #
        #     vals.append(np.mean(imgs[i][:, :, 0]))
        #     axarr[0, i].set_xticks([])
        #     # for minor ticks
        #     axarr[0, i].set_yticks([])
        #
        #     # axarr[1, i].set_xticks([])
        #     # for minor ticks
        #     # axarr[1, i].set_yticks([])

        #-- Plotting
        f, axarr = plt.subplots(1, len(imgs), figsize=(20, 20))
        f.suptitle("MIP", fontsize=50)
        vals = []
        for i in range(len(imgs)):
            axarr[i].imshow(imgs[i], vmin=0., vmax=1., cmap=plt.cm.viridis)

            # vals.append(np.mean(imgs[i][:, :, 0]))
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
            # print(df['label'][idx])
            # print(label_drug_dict[df['label'][idx]])
            print(df['path'][idx])
            # print(df['frame_id'][idx])
            print("")

        # patches = []
        # for i, idx in enumerate(idxs):
        #     l = label_drug_dict[df['label'][idx]]
        #     ds_id = picked_image_paths[idx].split("/")[-2]
        #     sample_id = picked_image_paths[idx].split("/")[-1].split(".npy")[0]
        #     sample_caption = f"{ds_id}/{sample_id}"
        #     patches.append(mpatches.Patch(color=colors[i], label="{l} ({sc})".format(l=l, sc=sample_caption)))
        #
        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        print("")

        return vis.get_picked_points()

    while True:
        _pick_points()


def setup_time_cmap(df):
    frame_ids = df['frame_id'].unique()
    times = [int(f) for f in frame_ids]

    max_time = max(times)
    min_time = min(times)

    cmap_time = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min_time, vmax=max_time)

    df['cmap_time'] = df['frame_id'].apply(lambda x: cmap_time(norm(int(x)))[:3])
    return df


cfg = load_config("/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml")
proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
save_dir = f"{proj_dir}/runs/"

ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_ablated-tmrm_r20251115_epoch=161-step=28836-val_loss=0.00.ckpt"

embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260202_3color-embeddings_2024v2-4D"

metadata_file = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/andre_3color_cancer/20260129-0/metadata.csv"
df_metadata = pd.read_csv(metadata_file)

# Add a dummy label
# df_metadata['label'] = 0

save_embeddings = False

keep_frames = [-1]
samples_per_class = -1
sample_stride = None # samples every nth frame per sample for 3D; eval'd only if keep_frames and samples_per_class are not set

# -- Projection Config --
reducer = "phate"
# reducer = "umap"

# -- Colors ---
# cmap = "time"
# cmap = "label"
# cmap = "tmrm"
cmap = None

# Seems to show the axis
PHATE_CONFIG = {
    "n_components": 3,
    "knn": 15,
    # "decay": 10,
    "knn_dist": "cosine",
    "mds_dist": "cosine",
    # "mds_solver": "smacof",
    "t": "auto",
    "random_state": 0,
}

# PHATE_CONFIG = {
#     "n_components": 3,
#     "knn": 10,
#     "decay": 10,
#     # "knn_dist": "cosine",
#     # "mds_dist": "cosine",
#     # "t": 'auto',
#     "t": 10,
#     "random_state": 0,
# }

# UMAP_CONFIG = {
#     "n_components": 3,
#     "n_neighbors": 20,
#     "min_dist": 0.01,
#     "random_state": 0,
#     "metric": "cosine",
# }

UMAP_CONFIG = {
    "n_components": 3,
    "n_neighbors": 15,
    "min_dist": 0.1,
    # "init": "spectral",
    "random_state": 0,
    "metric": "cosine",
}

if save_embeddings:
    # Model Setup
    model = Lightweight3DResNet(embedding_size=2048,
                                cfg=cfg,
                                apply_aug=False,
                                decoder_checkpoint_path=None,
                                )

    model = SimCLRRunner.load_from_checkpoint(ckpt_path, model=model, cfg=cfg, strict=False).model
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    os.makedirs(embeddings_dir, exist_ok=True)

    cell_ids = df_metadata['cell_id'].unique()

    embeddings = []
    resnet_embeddings = []
    # for i, frame_path in enumerate(df_metadata['path']):
    # for i, frame_path in tqdm(enumerate(df_metadata['path']), total=len(df_metadata)):
    for i, cell_id in tqdm(enumerate(cell_ids), total=len(cell_ids)):
        df_cell = df_metadata[df_metadata['cell_id'] == cell_id]

        frames = []
        for frame_path in df_cell['path']:
            frame = np.load(frame_path)
            frames.append(frame[None, :, :, :])

        movie = np.stack(frames, axis=0) # T, C, D, H, W
        movie = movie[None, ...] # Add batch dim (B, T, C, D, H, W)

        with torch.no_grad():
            movie = torch.from_numpy(movie).to(device) # (1, C, D, H, W)

            # model expects (B, T, C, D, H, W)
            features, resnet_features, _ = model(movie)
            # features, _ = model(frame) # (1 1 2048)
            features = F.normalize(features, dim=-1)
            resnet_features = F.normalize(resnet_features, dim=-1)
            # features = features[0, :, :] # (2048,)
            embeddings.append(features.detach().cpu().numpy().squeeze(0).astype(np.float32))
            resnet_embeddings.append(resnet_features.detach().cpu().numpy().squeeze(0).astype(np.float32))

    embeddings = np.concatenate(embeddings, axis=0) # (N, D)
    resnet_embeddings = np.concatenate(resnet_embeddings, axis=0)
    np.save(f"{embeddings_dir}/embeddings_raw.npy", embeddings)
    np.save(f"{embeddings_dir}/embeddings_resnet.npy", resnet_embeddings)
    print(f"Saved embeddings to {embeddings_dir}")

    # Add the features to the metadata dataframe
    df_metadata['embedding'] = embeddings.tolist()
    df_metadata.to_parquet(f"{embeddings_dir}/metadata.parquet")
    print(f"Saved metadata to {embeddings_dir}/metadata.parquet")

# Load embeddings
embeddings = np.load(f"{embeddings_dir}/embeddings_raw.npy")
df_metadata = pd.read_parquet(f"{embeddings_dir}/metadata.parquet")

print("WARNING: FILTERING TO LAST FRAME PER CELL ONLY")
length = 56
cell_ids = df_metadata['cell_id'].unique()
keep_idxs = []
n_frames= []
for cell_id in cell_ids:
    df_cell = df_metadata[df_metadata['cell_id'] == cell_id]
    if len(df_cell) < length:
        continue

    if len(df_cell) > length:
        # Keep the 56th frame per cell
        keep_idxs.append(df_cell.index[length-1])
    elif len(df_cell) == length:
        keep_idxs.append(df_cell.index[-1])
    else:
        raise ValueError(f"Unexpected number of frames for cell {cell_id}: {len(df_cell)}")
df_metadata = df_metadata.iloc[keep_idxs].reset_index(drop=True)

# Plot a histogram of the number of frames per cell
# plt.hist(n_frames, bins=50)
# plt.show()
# print("WARNING: FILTERING BY FRAME ID")
# df_metadata = df_metadata[df_metadata['frame_id'] > 50].reset_index(drop=True)

X = np.array(np.stack(df_metadata['embedding'].values))
# X = np.array(embeddings[-1, :])
if reducer == "phate":
    phate_operator = phate.PHATE(**PHATE_CONFIG)
    X_reduced = phate_operator.fit_transform(X)
elif reducer == "umap":
    umap_operator = umap.UMAP(**UMAP_CONFIG)
    X_reduced = umap_operator.fit_transform(X)
else:
    raise ValueError(f"Unknown reducer: {reducer}")

df_metadata[f'{reducer}_embeddings'] = X_reduced.tolist()

df_metadata = setup_time_cmap(df_metadata)

pick_points(df_metadata, reducer=reducer, cmap=cmap)
