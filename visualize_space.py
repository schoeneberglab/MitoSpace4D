"""
Created 202620117 by Eric Arkfeld

No Model eval; visualization only using phate for down projection

set up for visualizing 3D embeddings across all frames.
consolidated visualization routines from vis.py
"""

import torch
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
        time_indices = df['frame_id'].iloc[idxs]
        picked_image_paths = df['img_path'].iloc[idxs]

        imgs = []
        for idx in idxs:
            if picked_image_paths is not None:
                img_4d = np.load(df['img_path'].iloc[idx])  # (t, c, d, h, w)

                if decoder is not None:
                    img_tensor = torch.from_numpy(img_4d).unsqueeze(0).cuda()  # (1, t, c, d, h, w)
                    with torch.no_grad():
                        img_tensor = decoder(img_tensor)
                    img_4d = img_tensor.squeeze(0).cpu().numpy()
                    img_4d = img_4d.astype(np.float32)

                if is_4d:
                    imgs.append(img_4d[:, -1, ...].max(axis=1))  # MIP last frame
                else:
                    imgs.append(img_4d[:, df['frame_id'].iloc[idx], ...].max(axis=1)) # MIP specified frame

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

        f, axarr = plt.subplots(2, len(imgs), figsize=(20, 20))
        f.suptitle("MIP", fontsize=50)
        vals = []
        for i in range(len(imgs)):
            mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
            tmrm_idx = 0

            axarr[0, i].imshow(imgs[i][tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
            axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)

            vals.append(np.mean(imgs[i][:, :, 0]))
            axarr[0, i].set_xticks([])
            # for minor ticks
            axarr[0, i].set_yticks([])

            # axarr[1, i].set_xticks([])
            # for minor ticks
            # axarr[1, i].set_yticks([])

        print(vals)
        plt.xticks([]), plt.yticks([])
        for idx in idxs:
            print(idx)
            print(df['label'][idx])
            print(label_drug_dict[df['label'][idx]])
            print(df['img_path'][idx])
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

def smooth_trajectory(group, window_size=10):
    group = group.sort_values(by='time').reset_index(drop=True)
    embeddings = np.stack(group['embedding'].values, axis=0)  # (T, D)
    smoothed_embeddings = np.copy(embeddings)
    half_window = window_size // 2
    for i in range(len(embeddings)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(embeddings), i + half_window + 1)
        smoothed_embeddings[i] = np.mean(embeddings[start_idx:end_idx], axis=0)
    group['embedding'] = list(smoothed_embeddings)
    return group

if __name__ == '__main__':
    # TODO: Set up to read mostly from config, optionally overwrite w/ args, add specific visualization entries, and copy updated to the embedding dir
    # Goal: be able to just provide embeddings dir and use previous settings stored there in a copy of the config file
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.update(vars(args))

    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    save_dir = f"{proj_dir}/runs/"

    # Kinetics 3D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm/"

    # Kinetics 4D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_kinetics-4D-embeddings_2024v2-161eps_ablated-tmrm"

    # 2024v2 3D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"

    # 2024v2 4D
    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-4D-embeddings_2024v2-161eps_ablated-tmrm"

    # Liver 4D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_4D-embeddings_2024v2-model"

    # Liver 3D
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_3D-embeddings_Kinetics3D-model"

    # metadata_file = "/home/earkfeld/Projects/MitoSpace4D/experiments/3DMS_phate_visualization/phate_kinetics_metadata.csv"
    # metadata_file = "/home/earkfeld/Projects/MitoSpace4D/experiments/3DMS_phate_visualization/phate_2024v2_metadata.csv"
    metadata_file = "/experiments/3DMS_phate_visualization/metadata/phate_liver_metadata.csv"

    # -- Sampling config (requires generating phate data!) --
    # Note: Ordered by execution priority. Use is exclusive.
    generate_phate_data = True
    keep_frames = [-1]
    samples_per_class = -1
    sample_stride = None # samples every nth frame per sample for 3D; eval'd only if keep_frames and samples_per_class are not set

    # -- Projection Config --
    reproject = True
    reducer = "umap" # "phate"

    # -- Trajectory Config --
    mean_trajectories = False # True
    smoothing_window_size = None # 10 # None

    pick_labels = None # [0] # [5]

    # -- Colors ---
    # cmap = "time"
    # cmap = "label"
    cmap = "tmrm"

    frames_per_region = 60 if "kinetics" in metadata_file else 20
    frames_per_movie = 20

    # Phate trajectory config
    PHATE_CONFIG = {
        "n_components": 3,
        "knn": 100,
        "decay": 40,
        "knn_dist": "cosine",
        "mds_dist": "cosine",
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

    UMAP_CONFIG = {
        "n_components": 3,
        "n_neighbors": 25,
        "min_dist": 0.1,
        "random_state": 0,
        "metric": "cosine",
    }

    decoder = None

    has_phate_data = osp.exists(osp.join(embeddings_dir, "phate_visualization.parquet"))

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(osp.join(proj_dir, "extraction_utils/drugs_to_labels.txt"), 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    batch_size = args.batch_size
    t_slice = 0
    z_slice = 30

    if not has_phate_data or generate_phate_data:
        print(f"Generating phate data for {embeddings_dir}...")

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

        # Set up label colors
        color_palette = get_label_colormap(proj_dir)
        df_data['cmap_label'] = df_data['label'].map(color_palette)

        # Set up a colormap by time
        cmap_time = plt.get_cmap('viridis')
        time_min = df_data['time'].min()
        time_max = df_data['time'].max()
        df_data['cmap_time'] = df_data['time'].apply(lambda x: cmap_time((x - time_min) / (time_max - time_min))[:3])

        df_data.to_parquet(osp.join(embeddings_dir, "phate_visualization.parquet"))
        print(f"Phate embeddings saved to {osp.join(embeddings_dir, 'phate_visualization.parquet')}")

        # -- Set up trajectories
        if mean_trajectories:
            df_data = df_data.groupby(['label', 'time']).agg({
                'embedding': lambda x: np.mean(np.stack(x.values), axis=0),
                'img_path': 'first',
                'label_name': 'first',
                'frame_id': 'first',
                'cmap_label': 'first',
                'cmap_time': 'first',
                'tmrm': lambda x: np.mean(np.stack(x.values), axis=0),
                'label': 'first',
                'time': 'first',
            }).reset_index(drop=True)

            # df_data = df_data.groupby('label').apply(smooth_trajectory).reset_index(drop=True)
            # Pass the variable explicitly using a lambda
            if smoothing_window_size is not None and smoothing_window_size > 0:
                print(f"Smoothing trajectory using a window size of {smoothing_window_size}...")
                df_data = df_data.reset_index()
                df_data = df_data.groupby('label').apply(
                    lambda x: smooth_trajectory(x, window_size=smoothing_window_size),
                ).reset_index(drop=True)

            # Create a column for the next_point index for line drawing
            df_data['next_point'] = df_data.groupby('label').cumcount() + 1
            df_data.loc[
                df_data.groupby('label')['next_point'].idxmax(), 'next_point'] = -1  # Last point has no next point

        if keep_frames is not None:
            # Filter the dataframe to only keep the specified frames
            frame_ids = np.array(sorted(np.unique(df_data['frame_id'])))
            keep_frame_ids = frame_ids[keep_frames]
            print(f"Keeping only frames {keep_frame_ids}...")
            df_data = df_data[df_data['frame_id'].isin(keep_frame_ids)].reset_index(drop=True)
        else:
            if sample_stride is not None and sample_stride > 0:
                print(f"Sampling every {sample_stride} time points...")
                df_data = df_data[df_data['time'] % sample_stride == 0].reset_index(drop=True)

            elif samples_per_class is not None and samples_per_class > 0:
                print(f"Sampling {samples_per_class} points per class...")
                df_data = df_data.groupby(['label']).apply(lambda x: x.sample(samples_per_class)).reset_index(drop=True)

        # Set up colormap by TMRM intensity (doing it last for best colormap for the given config)
        tmrm_max = df_data['tmrm'].max()
        tmrm_min = df_data['tmrm'].min()
        cmap_tmrm = plt.get_cmap('viridis')
        # df_data['cmap_tmrm'] = df_data['tmrm'].apply(lambda x: cmap_tmrm((x - tmrm_min) / (tmrm_max - tmrm_min))[:3])
        df_data['cmap_tmrm'] = df_data['tmrm'].apply(lambda x: cmap_tmrm((np.log1p(x) - tmrm_min) / (tmrm_max - tmrm_min))[:3])

        import seaborn as sns
        # sns.histplot(df_data['cmap_tmrm'], kde=True)
        plt.show()

    else:
        print(f"Loading phate data from {osp.join(embeddings_dir, 'phate_visualization.parquet')}...")
        df_data = pd.read_parquet(osp.join(embeddings_dir, "phate_visualization.parquet"))

    # Check for any entries with duplicate embedding values

    if reproject or f"{reducer}_embeddings" not in df_data.columns:
        print(f"Projecting embeddings using {reducer}... ")

        X = np.stack(df_data['embedding'].values, axis=0)
        print(f"Embeddings shape: {X.shape}, dtype: {X.dtype}")

        if reducer == "phate":
            phate_operator = phate.PHATE(**PHATE_CONFIG)
            X_reduced = phate_operator.fit_transform(X)
        elif reducer == "umap":
            umap_operator = umap.UMAP(**UMAP_CONFIG)
            X_reduced = umap_operator.fit_transform(X)
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

        df_data[f"{reducer}_embeddings"] = list(X_reduced)
        df_data.to_parquet(osp.join(embeddings_dir, "phate_visualization.parquet"))

    if pick_labels:
        print(f"Displaying labels: {pick_labels}")
        df_data = df_data[df_data['label'].isin(pick_labels)].reset_index(drop=True)

    # generate_phate_trajectories(df_data, output_dir="kinetics_3d-phate_visualizations")
    pick_points(df_data, reducer=reducer, cmap=cmap, decoder=decoder)
