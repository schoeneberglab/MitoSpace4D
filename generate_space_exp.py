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
from simclr.models_simple import Lightweight3DResNet # newly trained model (some dict changes)
# from simclr.models_simple_original import Lightweight3DResNet
#
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
                    # default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/2025_kinetics_data/processed_data", # Kinetics Dataset
                    default="/home/earkfeld/Projects/MitoSpace4D/data/2025_kinetics_encoded_data", # Kinetics Dataset Encoded
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data"
                    # default="/mnt/aquila/SSD_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_3b"
                    # default="/run/user/1002/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/leukemia_drug_resistance_data")
                    # default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/cancer_drug_resistance_data/Trial_4",
                    # default="/home/earkfeld/Projects/MitoSpace4D/data/2025_kinetics_encoded_data"
                    )

parser.add_argument('--decoder_ckpt', help='Path to decoder checkpoint',
                    default="/home/earkfeld/Projects/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
                    # default=None,
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

    # embeddings_dir = osp.join("/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classification/deepprofiler_features/PTEN_deepprofiler_pooled-clones")
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classification/deepprofiler_features/2024v2/2024v2_deepprofiler"

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-val-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-all-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm"

    embeddings_dir = osp.join(save_dir, '20260115_kinetics-embeddings_kinetics-resnet3d_ablated_tmrm')
    os.makedirs(embeddings_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    print(f"Ckpt Path: {args.checkpoint_path}")
    print(f"Data Path: {args.data_path}")
    image_paths = None  # define upfront; populated only in visualize pass or left None

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(osp.join(proj_dir, "extraction_utils/drugs_to_labels.txt"), 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug
    
    labels = [args.labels] if args.labels else [list(drug_labels_dict.values())]  # Default to all conditions in the drug label dict
    # labels = [[5]]

    batch_size = args.batch_size
    t_slice = 0
    z_slice = 30

    if args.cmap == 'temporal':
        colors = get_temporal_colormap(embeddings_dir)
    elif args.cmap == 'region':
        colors = get_region_colormap(embeddings_dir)
    elif args.cmap == 'dataset':
        colors = get_dataset_colormap(embeddings_dir)
    elif args.cmap == 'tmrm':
        colors = get_tmrm_colormap(embeddings_dir)
    else:
        colors = get_label_colormap(proj_dir)  # Default to label colormap

    emb_raw_path = osp.join(embeddings_dir, 'embeddings_raw.npy')   # (N, 2048) float32
    emb_resnet_path = osp.join(embeddings_dir, 'embeddings_resnet.npy') # (N, 512) float32
    lbl_path     = osp.join(embeddings_dir, 'labels.npy')           # (N,) int32
    img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')      # (N,) object
    img_times    = osp.join(embeddings_dir, 'image_times.npy')      # (N,) int32
    umap_transform_path = osp.join(embeddings_dir, 'umap_reducer.pkl')

    # Build and load model
    model = Lightweight3DResNet(embedding_size=2048,
                                cfg=cfg, 
                                apply_aug=False, 
                                decoder_checkpoint_path=args.decoder_ckpt,
                                )

    model = SimCLRRunner.load_from_checkpoint(checkpoint_path, model=model, cfg=cfg, strict=False).model
    model.eval().to(device)

    decoder = model.decoder

    for param in model.parameters():
        param.requires_grad = False

    if args.save_embeddings:

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
                )[args.to_load]
            )

        # ---- Accumulate into Python lists; save once at the end ----
        embeddings_list = []
        resnet_embeddings_list = []
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
                if args.decoder_ckpt is None:
                    im = im.permute(0, 2, 1, 3, 4, 5).contiguous()

                # print the data type of im
                # print(f"im dtype: {im.dtype}")

                #-- generating per-frame (3D spatial only) embeddings
                if args.single_frames:
                    # labels for each frame in the batch
                    # print(f"Generating single-frame embeddings for batch {i+1}/{len(loader)} of dataset {loader_idx+1}/{n_datasets}...")
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

                        # model expects (B, T, C, D, H, W)
                        features, resnet_features, _ = model(im.to(device))
                        # print(features.size())
                        features = F.normalize(features, dim=-1)  # (B, 2048) or (B, T, 2048)
                        resnet_features = F.normalize(resnet_features, dim=-1) # (B, T, 512)

                    embeddings_list.extend(features.detach().cpu().numpy().astype(np.float32))
                    resnet_embeddings_list.extend(resnet_features.detach().cpu().numpy().astype(np.float32))
                    labels_list.extend(lbl.detach().cpu().numpy().reshape(-1).astype(np.int32))

                pbar.update(1)
            pbar.close()
        
        # Convert lists to arrays and save once
        embeddings_arr = np.asarray(embeddings_list, dtype=np.float32)
        resnet_embeddings_arr = np.asarray(resnet_embeddings_list, dtype=np.float32)

        labels_arr = np.asarray(labels_list, dtype=np.int32)
        np.save(emb_raw_path, embeddings_arr)
        np.save(emb_resnet_path, resnet_embeddings_arr)
        np.save(lbl_path, labels_arr)
        if args.single_frames and image_times_list is not None:
            image_times_arr = np.asarray(image_times_list, dtype=np.int32)
            np.save(img_times, image_times_arr)

        # Saving image paths (text file with one path per line)
        with open(img_pathfile, 'w') as f:
            for pth in img_pth_list:
                f.write(f"{pth}\n")

        np.save(osp.join(embeddings_dir, 'label_names.npy'), np.array(list(drug_labels_dict.keys())))

        # TODO: integrate colormaps
        # cell_region_map = osp.join(args.data_path, "cell_to_region_new.csv")
        # create_colormap(img_pathfile, cell_region_map, embedding_dir=embeddings_dir)

    if args.reproject or args.save_embeddings:
        # ---- UMAP: fit on subsample, transform all at once; no memmap, no blocks ----
        feats = np.load(emb_raw_path)  # (N, T, D) fully loaded
        # feats = np.load(emb_resnet_path)
        print(f"Features shape: {feats.shape}, dtype: {feats.dtype}")
        
        if not args.single_frames:
            if len(feats.shape) == 3:
                # Get last frame only
                feats = feats[:, -1, :]  # Get last frame
                # feats = np.mean(feats, axis=1)  # Average over time
        else:
            if len(feats.shape) == 3:
                # Flatten (N, T, D) -> (N*T, D)
                feats = einops.rearrange(feats, 'n t d -> (n t) d')

        print(f"Features shape: {feats.shape}, dtype: {feats.dtype}")

        # --- Apply TVN prior to PCA/UMAP if requested ---
        if args.tvn:
            feats = perform_tvn(feats,
                                np.load(lbl_path),
                                np.loadtxt(img_pathfile, dtype=str).tolist(),
                                np.array(list(drug_labels_dict.keys())),
                                scope="global",
                                control_label=args.control_label,)
            # try:
                # labels_all = np.load(lbl_path)  # (N,)
                # ctrl_mask = np.array([label_drug_dict.get(int(l)) == 'control' for l in labels_all], dtype=bool)
                # if ctrl_mask.sum() > 0:
                #     feats = tvn_global(feats, ctrl_mask, eps=1e-6, ledoit_wolf=False).astype(np.float32)
                #     print(f"Applied TVN using {ctrl_mask.sum()} control samples (control).")
                # else:
                #     print("Warning: --tvn set but no control samples found in labels; skipping TVN.")
            # except Exception as e:
            #     print(f"Warning: TVN failed with error: {e}. Proceeding without TVN.")

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

        if args.load_transform:
            print("Loading UMAP transform...")
            reducer = joblib.load(umap_transform_path)
            emb3d = reducer.transform(feats_red).astype(np.float32)
        else:

            # -- original umap
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
                n_neighbors=25,
                min_dist=0.01,
                metric='cosine',
            )

        # #-- Kinetics CuML umap (3D embeddings) - settings for large N
        # reducer = UMAP(
        #     n_components=3,
        #     n_neighbors=25,
        #     min_dist=0.1,
        #     # spread=1.5,
        #     negative_sample_rate=2,
        #     local_connectivity=3,
        #     metric='cosine',
        #     n_epochs=1000,
        #     learning_rate=0.5,
        #     repulsion_strength=1.5,
        #     # init='pca',
        # )

            emb3d = reducer.fit_transform(feats_red).astype(np.float32)
            # Save the transform for future use
            joblib.dump(reducer, umap_transform_path)
        
        # trust_score = trustworthiness(feats_red, emb3d)
        # print(f"UMAP trustworthiness score: {trust_score}")
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
                    datasets=args.datasets,
                    decoder=decoder,
                   )
        exit()
