import torch
import numpy as np
import os
import random
import yaml
import tqdm
import umap
import argparse
import os.path as osp
import matplotlib.patches as mpatches
import open3d as o3d
import time
from utils.vis import make_mitospace
from data_aug.dataset_utils import get_mitospace_data_loaders
from train_simclr import SimCLRRunner
import torch.nn.functional as F
from utils.utils import normalize, load_config, get_fpaths
from torch.utils.data import DataLoader
from utils.utils import get_drug_label_maps, increase_contrast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from simclr.models_simple import Lightweight3DResNet
from evaluate import cosine_distance, nearest_neighbor_evaluation

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--checkpoint_path', help='Checkpoint path')
parser.add_argument('--config', default='/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml',
                    type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict')
parser.add_argument('--load_epoch', help='Load weights from this epoch')
parser.add_argument('--save_embeddings', default=False, help='Save embeddings')
parser.add_argument('--visualise_space', default=True, help='Visualise MitoSpace')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    get_embeddings = False
    evaluate_result_flag = True
    if get_embeddings:
        args = parser.parse_args()
        cfg = load_config(args.config)
        proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"

        print("Experiment name:", cfg['experiment_name'])
        save_dir = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}"

        checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=287-step=83534-val_loss=0.00.ckpt"
        model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                    apply_aug=False)

        model = SimCLRRunner.load_from_checkpoint(
            checkpoint_path, model=model, cfg=cfg
        )
        model.eval()

        data_folder = '/media/dhruvagarwal/easystore/mitodevXmitospace/processed_data'
        labels = np.load('/media/dhruvagarwal/easystore/mitodevXmitospace/labels.npy')
        embeddings = []

        pbar = tqdm.tqdm(total=len(labels))
        for i in range(len(labels)):
            data = np.load(os.path.join(data_folder, f'{i:06d}.npy'))
            data = torch.from_numpy(data).unsqueeze(0).to(device) / 255.  # Normalize to [0, 1]
            with torch.no_grad():
                features, _ = model.model(data)
                features = F.normalize(features, dim=-1)

            embeddings.append(features.detach().cpu().numpy())
            pbar.update(1)

        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = embeddings[:, -1, :]

        reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
        embeddings = reducer.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

        np.save('/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/OrganoAgeSpace.npy', embeddings)

    else:
        embeddings_mitodev = np.load('/mitodevXmitospace/OrganoAgeSpace.npy')
        embeddings_mitospace = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings.npy')
        labels_mitodev = np.load('/media/dhruvagarwal/easystore/mitodevXmitospace/labels.npy') + 30
        labels_mitospace = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/labels.npy')* 0
        embeddings = np.concatenate([embeddings_mitodev, embeddings_mitospace[:, -1]])
        labels = np.concatenate([labels_mitodev, labels_mitospace])

        color_palette = {
            30: [0, 0, 0],  # Black
            31: [186, 85, 211],  # Medium Orchid (Soft Purple)
            32: [148, 0, 211],  # Dark Violet (Deep Purple)
            45: [0, 0, 255],  # Bright Blue
            46: [30, 144, 255],  # Dodger Blue
            47: [100, 149, 237],  # Cornflower Blue
            50: [70, 130, 180],  # Steel Blue
            97: [0, 255, 0],  # Bright Green
            98: [50, 205, 50],  # Lime Green
            161: [255, 0, 0],  # Bright Red
            162: [220, 20, 60],  # Crimson
            0: [128, 128, 0],  # Yellow
        }

        colors = [np.array(color_palette[label]) / 255. for label in labels]

        reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
        embeddings = reducer.fit_transform(embeddings)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(embeddings)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create legend
        legend_patches = []
        for key, value in color_palette.items():
            legend_patches.append(mpatches.Patch(color=[x / 255 for x in value], label=key))
        plt.figure(figsize=(10, 10))
        plt.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, 0.5))
        plt.axis('off')
        plt.show()

        o3d.visualization.draw_geometries([pcd])
