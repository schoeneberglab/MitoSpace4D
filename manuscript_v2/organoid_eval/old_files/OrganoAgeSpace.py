import torch
import numpy as np
import os
import random
import yaml
import tqdm
import umap
import argparse
import os.path as osp
from utils.utils import *
import time
from utils.vis import make_mitospace
from train_simclr import SimCLRRunner
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from simclr.models_simple import Lightweight3DResNet

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--checkpoint_path', help='Checkpoint path')
parser.add_argument('--config', default='/tscc/nfs/home/d5agarwal/projects/MitoSpace4D/simclr/config.yaml',
                    type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict')
parser.add_argument('--load_epoch', help='Load weights from this epoch')
parser.add_argument('--save_embeddings', default=False, help='Save embeddings')
parser.add_argument('--visualise_space', default=True, help='Visualise MitoSpace')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/"

    print("Experiment name:", cfg['experiment_name'])
    save_dir = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}"

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=287-step=83534-val_loss=0.00.ckpt"
    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=False)

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )
    model.eval()

    data_folder = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/mitodevXmitospace/processed_data'
    labels = np.load('/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/mitodevXmitospace/labels.npy')
    embeddings = []

    pbar = tqdm.tqdm(total=len(labels))
    for i in range(len(labels)):
        data = np.load(os.path.join(data_folder, f'{i:06d}.npy'))
        data = torch.from_numpy(data).unsqueeze(0).to(device) / 255. # Normalize to [0, 1]
        with torch.no_grad():
            features, _ = model.model(data)
            features  = F.normalize(features, dim=-1)
            features = features[:, -1] # last timestep

        embeddings.append(features.detach().cpu().numpy())
        pbar.update(1)

    embeddings = np.concatenate(embeddings, axis=0)

    #reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
    #embeddings = reducer.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    np.save('/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/OragnoAgeSpace.npy', embeddings)
