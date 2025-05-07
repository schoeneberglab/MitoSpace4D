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
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"

    print("Experiment name:", cfg['experiment_name'])
    save_dir = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}"

    image_paths = get_fpaths("/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data")

    pick_labels = [list(range(0, 26))]
    t_slice = 0
    z_slice = 30

    colors = {}
    with open(f"{proj_dir}/extraction_utils/colors_phenotypic.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            print(parts)
            if len(parts) == 6:
                date, label, index, r, g, b = parts
                if float(r) >= 1 or float(g) >= 1 or float(b) >= 1:
                    colors[int(index)] = [float(r) / 255, float(g) / 255, float(b) / 255]
                else:
                    colors[int(index)] = [float(r), float(g), float(b)]

    if args.visualise_space:
        if not osp.exists(f"{save_dir}/embeddings/"):
            print("Embeddings are not saved. Please run the script again with --save_embeddings flag")

        make_mitospace(embedding_dir=f"{save_dir}/embeddings/", pick_labels=pick_labels[0], color_palette=colors,
                       image_paths=image_paths)
        exit()

    # checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=287-step=83534-val_loss=0.00.ckpt"
    # model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
    #                             apply_aug=False)
    #
    # model = SimCLRRunner.load_from_checkpoint(
    #     checkpoint_path, model=model, cfg=cfg
    # )
    # model.eval()
    #
    # drug_labels_dict = {}
    # label_drug_dict = {}
    # with open(f"/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt", 'r') as f:
    #     for line in f:
    #         folder, drug, label = line.split()
    #         drug_labels_dict[drug] = int(label)
    #         label_drug_dict[int(label)] = drug
    #
    # data_paths = ['/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/']
    #
    # loaders = []
    # for data_path, pick_label in zip(data_paths, pick_labels):
    #     loaders.append(get_mitospace_data_loaders(data_path, shuffle=False, batch_size=1, to_load=["all"], seed=None,
    #                                               pick_labels=pick_label,
    #                                               samples_per_drug=None)['all'])
    #
    # embeddings = []
    # labels = []
    # images = []
    #
    # for loader_idx, loader in enumerate(loaders):
    #     pbar = tqdm.tqdm(total=len(loader))
    #     for i, batch in enumerate(iter(loader)):
    #
    #         if isinstance(batch, list):
    #             im, lbl = batch[0], batch[1]
    #         else:
    #             im, lbl = batch["images"], batch["classes"]
    #
    #         print("Making Sample Static")
    #         im = im[:, 0:1].repeat(1, 20, 1, 1, 1, 1)
    #         lbl = lbl*0 + 27
    #         ##########################################
    #
    #         labels.append(lbl.detach().cpu().numpy())
    #
    #         with torch.no_grad():
    #             images.append(im[:, t_slice, :, z_slice].detach().cpu().numpy())
    #             features, _ = model.model(im.to(0))
    #             features = F.normalize(features, dim=-1)
    #
    #         embeddings.append(features.detach().cpu().numpy())
    #         pbar.update(1)

    embeddings = np.load(osp.join(save_dir, 'embeddings', 'embeddings_static_oligo.npy'))
    embeddings = embeddings[:, -1]
    # images = np.load(osp.join(save_dir, 'embeddings', 'images.npy'))
    labels = np.load(osp.join(save_dir, 'embeddings', 'labels_static_oligo.npy'))

    # embeddings = np.concatenate(embeddings)
    # images = np.concatenate(images)
    # labels = np.concatenate(labels)

    reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
    embeddings = reducer.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    # do a pca projection into 10 components
    # pca = PCA(n_components=3)
    # embeddings_pca = pca.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    if args.save_embeddings:
        os.makedirs(osp.join(save_dir, 'embeddings'), exist_ok=True)

        # np.save(osp.join(save_dir, 'embeddings_combined', 'embeddings_umap.npy'), embeddings)
        np.save(osp.join(save_dir, 'embeddings', 'embeddings_umap_static_oligo.npy'), embeddings)
        np.save(osp.join(save_dir, 'embeddings', 'labels.npy'), labels)
        np.save(osp.join(save_dir, 'embeddings', 'images.npy'), images)
        np.save(osp.join(save_dir, 'embeddings', 'label_names.npy'), np.array(list(drug_labels_dict.keys())))

    make_mitospace(embedding_dir=f"{save_dir}/embeddings/", pick_labels=pick_labels)
