import torch
import numpy as np
import os
import random
import yaml
import tqdm
import umap
import argparse
import os.path as osp

from simclr.simclr import load_resnet_model
import time
from utils.vis import make_mitospace
from data_aug.dataset_utils import get_mitospace_data_loaders
from train_simclr import SimCLRRunner
import torch.nn.functional as F
from utils.utils import normalize, load_config
from torch.utils.data import DataLoader
from utils.utils import get_drug_labels, increase_contrast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--checkpoint_path', help='Checkpoint path')
parser.add_argument('--config', default='/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml',
                    type=str, help='Config path.')
parser.add_argument('--data_path', help='Data to predict')
parser.add_argument('--load_epoch', help='Load weights from this epoch')
parser.add_argument('--save_embeddings', default=True, help='Save embeddings')
parser.add_argument('--visualise_space', default=False, help='Visualise MitoSpace')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"

    print("Experiment name:", cfg['experiment_name'])
    save_dir = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}"

    pick_labels = [None, None]

    if args.visualise_space:
        if not osp.exists(f"{save_dir}/embeddings/"):
            print("Embeddings are not saved. Please run the script again with --save_embeddings flag")

        make_mitospace(embedding_dir=f"{save_dir}/embeddings/", pick_labels=pick_labels[0])
        exit()

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=295-step=49136-val_loss=0.00.ckpt"
    model = load_resnet_model(cfg=cfg, ckpt_path=checkpoint_path, device=device, eval_mode=True)

    drug_labels_dict, label_drug_dict = get_drug_labels(f"{proj_dir}/extraction_utils/drugs_to_labels.txt")

    data_paths = ['/home/dhruvagarwal/projects/MitoSpace4D/data/2023_data/']
                  # '/home/dhruvagarwal/projects/MitoSpace/data/Cal27NewHiroAndre/20240507']

    loaders = []
    for data_path, pick_label in zip(data_paths, pick_labels):
        loaders.append(get_mitospace_data_loaders(data_path, shuffle=False, batch_size=6, to_load=["all"], seed=None,
                                                  pick_labels=pick_label,
                                                  samples_per_drug=None)['all'])

    embeddings = []
    labels = []

    for loader_idx, loader in enumerate(loaders):
        for i, batch in tqdm.tqdm(enumerate(iter(loader))):

            if isinstance(batch, list):
                im, lbl = batch[0], batch[1]
            else:
                im, lbl = batch["images"], batch["classes"]

            labels.append(lbl.detach().cpu().numpy())

            with torch.no_grad():
                features, _ = model.model(im.to(0))
                features = F.normalize(features, dim=1)

            embeddings.append(features.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)

    labels = np.concatenate(labels)
    reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
    embeddings = reducer.fit_transform(embeddings.reshape(embeddings.shape[0], -1))

    if args.save_embeddings:
        os.makedirs(osp.join(save_dir, 'embeddings'), exist_ok=True)

        np.save(osp.join(save_dir, 'embeddings', 'embeddings.npy'), embeddings)
        np.save(osp.join(save_dir, 'embeddings', 'labels.npy'), labels)
        # np.save(osp.join(save_dir, 'embeddings', 'images.npy'), images)
        np.save(osp.join(save_dir, 'embeddings', 'label_names.npy'), np.array(list(drug_labels_dict.keys())))

    make_mitospace(embedding_dir=f"{save_dir}/embeddings/", pick_labels=pick_labels)
