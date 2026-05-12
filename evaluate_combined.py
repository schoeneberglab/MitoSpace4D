"""
Consolidated evaluation script combining embedding generation (generate_space.py) and
k-NN / confusion-matrix evaluation (evaluate.py).

UMAP projection / visualization has been removed (now handled by vis.py).
Confusion-matrix visualization is retained.
"""

import argparse
import os
import os.path as osp
import pickle
import random
import yaml

import einops
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from skimage.filters import threshold_otsu
from sklearn.metrics import (
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    pairwise_distances,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from tqdm import tqdm as tqdm_bar

from data.dataset_utils import get_mitospace_data_loaders
from data.mitospace_dataset import *
from simclr.model import Lightweight3DResNet
from train_simclr import SimCLRRunner
from utils.utils import *

np.random.seed(0)
random.seed(0)

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='MitoSpace Combined Evaluation')

parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml',
                    type=str, help='Config path.')

parser.add_argument('--evaluate_set', default='test',
                    type=str, help='Set on which to run evaluation')

parser.add_argument('--dist_metric', default='cosine',
                    type=str, help='Metric to use for distance calculation between embeddings')

parser.add_argument('--labels', nargs='+', type=int, default=None,
                    help='List of labels to evaluate on')

parser.add_argument('--checkpoint_path', help='Checkpoint path',
                    default="/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/checkpoints/ms4d_2024v3_epoch=252-step=27071_best.ckpt"
                    )
parser.add_argument('--data_path', help='Data to predict',
                    default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v3_data/processed_data",
                    )

parser.add_argument('--embeddings_dir', help='Directory to save/load embeddings', default=None)
parser.add_argument('--save_embeddings', default=True, action='store_true', help='Save embeddings')
parser.add_argument('--save_tmrm_intensities', default=False, action='store_true', help='Save TMRM intensities')

parser.add_argument('--cmap', default='label', help='Color map to use.',
                    choices=['label', 'temporal', 'region', 'dataset', 'tmrm'])

parser.add_argument('--to_load', default='all', type=str, choices=["all", "train", "val"],
                    help='Which splits to load. Default is "all".')

parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dataloaders')

parser.add_argument('--random_init', default=False, action='store_true',
                    help='Use a randomly initialized model instead of loading from a checkpoint.')
parser.add_argument('--vary_intensity', default=0.0, type=float,
                    help='Randomly vary image intensity by +/- this normalized fraction per sample (e.g. 0.1 for +/-10%%).')
parser.add_argument('--binarize', default=False,
                    help='Binarize the images by thresholding at 0 using otsu thresholding',
                    action='store_true')
parser.add_argument('--visualize_weights', default=False, help="Visualize model weights", action='store_true')

def get_label_colormap(proj_dir):
    colors = {}
    with open(f"{proj_dir}/metadata/colors.txt", "r") as file:
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

def get_tmrm_intensity(img):
    # (B, T, C, Z, Y, Z)
    raw_mean_intensities = np.zeros(img.shape[1])
    otsu_mean_intensities = np.zeros(img.shape[1])

    for t in range(img.shape[1]):
        thr = threshold_otsu(img[0, t, 1, ...])
        mask = img[0, t, 1, ...] > thr

        raw_mean_intensities[t] = img[0, t, 0, ...].mean()

        img[0, t, 0, ...] = img[0, t, 0, ...] * mask
        otsu_mean_intensities[t] = img[0, t, 0, ...].mean()

    return raw_mean_intensities, otsu_mean_intensities


def img_normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def generate_embeddings(args, cfg, embeddings_dir, drug_labels_dict, label_drug_dict):
    """Generate embeddings using the SimCLR model and save them under embeddings_dir."""
    os.makedirs(embeddings_dir, exist_ok=True)

    # dump all args/configs
    with open(osp.join(embeddings_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    with open(osp.join(embeddings_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    checkpoint_path = args.checkpoint_path
    print(f"Ckpt Path: {args.checkpoint_path}")
    print(f"Data Path: {args.data_path}")

    pick_labels = [args.labels] if args.labels else [
        list(drug_labels_dict.values())]  # Default to all conditions in the drug label dict

    print("labels", pick_labels)

    batch_size = args.batch_size

    emb_raw_path = osp.join(embeddings_dir, 'embeddings.npy')  # (N, 2048) float32
    emb_resnet_path = osp.join(embeddings_dir, 'embeddings_resnet.npy')  # (N, 512) float32
    lbl_path = osp.join(embeddings_dir, 'labels.npy')  # (N,) int32
    img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')  # (N,) object

    # Build and load model
    model = Lightweight3DResNet(embedding_size=2048,
                                cfg=cfg,
                                apply_aug=False,
                                decoder_checkpoint_path=None,
                                )

    if args.random_init:
        print("Using randomly initialized model (no checkpoint loaded).")
    else:
        model = SimCLRRunner.load_from_checkpoint(checkpoint_path, model=model, cfg=cfg, strict=False).model
    model.eval().to(device)

    for param in model.parameters():
        param.requires_grad = False

    data_paths = [args.data_path]
    loaders = []
    for data_path, pick_label in zip(data_paths, pick_labels):
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
    tmrm_pth_list = []
    tmrm_intensities_list = []

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

            # Add channel dimension to the im
            im = np.expand_dims(im, axis=2)  # (B, T, C=1, Z, Y, X)

            if n_frames is None:
                n_frames = im.shape[1]  # Get number of frames

            img_pth_list.extend(img_pth)

            im = img_normalize(im)

            if args.vary_intensity > 0:
                scale = 1.0 + np.random.uniform(-args.vary_intensity, args.vary_intensity)
                im = np.clip(im * scale, 0.0, 1.0)

            if args.binarize:
                thr = threshold_otsu(im)
                im = (im > thr).astype(np.float32)

            with torch.no_grad():
                # model expects (B, T, C, D, H, W)
                features, resnet_features, _ = model(torch.from_numpy(im).to(device), get_resnet_feats=True)
                features = F.normalize(features, dim=-1)  # (B, 2048) or (B, T, 2048)
                resnet_features = F.normalize(resnet_features, dim=-1)  # (B, T, 512)

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

    # Saving image paths (text file with one path per line)
    with open(img_pathfile, 'w') as f:
        for pth in img_pth_list:
            f.write(f"{pth}\n")

    np.save(osp.join(embeddings_dir, 'label_names.npy'), np.array(list(drug_labels_dict.keys())))

def nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs,
                                num_neighbors=[100], verbose=True):
    preds = None

    # normalize the distance matrix
    dist_matrix = (dist_matrix + 1) / 2

    for k in num_neighbors:
        if verbose:
            print(f"################ Evaluation for {k} Neighbors #################")

        correct_preds = {top_n: 0 for top_n in top_ns}
        correct_preds_per_class = {top_n: {lbl: 0 for lbl in np.unique(train_labels)} for top_n in top_ns}
        preds = {top_n: [] for top_n in top_ns}  # overrides for every k: needs to be changed

        # save the indices of the correct and incorrect predictions
        correct_preds_idxs = {top_n: [] for top_n in top_ns}
        incorrect_preds_idxs = {top_n: [] for top_n in top_ns}

        pbar = tqdm_bar(total=len(dist_matrix)) if verbose else None
        for i in range(len(dist_matrix)):
            eval_lbl = eval_labels[i]  # ground truth label
            k_nearest_nbs = train_labels[dist_matrix_idxs[i][:k]]
            k_nearest_dist = dist_matrix[i][:k]

            for top_n in top_ns:
                top_most_freq_lbls = topKfrequent(k_nearest_nbs, k_nearest_dist, top_n, weighted=True)
                if eval_lbl in top_most_freq_lbls:
                    correct_preds[top_n] += 1
                    correct_preds_per_class[top_n][eval_lbl] += 1
                    preds[top_n].append(eval_lbl)

                    # save the indices of the correct predictions
                    correct_preds_idxs[top_n].append(i)

                else:
                    preds[top_n].append(top_most_freq_lbls[0])

                    # save the indices of the incorrect predictions
                    incorrect_preds_idxs[top_n].append(i)

            if pbar is not None:
                pbar.update(1)

        for top_n in top_ns:
            correct = correct_preds[top_n]

            if verbose:
                print(f"------------------Top-{top_n}------------------------")
                print(f"Correct: {correct}; Total: {len(dist_matrix)}")
                acc = correct * 100. / len(eval_labels)
                print("Accuracy(%): ", acc)
                print()

            # print per class accuracy
            for lbl in np.unique(train_labels):
                total = np.sum(eval_labels == lbl)
                correct = correct_preds_per_class[top_n][lbl]
                if verbose:
                    print(
                        f"Class {lbl} has {correct} correct predictions out of {total} samples: Accuracy: {correct * 100. / total}")

    return preds, correct_preds_idxs, incorrect_preds_idxs


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return e_x / e_x.sum(axis=0)


def remove_outliers(embeddings, label_idxs, thresh=2):
    for lbl in label_idxs.keys():
        cluster = embeddings[label_idxs[lbl]]
        centroid = np.mean(cluster, axis=0)[None]

        mean = np.mean(np.linalg.norm(cluster - centroid, axis=1))
        std = np.std(np.linalg.norm(cluster - centroid, axis=1))

        outliers = np.where(np.linalg.norm(cluster - centroid, axis=1) > mean + thresh * std)[0]
        label_idxs[lbl] = np.delete(label_idxs[lbl], outliers)
    return label_idxs

def distance_distribution_metric_evaluation(eval_labels, ref_labels,
                                            eval_embeddings, ref_embeddings, k_neighbors=100):
    eval_label_idxs = {}
    ref_label_idxs = {}

    assert (np.unique(eval_labels) == np.unique(ref_labels)).all()  # checks if the labels are the same
    unique_labels = np.unique(eval_labels)

    for lbl in np.unique(unique_labels):
        eval_label_idxs[lbl] = np.where(eval_labels == lbl)[0]

    for lbl in np.unique(unique_labels):
        ref_label_idxs[lbl] = np.where(ref_labels == lbl)[0]

    # remove outliers
    eval_label_idxs = remove_outliers(eval_embeddings, eval_label_idxs, thresh=2)
    ref_label_idxs = remove_outliers(ref_embeddings, ref_label_idxs, thresh=2)

    eval_label_cluster_centroids = {}
    ref_label_cluster_centroids = {}

    for lbl in np.unique(unique_labels):
        eval_label_cluster_centroids[lbl] = np.mean(eval_embeddings[eval_label_idxs[lbl]], axis=0)
        ref_label_cluster_centroids[lbl] = np.mean(ref_embeddings[ref_label_idxs[lbl]], axis=0)

    eval_label_cluster_centroids = np.array([eval_label_cluster_centroids[lbl] for lbl in np.unique(unique_labels)])
    ref_label_cluster_centroids = np.array([ref_label_cluster_centroids[lbl] for lbl in np.unique(unique_labels)])

    eval_label_cluster_centroids = eval_label_cluster_centroids / np.linalg.norm(eval_label_cluster_centroids, axis=1)[
        :, None]
    ref_label_cluster_centroids = ref_label_cluster_centroids / np.linalg.norm(ref_label_cluster_centroids, axis=1)[:,
    None]

    ref_dist_from_ref_centroid = ref_embeddings @ ref_label_cluster_centroids.T
    eval_dist_from_eval_centroid = eval_embeddings @ eval_label_cluster_centroids.T

    ref_dist_from_ref_centroid = softmax(ref_dist_from_ref_centroid).astype(np.float16)
    eval_dist_from_eval_centroid = softmax(eval_dist_from_eval_centroid).astype(np.float16)

    predictions = []
    batch_size = 1024  # Define the batch size

    num_batches = len(eval_embeddings) // batch_size + (len(eval_embeddings) % batch_size != 0)

    pb = tqdm_bar(total=num_batches)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(eval_embeddings))

        batch_eval_dist_from_eval_centroid = eval_dist_from_eval_centroid[start_idx:end_idx][:, None]
        entropy_dists = entropy(batch_eval_dist_from_eval_centroid,
                                ref_dist_from_ref_centroid[None],
                                axis=-1)

        sorted_idxs = np.argsort(entropy_dists, axis=-1)
        k_nearest_idxs = sorted_idxs[:, :k_neighbors]

        k_nearest_nbs = ref_labels[k_nearest_idxs]

        for sub_batch in k_nearest_nbs:
            freq_dict = {lbl: 0 for lbl in np.unique(unique_labels)}
            for lbl in sub_batch:
                freq_dict[lbl] += 1

            predictions.append(max(freq_dict, key=freq_dict.get))

        pb.update(1)

    accuracy = (predictions == eval_labels).sum() * 100. / len(eval_embeddings)

    print(accuracy)


def cosine_distance(eval_embeddings, train_embeddings, weighted=False, temperature=1.):
    dist_matrix = eval_embeddings @ train_embeddings.T
    if weighted:
        dist_matrix = dist_matrix / temperature
        dist_matrix = np.exp(dist_matrix)

    dist_matrix_idxs = (-1 * dist_matrix).argsort(1)  # because we want to sort in descending order of distances
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    return dist_matrix_sorted, dist_matrix_idxs


def l2_distance(eval_embeddings, train_embeddings):
    dist_matrix = np.linalg.norm(eval_embeddings[:, None] - train_embeddings[None, :], axis=-1)
    dist_matrix = dist_matrix.argsort(1)
    return dist_matrix


def balance_label_counts(embeddings, labels, seed=1123, shuffle=True):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"embeddings has {embeddings.shape[0]} rows but labels has length {labels.shape[0]}")

    rng = np.random.default_rng(seed)

    unique_labels = np.unique(labels)
    label_counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    min_count = min(label_counts.values())

    indices = []
    for lbl in unique_labels:
        lbl_indices = np.where(labels == lbl)[0]
        selected = rng.choice(lbl_indices, size=min_count, replace=False)
        indices.append(selected)

    indices = np.concatenate(indices)

    # Re-shuffle the labels
    if shuffle:
        rng.shuffle(indices)

    return embeddings[indices], labels[indices]


def filter_by_label(pick_labels, embeddings, labels, drug_labels_dict, label_drug_dict):
    drug_labels_dict = {drug: label for drug, label in drug_labels_dict.items() if label in pick_labels}
    label_drug_dict = {label: drug for label, drug in label_drug_dict.items() if label in pick_labels}

    # mask the embeddings and labels
    mask = np.isin(labels, pick_labels)
    embeddings = embeddings[mask]
    labels = labels[mask]
    return embeddings, labels, drug_labels_dict, label_drug_dict


def split_dataset(embeddings, labels, split_perc=0.9, balanced=True, seed=1123):
    if balanced:
        # Split each label separately to maintain class distribution
        unique_labels = np.unique(labels)
        train_indices = []
        val_indices = []
        for lbl in unique_labels:
            lbl_indices = np.where(labels == lbl)[0]
            train_idx, val_idx = train_test_split(lbl_indices, train_size=split_perc, random_state=seed, shuffle=True)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
    else:
        # Split the entire dataset at once
        all_indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(all_indices, train_size=split_perc, random_state=seed,
                                                      shuffle=True)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    return embeddings[train_indices], labels[train_indices], embeddings[val_indices], labels[val_indices]


def plot_cm(gt_labels, pred_labels, label_drug_dict,
            verbose=True, make_plot=True, show_values=True,
            title='Confusion matrix', cmap=None, normalize=True, save_path=None):
    """
    Computes and plots a confusion matrix.

    Args:
        gt_labels: Ground truth labels.
        pred_labels: Predicted labels.
        label_drug_dict: Dictionary mapping class indices to string labels.
        verbose: If True, prints per-class Top-1 accuracy.
        make_plot: If True, renders the matplotlib plot.
        show_values: If True, prints the cell counts inside the matrix squares.
        title: Title of the plot.
        cmap: Colormap to use (defaults to 'Blues').
        normalize: If True, normalizes the colors of the matrix squares.
    """
    # 1. Compute the confusion matrix
    labels_idx = sorted(list(label_drug_dict.keys()))
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels_idx)

    # 2. Print verbose per-class accuracy
    if verbose:
        print("per class accuracy Top-1")
        for i in range(cm.shape[0]):
            row_sum = np.sum(cm[i, :])
            acc = cm[i, i] * 100.0 / row_sum if row_sum > 0 else 0.0
            label_name = label_drug_dict[labels_idx[i]]
            print(f"{label_name}: {acc:.2f}%")

    # 3. Generate the plot
    if make_plot:
        if cmap is None:
            cmap = plt.get_cmap('Blues')

        fig, ax = plt.subplots(figsize=(12, 10))

        label_names = [label_drug_dict[idx] for idx in labels_idx]
        tickmarks = np.arange(len(label_names))
        ax.set_xticks(tickmarks)
        ax.set_xticklabels(label_names, rotation=90)
        ax.set_yticks(tickmarks)
        ax.set_yticklabels(label_names)

        # Create a copy for plot colors so we don't overwrite the returned cm
        if normalize:
            # Add a small epsilon or use np.errstate to avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_plot = np.nan_to_num(cm_plot)
        else:
            cm_plot = cm.copy()

        im = ax.imshow(cm_plot, cmap=cmap, interpolation='nearest',
                       vmin=0.0, vmax=1.0)
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # 4. Optional: Add text values to the matrix squares
        if show_values:
            thresh = cm_plot.max() / 1.5 if cm_plot.max() > 0 else 0.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j],
                            horizontalalignment="center",
                            color="white" if cm_plot[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    return cm

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.update(vars(args))
    proj_dir = "/home/earkfeld/Projects/MitoSpace4D"

    already_have_embeddings = True
    balance_classes = True
    balanced_split = True

    split_perc = 0.8
    top_ns = [1, 3]

    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_testing"

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/metadata/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    if not already_have_embeddings:
        print("Generating embeddings via model inference...")
        generate_embeddings(args, cfg, embeddings_dir, drug_labels_dict, label_drug_dict)

    print("Loading pre-extracted embeddings...")
    embeddings = np.load(
        f'{embeddings_dir}/embeddings.npy')
    labels = np.load(
        f'{embeddings_dir}/labels.npy')

    # Filter the drug label dicts to only include the labels present in the dataset
    unique_labels_in_dataset = set(labels)
    drug_labels_dict = {drug: label for drug, label in drug_labels_dict.items() if
                        label in unique_labels_in_dataset}
    label_drug_dict = {label: drug for label, drug in label_drug_dict.items() if label in unique_labels_in_dataset}

    if args.labels:
        embeddings, labels, drug_labels_dict, label_drug_dict = filter_by_label(args.labels, embeddings, labels,
                                                                                drug_labels_dict, label_drug_dict)

    if balance_classes:
        embeddings, labels = balance_label_counts(embeddings, labels)
        print(f"Balanced to {len(np.unique(labels))} classes with {np.bincount(labels)} samples each")
        print({k: (labels == k).sum() for k in np.unique(labels)})

    train_embeddings, train_labels, eval_embeddings, eval_labels = split_dataset(embeddings,
                                                                                 labels,
                                                                                 split_perc=split_perc,
                                                                                 balanced=balanced_split)

    if len(train_embeddings.shape) > 2:
        train_embeddings = train_embeddings[:, -1, :]  # take only the final time step

    if len(eval_embeddings.shape) > 2:
        eval_embeddings = eval_embeddings[:, -1, :]  # take only the final time step

    # Evaluation on cosine similarity
    if args.dist_metric == 'cosine':
        print("Evaluating full dimensional embeddings using cosine distance")
        dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings, train_embeddings, weighted=False,
                                                        temperature=cfg["training"]["loss"]["temperature"])
        preds, correct_preds_idxs, incorrect_preds_idxs = nearest_neighbor_evaluation(eval_labels,
                                                                                      train_labels,
                                                                                      top_ns,
                                                                                      dist_matrix,
                                                                                      dist_matrix_idxs=dist_matrix_idxs)

        # plot confusion matrix
        cm = plot_cm(eval_labels, preds[1], label_drug_dict,
                     verbose=False,
                     show_values=False,
                     )  # top 1 confusion matrix

        cm = plot_cm(eval_labels, preds[1], label_drug_dict,
                     verbose=False,
                     show_values=True,
                     )  # top 1 confusion matrix with values

        for top_n in top_ns:
            # save csv with columns for drug name, number of correct, number total, and accuracy
            df = pd.DataFrame(columns=['drug', 'label', 'n_correct', 'n_total', 'accuracy'])
            # print per class accuracy
            for lbl in np.unique(train_labels):
                total = np.sum(eval_labels == lbl)
                correct = np.sum(eval_labels[correct_preds_idxs[top_n]] == lbl)
                drug_name = label_drug_dict[lbl]
                accuracy = correct * 100. / total if total > 0 else 0.
                df = pd.concat([df, pd.DataFrame(
                    {'drug': [drug_name], 'label': [lbl], 'n_correct': [correct], 'n_total': [total],
                     'accuracy': [accuracy]})], ignore_index=True)

            # Save the average across all classes
            avg_accuracy = df['accuracy'].mean()
            total_correct = df['n_correct'].sum()
            total_samples = df['n_total'].sum()
            df = pd.concat([df, pd.DataFrame(
                {'drug': ['AVERAGE'], 'label': ['N/A'], 'n_correct': [total_correct], 'n_total': [total_samples],
                 'accuracy': [avg_accuracy]})], ignore_index=True)

    # Evaluate on L2 Distance
    if args.dist_metric == 'l2':
        print("Building Tree")
        tree = KDTree(train_embeddings)
        print("Tree Built")

        num_neighbors = [100]

        accs = []
        for k in num_neighbors:
            dist, nearest_ind = tree.query(eval_embeddings, k=k)
            predicted_labels = []

            for top_n in top_ns:
                correct = 0
                for i in tqdm_bar(range(len(eval_embeddings))):
                    eval_lbl = eval_labels[i]
                    k_nearest_nbs = train_labels[nearest_ind[i]]
                    top_most_freq_lbls = topKfrequent(k_nearest_nbs, top_n)
                    correct += 1 if eval_lbl in top_most_freq_lbls else 0

                print(f"------------------Top-{top_n} Evaluation for {k} Neighbors------------------------")
                print(f"Correct: {correct}; Total: {len(eval_embeddings)}")
                acc = correct * 100. / len(eval_labels)
                accs.append(acc)
                print("Accuracy(%): ", acc)
