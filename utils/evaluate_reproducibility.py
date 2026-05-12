import pickle
import random

import numpy as np
from umap import UMAP
from tqdm import tqdm
from sklearn.neighbors import KDTree
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, confusion_matrix, pairwise_distances
import torch.nn.functional as F

from data_aug.mitospace_dataset import *
from simclr.conv3d_lstm import *
import argparse

from simclr.models_simple import Lightweight3DResNet
from utils.utils import *
from data_aug.dataset_utils import get_mitospace_data_loaders
import torch
from train_simclr import SimCLRRunner

from utils.vis_original import plot_cm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.stats import entropy

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='MitoSpace Evaluation')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--config', default='/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml',
                    type=str, help='Config path.')
parser.add_argument('--evaluate_set', default='test',
                    type=str, help='Set on which to run evaluation')
parser.add_argument('--dist_metric', default='cosine',
                    type=str, help='Metric to use for distance calculation between embeddings')
parser.add_argument('--labels', nargs='+', type=int, default=None,
                    help='List of labels to evaluate on')


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

        pbar = tqdm(total=len(dist_matrix)) if verbose else None
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

    pb = tqdm(total=num_batches)
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


def extract_embeddings_from_model(dataloader, model, normalize_embeddings=True, get_images=False, get_labels=False,
                                  messup_tmrm=False, visualise_model_layer=True, get_fpaths=False):
    embeddings = []

    images = [] if get_images else None
    labels = [] if get_labels else None
    im_paths = [] if get_fpaths else None

    pbar = tqdm(total=len(dataloader))

    if visualise_model_layer:
        prefinal_activation = []

        def get_activation(name):
            def hook(model, input, output):
                prefinal_activation.append(output.detach().cpu().numpy())

            return hook

        model.backbone.fc[0].register_forward_hook(get_activation('prefinal'))

    for batch in dataloader:
        if isinstance(batch, list):
            im, lbl, im_path = batch
        else:
            im, lbl, im_path = batch["images"], batch["classes"], batch["image_paths"]

        if messup_tmrm:
            tmrm_idx = 0
            im[:, tmrm_idx] = (im[:, tmrm_idx] - im[:, tmrm_idx].min()) / (
                    im[:, tmrm_idx].max() - im[:, tmrm_idx].min())

        with torch.no_grad():
            # with torch.autocast(device_type="cuda"):
            # with torch.amp.autocast(device_type='cuda'):
            # im = 2 * im - 1  # zero mean normalization
            features, _ = model(im.to('cuda'))

        if normalize_embeddings:
            features = F.normalize(features, dim=-1)

        embeddings.append(features.detach().cpu().numpy())

        if get_images:
            images.append(im.detach().cpu().numpy())
        if get_labels:
            labels.append(lbl.detach().cpu().numpy())
        if get_fpaths:
            im_paths.append(im_path)

        pbar.update(1)

    embeddings = np.concatenate(embeddings)
    # embeddings = np.concatenate(prefinal_activation)

    if get_images:
        images = np.concatenate(images)
    if get_labels:
        labels = np.concatenate(labels)
    if get_fpaths:
        im_paths = np.concatenate(im_paths)

    return embeddings, images, labels, im_paths


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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# def plot_cm(gt_labels, pred_labels, label_drug_dict,
#             verbose=True, make_plot=True, show_values=True,
#             title='Confusion matrix', cmap=None, normalize=True,
#             vmin=None, vmax=None, save_path=None):
#     """
#     Computes and plots a confusion matrix.
#
#     Args:
#         gt_labels: Ground truth labels.
#         pred_labels: Predicted labels.
#         label_drug_dict: Dictionary mapping class indices to string labels.
#         verbose: If True, prints per-class Top-1 accuracy.
#         make_plot: If True, renders the matplotlib plot.
#         show_values: If True, prints the cell counts inside the matrix squares.
#         title: Title of the plot.
#         cmap: Colormap to use (defaults to 'Blues').
#         normalize: If True, normalizes the colors of the matrix squares.
#         vmin, vmax: Min and max values for the colormap scaling.
#     """
#     # 1. Compute the confusion matrix
#     labels_idx = sorted(list(label_drug_dict.keys()))
#     cm = confusion_matrix(gt_labels, pred_labels, labels=labels_idx)
#
#     # 2. Print verbose per-class accuracy
#     if verbose:
#         print("per class accuracy Top-1")
#         for i in range(cm.shape[0]):
#             row_sum = np.sum(cm[i, :])
#             acc = cm[i, i] * 100.0 / row_sum if row_sum > 0 else 0.0
#             label_name = label_drug_dict[labels_idx[i]]
#             print(f"{label_name}: {acc:.2f}%")
#
#     # 3. Generate the plot
#     if make_plot:
#         if cmap is None:
#             cmap = plt.get_cmap('Blues')
#
#         plt.figure(figsize=(10, 10), constrained_layout=True)
#
#         label_names = [label_drug_dict[idx] for idx in labels_idx]
#         tickmarks = np.arange(len(label_names))
#         plt.xticks(tickmarks, label_names, rotation=90)
#         plt.yticks(tickmarks, label_names)
#
#         # Create a copy for plot colors so we don't overwrite the returned cm
#         if normalize:
#             # Add a small epsilon or use np.errstate to avoid division by zero
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#                 cm_plot = np.nan_to_num(cm_plot)
#         else:
#             cm_plot = cm.copy()
#
#         im = plt.imshow(cm_plot, cmap=cmap, interpolation='nearest',
#                         vmin=vmin, vmax=vmax)
#         plt.title(title)
#         plt.colorbar(im)
#
#         # 4. Optional: Add text values to the matrix squares
#         if show_values:
#             thresh = cm_plot.max() / 1.5 if cm_plot.max() > 0 else 0.0
#             for i in range(cm.shape[0]):
#                 for j in range(cm.shape[1]):
#                     plt.text(j, i, cm[i, j],
#                              horizontalalignment="center",
#                              color="white" if cm_plot[i, j] > thresh else "black")
#
#         plt.ylabel('True label')
#         plt.xlabel('Predicted label')
#         if save_path is not None:
#             plt.savefig(save_path)
#         else:
#             plt.show()
#
#     return cm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm  # 1. Import LogNorm


def plot_cm(gt_labels, pred_labels, label_drug_dict,
            verbose=True, make_plot=True, show_values=True,
            title='Confusion matrix', cmap=None, normalize=True,
            vmin=0., vmax=1., save_path=None):
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
        vmin, vmax: Min and max values for the colormap scaling.
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

        plt.figure(figsize=(10, 10), constrained_layout=True)

        label_names = [label_drug_dict[idx] for idx in labels_idx]
        tickmarks = np.arange(len(label_names))
        plt.xticks(tickmarks, label_names, rotation=90)
        plt.yticks(tickmarks, label_names)

        # Create a copy for plot colors so we don't overwrite the returned cm
        if normalize:
            # Add a small epsilon or use np.errstate to avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_plot = np.nan_to_num(cm_plot)
        else:
            cm_plot = cm.copy()

        # 2. Use norm=LogNorm() instead of passing vmin/vmax directly to imshow
        # Note: LogNorm doesn't handle absolute zero well (log(0) is undefined).
        # Matplotlib will automatically mask 0 values, rendering them as the lowest color.
        im = plt.imshow(cm_plot, cmap=cmap, interpolation='nearest',
                        # norm=LogNorm(vmin=vmin, vmax=vmax)
                        )

        plt.title(title)
        plt.colorbar(im)

        # 4. Optional: Add text values to the matrix squares
        if show_values:
            # For log scale, you might want to adjust the text color threshold.
            # We use a lower threshold here to account for the logarithmic visual shift.
            thresh = cm_plot.max() / 10.0 if cm_plot.max() > 0 else 0.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j],
                             horizontalalignment="center",
                             color="white" if cm_plot[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    return cm

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/earkfeld/Projects/MitoSpace4D"

    already_have_embeddings = True
    balance_classes = True
    balanced_split = True

    split_perc = 0.8
    # top_ns = cfg["evaluate"]["top_ns"]
    top_ns = [1, 3]

    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_reproducibility_252eps'
    embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_reproducibility_mitobright_252eps'

    # cmat_title = "Top1 KNN Confusion Matrix - MitoTracker DeepRed"
    cmat_title = "Top1 KNN Confusion Matrix - MitoBright"


    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    # args.labels = [33, 34]
    # args.labels = [33, 34, 35]

    if already_have_embeddings:
        print("Loading pre-extracted embeddings...")
        embeddings = np.load(
            f'{embeddings_dir}/embeddings.npy')
        # f'{embeddings_dir}/embeddings_resnet.npy')
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
            # balance_label_counts(embeddings, labels)
            print(f"Balanced to {len(np.unique(labels))} classes with {np.bincount(labels)} samples each")
            print({k: (labels == k).sum() for k in np.unique(labels)})

        # shuffle them with seed 1123
        # random.seed(1123)
        # data = list(zip(embeddings, labels))
        # random.shuffle(data)
        # embeddings, labels = zip(*data)
        # embeddings, labels = np.array(embeddings), np.array(labels)

        # len_all_data = round(len(labels) * 1.)
        # train_split = round(len_all_data * split_perc)
        # val_split = round(len_all_data * (1 - split_perc))
        #
        # train_embeddings, eval_embeddings = embeddings[:train_split], embeddings[train_split: train_split + val_split]
        # train_labels, eval_labels = labels[:train_split], labels[train_split: train_split + val_split]

        train_embeddings, train_labels, eval_embeddings, eval_labels = split_dataset(embeddings,
                                                                                     labels,
                                                                                     split_perc=split_perc,
                                                                                     balanced=balanced_split)

        # train_embeddings = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings.npy')
        # train_labels = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/labels.npy')
        #
        # eval_embeddings = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings_5/embeddings.npy')
        # eval_labels = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings_5/labels.npy')

        if len(train_embeddings.shape) > 2:
            train_embeddings = train_embeddings[:, -1, :]  # take only the final time step

        if len(eval_embeddings.shape) > 2:
            eval_embeddings = eval_embeddings[:, -1, :]  # take only the final time step

    else:
        print("Generating embeddings...")
        model = Lightweight3DResNet(embedding_size=2048,
                                    cfg=cfg,
                                    # cfg_aug=cfg['data_params']['transforms'],
                                    apply_aug=False)

        # checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=21-step=14212-val_loss=0.00.ckpt"
        # checkpoint_path = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitoSpace4D_resnetbilstm_encoded_normal_eps287.ckpt"
        # checkpoint_path = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_2024v2_decoupled-tmrm_r20251115_epoch=138-step=24742-val_loss=0.00.ckpt"
        checkpoint_path = None
        dataset_name = cfg["evaluate"]["dataset"]

        # print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

        model = SimCLRRunner.load_from_checkpoint(
            checkpoint_path, model=model, cfg=cfg
        )
        model.eval()

        loaders_reference = get_mitospace_data_loaders(
            f'{proj_dir}/data/2024_subdata/',
            shuffle=False, batch_size=2, to_load=["train"],
            timesteps=cfg['data_params']['timesteps'],
            zstacks=cfg['data_params']['zstacks'],
            samples_per_drug=cfg['data_params']['samples_per_drug'],
            pick_labels=None)

        loaders_eval = get_mitospace_data_loaders(
            f'{proj_dir}/data/2024_subdata/',
            shuffle=False, batch_size=2, to_load=["val"],
            timesteps=cfg['data_params']['timesteps'],
            zstacks=cfg['data_params']['zstacks'],
            samples_per_drug=cfg['data_params']['samples_per_drug'],
            pick_labels=None
        )

        train_loader, eval_loader = (loaders_reference["train"], loaders_eval["val"])

        train_embeddings, train_images, train_labels, train_im_paths = extract_embeddings_from_model(train_loader,
                                                                                                     model.model,
                                                                                                     normalize_embeddings=True,
                                                                                                     get_images=False,
                                                                                                     get_labels=True,
                                                                                                     messup_tmrm=False,
                                                                                                     visualise_model_layer=False,
                                                                                                     get_fpaths=True)

        eval_embeddings, eval_images, eval_labels, eval_im_paths = extract_embeddings_from_model(eval_loader,
                                                                                                 model.model,
                                                                                                 normalize_embeddings=True,
                                                                                                 get_images=False,
                                                                                                 get_labels=True,
                                                                                                 messup_tmrm=False,
                                                                                                 visualise_model_layer=False,
                                                                                                 get_fpaths=True)

    # eval_embeddings, eval_images, eval_labels = train_embeddings, train_images, train_labels

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

        # with open(f'{proj_dir}/correct_preds_idxs.pkl', 'wb') as f:
        #     pickle.dump(correct_preds_idxs, f)
        # with open(f'{proj_dir}/incorrect_preds_idxs.pkl', 'wb') as f:
        #     pickle.dump(incorrect_preds_idxs, f)

        #

        # plot confusion matrix
        # cm = plot_cm(eval_labels, preds[1], label_drug_dict, verbose=False)  # top 1 confusion matrix
        cm = plot_cm(eval_labels, preds[1], label_drug_dict,
                     # vmax=1.0,
                     # vmin=0.,
                     verbose=False,
                     show_values=False,
                     title=cmat_title,
                     # save_path=f'{embeddings_dir}/knn-top1_confusion-matrix.png'
                     )  # top 1 confusion matrix
        # save the confusion_matrix to a .png file in the embeddings_dir
        # plt.savefig(f'{embeddings_dir}/confusion-matrix_knn-top1.png')

        cm = plot_cm(eval_labels, preds[1], label_drug_dict,
                     # vmax=1.0,
                     # vmin=0.,
                     verbose=False,
                     show_values=True,
                     title=cmat_title,
                     # save_path=f'{embeddings_dir}/knn-top1_confusion-matrix_with_values.png'
                     )  # top 1 confusion matrix with values
        # plt.savefig(f'{embeddings_dir}/confusion-matrix_knn-top1_with_values.png')

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

            # df.to_csv(f'{embeddings_dir}/knn-top{top_n}_per_drug_accuracy.csv', index=False)

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
                for i in tqdm(range(len(eval_embeddings))):
                    eval_lbl = eval_labels[i]
                    k_nearest_nbs = train_labels[nearest_ind[i]]
                    top_most_freq_lbls = topKfrequent(k_nearest_nbs, top_n)
                    correct += 1 if eval_lbl in top_most_freq_lbls else 0

                print(f"------------------Top-{top_n} Evaluation for {k} Neighbors------------------------")
                print(f"Correct: {correct}; Total: {len(eval_embeddings)}")
                acc = correct * 100. / len(eval_labels)
                accs.append(acc)
                print("Accuracy(%): ", acc)

    # Measure cluster quality in both high and low dimensional space
    reducer = UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.1, metric='cosine')
    all_embeddings = np.concatenate([train_embeddings, eval_embeddings])
    all_embeddings_reduced = reducer.fit_transform(all_embeddings.reshape(all_embeddings.shape[0], -1))
    all_embeddings_reduced = all_embeddings_reduced / np.linalg.norm(all_embeddings_reduced, axis=1)[:, None]

    train_embeddings_reduced = all_embeddings_reduced[:len(train_embeddings)]
    eval_embeddings_reduced = all_embeddings_reduced[len(train_embeddings):]

    # # Evaluation on cosine similarity
    # if args.dist_metric == 'cosine':
    #     print("Evaluating umap projected embeddings using cosine distance")
    #     dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings_reduced, train_embeddings_reduced)
    #     preds = nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, num_neighbors=[100], dist_matrix_idxs=dist_matrix_idxs)
    #
    #     # plot confusion matrix
    #     # cm = plot_cm(eval_labels, preds[1], label_drug_dict)
    #
    # # Evaluate on L2 Distance
    # if args.dist_metric == 'l2':
    #     print("Building Tree")
    #     tree = KDTree(train_embeddings_reduced)
    #     print("Tree Built")
    #
    #     num_neighbors = [100]
    #
    #     accs = []
    #     for k in num_neighbors:
    #         dist, nearest_ind = tree.query(eval_embeddings_reduced, k=k)
    #         predicted_labels = []
    #
    #         for top_n in top_ns:
    #             correct = 0
    #             for i in tqdm(range(len(eval_embeddings_reduced))):
    #                 eval_lbl = eval_labels[i]
    #                 k_nearest_nbs = train_labels[nearest_ind[i]]
    #                 top_most_freq_lbls = topKfrequent(k_nearest_nbs, top_n)
    #                 correct += 1 if eval_lbl in top_most_freq_lbls else 0
    #
    #             print(f"------------------Top-{top_n} Evaluation for {k} Neighbors------------------------")
    #             print(f"Correct: {correct}; Total: {len(eval_embeddings_reduced)}")
    #             acc = correct * 100. / len(eval_labels)
    #             accs.append(acc)
    #             print("Accuracy(%): ", acc)

    original_distances = pairwise_distances(all_embeddings)
    reduced_distances = pairwise_distances(all_embeddings_reduced)

    # Normalize distances (optional but recommended)
    original_distances_normalized = (original_distances - np.min(original_distances)) / (
            np.max(original_distances) - np.min(original_distances))
    reduced_distances_normalized = (reduced_distances - np.min(reduced_distances)) / (
            np.max(reduced_distances) - np.min(reduced_distances))

    # Flatten the distance matrices to 1D arrays
    original_distances_flat = original_distances_normalized.flatten()
    reduced_distances_flat = reduced_distances_normalized.flatten()

    # Compute KL divergence
    kl_divergence = entropy(original_distances_flat, reduced_distances_flat)

    print("KL Divergence:", kl_divergence)