import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    pairwise_distances,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from data_aug.dataset_utils import get_mitospace_data_loaders
from data_aug.mitospace_dataset import *
from simclr.conv3d_lstm import *
from simclr.model import Lightweight3DResNet
from train_simclr import SimCLRRunner
from utils.utils import *
from utils.vis_original import plot_cm

global device
device = "cuda" if torch.cuda.is_available() else "cpu"
import torch.multiprocessing
from scipy.stats import entropy

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser(description="MitoSpace Evaluation")
parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")
parser.add_argument(
    "--config",
    default="/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml",
    type=str,
    help="Config path.",
)
parser.add_argument(
    "--evaluate_set", default="test", type=str, help="Set on which to run evaluation"
)
parser.add_argument(
    "--dist_metric",
    default="cosine",
    type=str,
    help="Metric to use for distance calculation between embeddings",
)
parser.add_argument(
    "--labels", nargs="+", type=int, default=None, help="List of labels to evaluate on"
)
parser.add_argument(
    "--test_date",
    default=None,
    type=str,
    help="Normalized date (YYYYMMDD, no -N suffix) to hold out as the test set",
)


def nearest_neighbor_evaluation(
    eval_labels,
    train_labels,
    top_ns,
    dist_matrix,
    dist_matrix_idxs,
    num_neighbors=[100],
    verbose=True,
):
    preds = None

    # normalize the distance matrix
    dist_matrix = (dist_matrix + 1) / 2

    for k in num_neighbors:
        if verbose:
            print(f"################ Evaluation for {k} Neighbors #################")

        correct_preds = {top_n: 0 for top_n in top_ns}
        correct_preds_per_class = {
            top_n: {lbl: 0 for lbl in np.unique(train_labels)} for top_n in top_ns
        }
        preds = {
            top_n: [] for top_n in top_ns
        }  # overrides for every k: needs to be changed

        # save the indices of the correct and incorrect predictions
        correct_preds_idxs = {top_n: [] for top_n in top_ns}
        incorrect_preds_idxs = {top_n: [] for top_n in top_ns}

        pbar = tqdm(total=len(dist_matrix)) if verbose else None
        for i in range(len(dist_matrix)):
            eval_lbl = eval_labels[i]  # ground truth label
            k_nearest_nbs = train_labels[dist_matrix_idxs[i][:k]]
            k_nearest_dist = dist_matrix[i][:k]

            for top_n in top_ns:
                top_most_freq_lbls = topKfrequent(
                    k_nearest_nbs, k_nearest_dist, top_n, weighted=True
                )
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
                acc = correct * 100.0 / len(eval_labels)
                print("Accuracy(%): ", acc)
                print()

            # print per class accuracy
            for lbl in np.unique(train_labels):
                total = np.sum(eval_labels == lbl)
                correct = correct_preds_per_class[top_n][lbl]
                if verbose:
                    print(
                        f"Class {lbl} has {correct} correct predictions out of {total} samples: Accuracy: {correct * 100. / total}"
                    )

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

        outliers = np.where(
            np.linalg.norm(cluster - centroid, axis=1) > mean + thresh * std
        )[0]
        label_idxs[lbl] = np.delete(label_idxs[lbl], outliers)
    return label_idxs


def distance_distribution_metric_evaluation(
    eval_labels, ref_labels, eval_embeddings, ref_embeddings, k_neighbors=100
):
    eval_label_idxs = {}
    ref_label_idxs = {}

    assert (
        np.unique(eval_labels) == np.unique(ref_labels)
    ).all()  # checks if the labels are the same
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
        eval_label_cluster_centroids[lbl] = np.mean(
            eval_embeddings[eval_label_idxs[lbl]], axis=0
        )
        ref_label_cluster_centroids[lbl] = np.mean(
            ref_embeddings[ref_label_idxs[lbl]], axis=0
        )

    eval_label_cluster_centroids = np.array(
        [eval_label_cluster_centroids[lbl] for lbl in np.unique(unique_labels)]
    )
    ref_label_cluster_centroids = np.array(
        [ref_label_cluster_centroids[lbl] for lbl in np.unique(unique_labels)]
    )

    eval_label_cluster_centroids = (
        eval_label_cluster_centroids
        / np.linalg.norm(eval_label_cluster_centroids, axis=1)[:, None]
    )
    ref_label_cluster_centroids = (
        ref_label_cluster_centroids
        / np.linalg.norm(ref_label_cluster_centroids, axis=1)[:, None]
    )

    ref_dist_from_ref_centroid = ref_embeddings @ ref_label_cluster_centroids.T
    eval_dist_from_eval_centroid = eval_embeddings @ eval_label_cluster_centroids.T

    ref_dist_from_ref_centroid = softmax(ref_dist_from_ref_centroid).astype(np.float16)
    eval_dist_from_eval_centroid = softmax(eval_dist_from_eval_centroid).astype(
        np.float16
    )

    predictions = []
    batch_size = 1024  # Define the batch size

    num_batches = len(eval_embeddings) // batch_size + (
        len(eval_embeddings) % batch_size != 0
    )

    pb = tqdm(total=num_batches)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(eval_embeddings))

        batch_eval_dist_from_eval_centroid = eval_dist_from_eval_centroid[
            start_idx:end_idx
        ][:, None]
        entropy_dists = entropy(
            batch_eval_dist_from_eval_centroid,
            ref_dist_from_ref_centroid[None],
            axis=-1,
        )

        sorted_idxs = np.argsort(entropy_dists, axis=-1)
        k_nearest_idxs = sorted_idxs[:, :k_neighbors]

        k_nearest_nbs = ref_labels[k_nearest_idxs]

        for sub_batch in k_nearest_nbs:
            freq_dict = {lbl: 0 for lbl in np.unique(unique_labels)}
            for lbl in sub_batch:
                freq_dict[lbl] += 1

            predictions.append(max(freq_dict, key=freq_dict.get))

        pb.update(1)

    accuracy = (predictions == eval_labels).sum() * 100.0 / len(eval_embeddings)

    print(accuracy)


def extract_embeddings_from_model(
    dataloader,
    model,
    normalize_embeddings=True,
    get_images=False,
    get_labels=False,
    messup_tmrm=False,
    visualise_model_layer=True,
    get_fpaths=False,
):
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

        model.backbone.fc[0].register_forward_hook(get_activation("prefinal"))

    for batch in dataloader:
        if isinstance(batch, list):
            im, lbl, im_path = batch
        else:
            im, lbl, im_path = batch["images"], batch["classes"], batch["image_paths"]

        if messup_tmrm:
            tmrm_idx = 0
            im[:, tmrm_idx] = (im[:, tmrm_idx] - im[:, tmrm_idx].min()) / (
                im[:, tmrm_idx].max() - im[:, tmrm_idx].min()
            )

        with torch.no_grad():
            # with torch.autocast(device_type="cuda"):
            # with torch.amp.autocast(device_type='cuda'):
            # im = 2 * im - 1  # zero mean normalization
            features, _ = model(im.to("cuda"))

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


def cosine_distance(eval_embeddings, train_embeddings, weighted=False, temperature=1.0):
    dist_matrix = eval_embeddings @ train_embeddings.T
    if weighted:
        dist_matrix = dist_matrix / temperature
        dist_matrix = np.exp(dist_matrix)

    dist_matrix_idxs = (-1 * dist_matrix).argsort(
        1
    )  # because we want to sort in descending order of distances
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    return dist_matrix_sorted, dist_matrix_idxs


def l2_distance(eval_embeddings, train_embeddings):
    dist_matrix = np.linalg.norm(
        eval_embeddings[:, None] - train_embeddings[None, :], axis=-1
    )
    dist_matrix = dist_matrix.argsort(1)
    return dist_matrix


def balance_label_counts(embeddings, labels, dates=None, seed=1123, shuffle=True):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but labels has length {labels.shape[0]}"
        )

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

    if dates is not None:
        dates = np.asarray(dates)
        return embeddings[indices], labels[indices], dates[indices]
    return embeddings[indices], labels[indices]


def filter_by_label(
    pick_labels, embeddings, labels, drug_labels_dict, label_drug_dict, dates=None
):
    drug_labels_dict = {
        drug: label for drug, label in drug_labels_dict.items() if label in pick_labels
    }
    label_drug_dict = {
        label: drug for label, drug in label_drug_dict.items() if label in pick_labels
    }

    # mask the embeddings and labels
    mask = np.isin(labels, pick_labels)
    embeddings = embeddings[mask]
    labels = labels[mask]
    if dates is not None:
        dates = np.asarray(dates)[mask]
        return embeddings, labels, drug_labels_dict, label_drug_dict, dates
    return embeddings, labels, drug_labels_dict, label_drug_dict


def split_dataset(embeddings, labels, split_perc=0.9, balanced=True, seed=1123):
    if balanced:
        # Split each label separately to maintain class distribution
        unique_labels = np.unique(labels)
        train_indices = []
        val_indices = []
        for lbl in unique_labels:
            lbl_indices = np.where(labels == lbl)[0]
            train_idx, val_idx = train_test_split(
                lbl_indices, train_size=split_perc, random_state=seed, shuffle=True
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
    else:
        # Split the entire dataset at once
        all_indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(
            all_indices, train_size=split_perc, random_state=seed, shuffle=True
        )

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    return (
        embeddings[train_indices],
        labels[train_indices],
        embeddings[val_indices],
        labels[val_indices],
    )


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm  # 1. Import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# Confusion-matrix font sizes: matplotlib defaults bumped uniformly by +4pt.
CM_FONT_RC = {
    "font.size": 14,  # default 10
    "axes.titlesize": 16,  # default 'large' (12)
    "axes.labelsize": 14,  # default 'medium' (10)
    "xtick.labelsize": 14,  # default 'medium' (10)
    "ytick.labelsize": 14,  # default 'medium' (10)
    "figure.titlesize": 16,  # default 'large' (12)
    "legend.fontsize": 14,  # default 'medium' (10)
}


def plot_cm(
    gt_labels,
    pred_labels,
    label_drug_dict,
    verbose=True,
    make_plot=True,
    show_values=True,
    title="Confusion matrix",
    cmap=None,
    normalize=True,
    vmin=0.0,
    vmax=1.0,
    save_path=None,
):
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
            cmap = plt.get_cmap("Blues")

        with plt.rc_context(CM_FONT_RC):
            fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

            label_names = [label_drug_dict[idx] for idx in labels_idx]
            tickmarks = np.arange(len(label_names))
            ax.set_xticks(tickmarks)
            ax.set_xticklabels(label_names, rotation=90)
            ax.set_yticks(tickmarks)
            ax.set_yticklabels(label_names)

            # Create a copy for plot colors so we don't overwrite the returned cm
            if normalize:
                # Add a small epsilon or use np.errstate to avoid division by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    cm_plot = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                    cm_plot = np.nan_to_num(cm_plot)
            else:
                cm_plot = cm.copy()

            if normalize:
                im = ax.imshow(
                    cm_plot, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0
                )
            else:
                im = ax.imshow(cm_plot, cmap=cmap, interpolation="nearest")

            ax.set_title(title)

            # Make colorbar the same height as the matrix grid.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            if normalize:
                fig.colorbar(im, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            else:
                fig.colorbar(im, cax=cax)

            # 4. Optional: Add text values to the matrix squares
            if show_values:
                # For log scale, you might want to adjust the text color threshold.
                # We use a lower threshold here to account for the logarithmic visual shift.
                thresh = cm_plot.max() / 10.0 if cm_plot.max() > 0 else 0.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j,
                            i,
                            cm[i, j],
                            horizontalalignment="center",
                            color="white" if cm_plot[i, j] > thresh else "black",
                        )

            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
            if save_path is not None:
                fig.savefig(save_path, bbox_inches="tight")
            else:
                plt.show()

    return cm


def run_fold(
    test_date,
    embeddings,
    labels,
    dates,
    drug_labels_dict,
    label_drug_dict,
    top_ns,
    cmat_title,
    args,
    cfg,
    crossval_dir,
):
    """Run KNN evaluation with `test_date` held out. Saves confusion-matrix PNGs into
    crossval_dir, returns (per_drug_rows, fold_summary_rows, fold_top1_data). Per-drug
    rows are only populated for the cosine path; the L2 path produces aggregate-only
    summary rows. `fold_top1_data` is (gt_labels, top1_preds) for cosine, else None —
    used by the caller to build sum-/micro-mean / macro-mean across-fold CMs."""
    test_mask = dates == test_date
    train_mask = ~test_mask
    if test_mask.sum() == 0:
        raise ValueError(f"No samples for test_date={test_date!r}.")

    train_embeddings = embeddings[train_mask]
    train_labels = labels[train_mask]
    eval_embeddings = embeddings[test_mask]
    eval_labels = labels[test_mask]

    print(f"Test:  {test_mask.sum()} samples  (date={test_date})")
    print(
        f"Train: {train_mask.sum()} samples  (dates={sorted(np.unique(dates[train_mask]).tolist())})"
    )

    if train_embeddings.ndim > 2:
        train_embeddings = train_embeddings[:, -1, :]
    if eval_embeddings.ndim > 2:
        eval_embeddings = eval_embeddings[:, -1, :]

    per_drug_rows = []
    fold_summary_rows = []
    fold_top1_data = None

    if args.dist_metric == "cosine":
        print("Evaluating cosine distance...")
        dist_matrix, dist_matrix_idxs = cosine_distance(
            eval_embeddings,
            train_embeddings,
            weighted=False,
            temperature=cfg["training"]["loss"]["temperature"],
        )
        preds, correct_preds_idxs, _ = nearest_neighbor_evaluation(
            eval_labels,
            train_labels,
            top_ns,
            dist_matrix,
            dist_matrix_idxs=dist_matrix_idxs,
            verbose=False,
        )

        fold_top1_data = (np.asarray(eval_labels), np.asarray(preds[1]))

        # Save top-1 confusion matrices (with and without numeric cell values).
        cm_path = osp.join(crossval_dir, f"cm_top1-test_date={test_date}.png")
        plot_cm(
            eval_labels,
            preds[1],
            label_drug_dict,
            verbose=False,
            show_values=False,
            title=cmat_title,
            save_path=cm_path,
        )
        plt.close("all")

        cm_path_vals = osp.join(
            crossval_dir, f"cm_top1_with_values-test_date={test_date}.png"
        )
        plot_cm(
            eval_labels,
            preds[1],
            label_drug_dict,
            verbose=False,
            show_values=True,
            title=cmat_title,
            save_path=cm_path_vals,
        )
        plt.close("all")

        for top_n in top_ns:
            n_correct = len(correct_preds_idxs[top_n])
            n_total = len(eval_labels)
            fold_summary_rows.append(
                {
                    "test_date": test_date,
                    "dist_metric": "cosine",
                    "top_n": top_n,
                    "n_correct": int(n_correct),
                    "n_total": int(n_total),
                    "accuracy": n_correct * 100.0 / n_total,
                }
            )
            for lbl in np.unique(train_labels):
                total = int(np.sum(eval_labels == lbl))
                correct = int(np.sum(eval_labels[correct_preds_idxs[top_n]] == lbl))
                drug_name = label_drug_dict[int(lbl)]
                acc = correct * 100.0 / total if total > 0 else 0.0
                per_drug_rows.append(
                    {
                        "test_date": test_date,
                        "top_n": top_n,
                        "drug": drug_name,
                        "label": int(lbl),
                        "n_correct": correct,
                        "n_total": total,
                        "accuracy": acc,
                    }
                )
            print(f"  Top-{top_n} accuracy: {n_correct * 100.0 / n_total:.2f}%")

    if args.dist_metric == "l2":
        print("Building KDTree...")
        tree = KDTree(train_embeddings)
        for k in [100]:
            dist, nearest_ind = tree.query(eval_embeddings, k=k)
            for top_n in top_ns:
                correct = 0
                for i in tqdm(range(len(eval_embeddings))):
                    eval_lbl = eval_labels[i]
                    k_nearest_nbs = train_labels[nearest_ind[i]]
                    top_most_freq_lbls = topKfrequent(k_nearest_nbs, top_n)
                    correct += 1 if eval_lbl in top_most_freq_lbls else 0
                acc = correct * 100.0 / len(eval_labels)
                fold_summary_rows.append(
                    {
                        "test_date": test_date,
                        "dist_metric": "l2",
                        "top_n": top_n,
                        "n_correct": int(correct),
                        "n_total": int(len(eval_labels)),
                        "accuracy": acc,
                    }
                )
                print(f"  L2 Top-{top_n} (k={k}): {acc:.2f}%")

    return per_drug_rows, fold_summary_rows, fold_top1_data


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/earkfeld/Projects/MitoSpace4D"

    already_have_embeddings = True
    balance_classes = True
    balanced_split = True

    split_perc = 0.8
    top_ns = [1, 3]

    cmat_title = "Top1 KNN Confusion Matrix - MitoTracker DeepRed"
    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_reproducibility_252eps"

    # cmat_title = "Top1 KNN Confusion Matrix - MitoBright"
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_reproducibility_mitobright_252eps'

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", "r") as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    if already_have_embeddings:
        print("Loading pre-extracted embeddings...")
        embeddings = np.load(f"{embeddings_dir}/embeddings.npy")
        labels = np.load(f"{embeddings_dir}/labels.npy")

        with open(f"{embeddings_dir}/image_paths.csv", "r") as f:
            image_paths = np.array([line.strip() for line in f if line.strip()])
        assert len(image_paths) == len(
            labels
        ), f"image_paths.csv has {len(image_paths)} rows but labels.npy has {len(labels)}"
        # Date = second-to-last path segment with the "-N" replicate suffix stripped
        dates = np.array([p.split("/")[-2].split("-")[0] for p in image_paths])
        print(f"Available dates: {sorted(np.unique(dates).tolist())}")

        unique_labels_in_dataset = set(labels)
        drug_labels_dict = {
            drug: label
            for drug, label in drug_labels_dict.items()
            if label in unique_labels_in_dataset
        }
        label_drug_dict = {
            label: drug
            for label, drug in label_drug_dict.items()
            if label in unique_labels_in_dataset
        }

        if args.labels:
            embeddings, labels, drug_labels_dict, label_drug_dict, dates = (
                filter_by_label(
                    args.labels,
                    embeddings,
                    labels,
                    drug_labels_dict,
                    label_drug_dict,
                    dates=dates,
                )
            )

        if balance_classes:
            embeddings, labels, dates = balance_label_counts(
                embeddings, labels, dates=dates
            )
            print(
                f"Balanced to {len(np.unique(labels))} classes with {np.bincount(labels)} samples each"
            )
            print({k: (labels == k).sum() for k in np.unique(labels)})

        # === K-fold cross-validation across days ===
        # Default: leave-one-day-out across every unique date in the dataset.
        # Override: pass --test_date YYYYMMDD to run a single fold.
        unique_dates = sorted(np.unique(dates).tolist())
        if args.test_date and args.test_date in unique_dates:
            fold_dates = [args.test_date]
            print(f"Running single fold for --test_date={args.test_date}")
        else:
            fold_dates = unique_dates
            print(
                f"Running {len(fold_dates)}-fold leave-one-day-out cross-validation across: {fold_dates}"
            )

        crossval_dir = osp.join(embeddings_dir, "crossval_perday")
        os.makedirs(crossval_dir, exist_ok=True)

        all_per_drug_rows = []
        all_fold_summary_rows = []
        all_top1_gt = []
        all_top1_preds = []

        for test_date in fold_dates:
            print(f"\n========== Fold: test_date={test_date} ==========")
            per_drug_rows, fold_summary_rows, fold_top1_data = run_fold(
                test_date=test_date,
                embeddings=embeddings,
                labels=labels,
                dates=dates,
                drug_labels_dict=drug_labels_dict,
                label_drug_dict=label_drug_dict,
                top_ns=top_ns,
                cmat_title=cmat_title,
                args=args,
                cfg=cfg,
                crossval_dir=crossval_dir,
            )
            all_per_drug_rows.extend(per_drug_rows)
            all_fold_summary_rows.extend(fold_summary_rows)
            if fold_top1_data is not None:
                all_top1_gt.append(fold_top1_data[0])
                all_top1_preds.append(fold_top1_data[1])

        # === Aggregate and write CSVs ===
        if all_per_drug_rows:
            per_drug_df = pd.DataFrame(all_per_drug_rows)
            per_drug_csv = osp.join(crossval_dir, "per_drug_accuracy_per_fold.csv")
            per_drug_df.to_csv(per_drug_csv, index=False)
            print(f"\nSaved per-drug per-fold accuracies → {per_drug_csv}")

            # Mean per drug across folds (per top_n).
            mean_per_drug = (
                per_drug_df.groupby(["top_n", "drug", "label"])
                .agg(
                    n_correct_sum=("n_correct", "sum"),
                    n_total_sum=("n_total", "sum"),
                    accuracy_mean=("accuracy", "mean"),
                    accuracy_std=("accuracy", "std"),
                    n_folds=("test_date", "nunique"),
                )
                .reset_index()
            )
            mean_per_drug["accuracy_micro"] = (
                mean_per_drug["n_correct_sum"]
                / mean_per_drug["n_total_sum"].replace(0, np.nan)
                * 100.0
            )
            mean_csv = osp.join(crossval_dir, "per_drug_accuracy_mean_across_folds.csv")
            mean_per_drug.to_csv(mean_csv, index=False)
            print(f"Saved mean-across-folds per-drug accuracies → {mean_csv}")
            print(mean_per_drug.head(20).to_string(index=False))

        if all_fold_summary_rows:
            fold_df = pd.DataFrame(all_fold_summary_rows)
            fold_df_csv = osp.join(crossval_dir, "fold_overall_accuracy.csv")
            fold_df.to_csv(fold_df_csv, index=False)
            print(f"\nSaved fold-level accuracy → {fold_df_csv}")
            print("\nFold-level overall accuracy:")
            print(fold_df.to_string(index=False))

            mean_across = (
                fold_df.groupby(["dist_metric", "top_n"])["accuracy"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            mean_across_csv = osp.join(crossval_dir, "mean_across_folds.csv")
            mean_across.to_csv(mean_across_csv, index=False)
            print("\nMean across folds:")
            print(mean_across.to_string(index=False))

        # === Aggregated top-1 confusion matrices across folds ===
        # Three variants are saved:
        #   - sum_across_folds:        raw cell counts summed across folds.
        #   - mean_across_folds (micro): concat all (gt, pred), then row-normalize
        #     once. Equivalent to weighting each fold by its sample count.
        #   - macro_mean_across_folds:  per-fold row-normalize, then nan-average
        #     across folds. Weights every fold equally regardless of size; rows
        #     missing in a fold are excluded via np.nanmean.
        if all_top1_gt:
            gt_concat = np.concatenate(all_top1_gt)
            preds_concat = np.concatenate(all_top1_preds)
            n_folds = len(all_top1_gt)

            sum_cm_path = osp.join(crossval_dir, "cm_top1-sum_across_folds.png")
            plot_cm(
                gt_concat,
                preds_concat,
                label_drug_dict,
                verbose=False,
                show_values=True,
                normalize=False,
                title=cmat_title,
                save_path=sum_cm_path,
            )
            plt.close("all")

            mean_cm_path = osp.join(
                crossval_dir, "cm_top1-mean_across_folds_normalized.png"
            )
            plot_cm(
                gt_concat,
                preds_concat,
                label_drug_dict,
                verbose=False,
                show_values=False,
                normalize=True,
                title=cmat_title,
                save_path=mean_cm_path,
            )
            plt.close("all")

            mean_cm_vals_path = osp.join(
                crossval_dir, "cm_top1-mean_across_folds_normalized_with_values.png"
            )
            plot_cm(
                gt_concat,
                preds_concat,
                label_drug_dict,
                verbose=False,
                show_values=True,
                normalize=True,
                title=cmat_title,
                save_path=mean_cm_vals_path,
            )
            plt.close("all")

            # Macro mean: row-normalize each fold's CM, then average across folds.
            labels_idx = sorted(label_drug_dict.keys())
            n_classes = len(labels_idx)
            per_fold_norm = np.full(
                (n_folds, n_classes, n_classes), np.nan, dtype=float
            )
            for i, (gt_f, pr_f) in enumerate(zip(all_top1_gt, all_top1_preds)):
                cm_f = confusion_matrix(gt_f, pr_f, labels=labels_idx).astype(float)
                row_sums = cm_f.sum(axis=1, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cm_norm_f = np.where(row_sums > 0, cm_f / row_sums, np.nan)
                per_fold_norm[i] = cm_norm_f

            macro_mean_cm = np.nan_to_num(np.nanmean(per_fold_norm, axis=0))
            label_names = [label_drug_dict[idx] for idx in labels_idx]
            tickmarks = np.arange(len(label_names))

            macro_paths = []
            for show_values, suffix in [(False, ""), (True, "_with_values")]:
                with plt.rc_context(CM_FONT_RC):
                    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
                    ax.set_xticks(tickmarks)
                    ax.set_xticklabels(label_names, rotation=90)
                    ax.set_yticks(tickmarks)
                    ax.set_yticklabels(label_names)
                    im = ax.imshow(
                        macro_mean_cm,
                        cmap=plt.get_cmap("Blues"),
                        interpolation="nearest",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    ax.set_title(cmat_title)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    fig.colorbar(im, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

                    if show_values:
                        thresh = (
                            macro_mean_cm.max() / 2.0
                            if macro_mean_cm.max() > 0
                            else 0.0
                        )
                        for i in range(n_classes):
                            for j in range(n_classes):
                                ax.text(
                                    j,
                                    i,
                                    f"{macro_mean_cm[i, j]:.2f}",
                                    horizontalalignment="center",
                                    color=(
                                        "white"
                                        if macro_mean_cm[i, j] > thresh
                                        else "black"
                                    ),
                                )
                    ax.set_ylabel("True label")
                    ax.set_xlabel("Predicted label")
                    macro_path = osp.join(
                        crossval_dir, f"cm_top1-macro_mean_across_folds{suffix}.png"
                    )
                    fig.savefig(macro_path, bbox_inches="tight")
                    plt.close("all")
                    macro_paths.append(macro_path)

            np.save(
                osp.join(crossval_dir, "cm_top1-per_fold_normalized.npy"), per_fold_norm
            )
            np.save(
                osp.join(crossval_dir, "cm_top1-macro_mean_across_folds.npy"),
                macro_mean_cm,
            )

            print(f"\nSaved aggregated confusion matrices ({n_folds} folds) →")
            print(f"  {sum_cm_path}")
            print(f"  {mean_cm_path}")
            print(f"  {mean_cm_vals_path}")
            for p in macro_paths:
                print(f"  {p}")

        # === UMAP / KL-divergence quality check on the full set (not per-fold) ===
        embeddings_2d = embeddings[:, -1, :] if embeddings.ndim > 2 else embeddings
        flat_embeddings = embeddings_2d.reshape(embeddings_2d.shape[0], -1)

        print("\n=== UMAP reduction on full dataset ===")
        reducer = UMAP(
            verbose=True, n_components=3, n_neighbors=25, min_dist=0.1, metric="cosine"
        )
        embeddings_umap = reducer.fit_transform(flat_embeddings)
        embeddings_umap = (
            embeddings_umap / np.linalg.norm(embeddings_umap, axis=1)[:, None]
        )

        original_distances = pairwise_distances(flat_embeddings)
        reduced_distances = pairwise_distances(embeddings_umap)
        original_distances_normalized = (
            original_distances - np.min(original_distances)
        ) / (np.max(original_distances) - np.min(original_distances))
        reduced_distances_normalized = (
            reduced_distances - np.min(reduced_distances)
        ) / (np.max(reduced_distances) - np.min(reduced_distances))
        kl_divergence = entropy(
            original_distances_normalized.flatten(),
            reduced_distances_normalized.flatten(),
        )
        print("KL Divergence:", kl_divergence)

    else:
        # Single-fold model-extraction path (unchanged) — runs when embeddings aren't cached.
        print("Generating embeddings...")
        model = Lightweight3DResNet(embedding_size=2048, cfg=cfg, apply_aug=False)
        checkpoint_path = None
        dataset_name = cfg["evaluate"]["dataset"]
        model = SimCLRRunner.load_from_checkpoint(checkpoint_path, model=model, cfg=cfg)
        model.eval()

        loaders_reference = get_mitospace_data_loaders(
            f"{proj_dir}/data/2024_subdata/",
            shuffle=False,
            batch_size=2,
            to_load=["train"],
            timesteps=cfg["data_params"]["timesteps"],
            zstacks=cfg["data_params"]["zstacks"],
            samples_per_drug=cfg["data_params"]["samples_per_drug"],
            pick_labels=None,
        )
        loaders_eval = get_mitospace_data_loaders(
            f"{proj_dir}/data/2024_subdata/",
            shuffle=False,
            batch_size=2,
            to_load=["val"],
            timesteps=cfg["data_params"]["timesteps"],
            zstacks=cfg["data_params"]["zstacks"],
            samples_per_drug=cfg["data_params"]["samples_per_drug"],
            pick_labels=None,
        )
        train_loader, eval_loader = loaders_reference["train"], loaders_eval["val"]

        train_embeddings, _, train_labels, _ = extract_embeddings_from_model(
            train_loader,
            model.model,
            normalize_embeddings=True,
            get_images=False,
            get_labels=True,
            messup_tmrm=False,
            visualise_model_layer=False,
            get_fpaths=True,
        )
        eval_embeddings, _, eval_labels, _ = extract_embeddings_from_model(
            eval_loader,
            model.model,
            normalize_embeddings=True,
            get_images=False,
            get_labels=True,
            messup_tmrm=False,
            visualise_model_layer=False,
            get_fpaths=True,
        )

        if args.dist_metric == "cosine":
            dist_matrix, dist_matrix_idxs = cosine_distance(
                eval_embeddings,
                train_embeddings,
                weighted=False,
                temperature=cfg["training"]["loss"]["temperature"],
            )
            preds, _, _ = nearest_neighbor_evaluation(
                eval_labels,
                train_labels,
                top_ns,
                dist_matrix,
                dist_matrix_idxs=dist_matrix_idxs,
            )
            plot_cm(
                eval_labels,
                preds[1],
                label_drug_dict,
                verbose=False,
                show_values=False,
                title=cmat_title,
                save_path=osp.join(embeddings_dir, "cm_top1.png"),
            )
            plt.close("all")
