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
from simclr.models import *
import argparse

from simclr.models_simple import Lightweight3DResNet
from utils.utils import *
from utils.wasserstein_distance import *
from data_aug.dataset_utils import get_mitospace_data_loaders
import torch
from train_simclr import SimCLRRunner

from utils.vis import plot_cm
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
            #im = 2 * im - 1  # zero mean normalization
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

def balance_label_counts(embeddings, labels, n_samples=None, seed=1123, shuffle=True):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"embeddings has {embeddings.shape[0]} rows but labels has length {labels.shape[0]}")

    rng = np.random.default_rng(seed)

    unique_labels = np.unique(labels)
    label_counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    min_count = min(label_counts.values())

    if n_samples is not None:
        min_count = min(min_count, n_samples)
        print(f"Min count is {min_count}")

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
        train_indices, val_indices = train_test_split(all_indices, train_size=split_perc, random_state=seed, shuffle=True)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    return embeddings[train_indices], labels[train_indices], embeddings[val_indices], labels[val_indices]

def filter_by_days(ds_days, embeddings, labels, img_paths):
    """ Filters data by day """
    ds_days = set(ds_days)

    filtered_embeddings = []
    filtered_labels = []
    filtered_img_paths = []

    for img_path in img_paths:
        # Check if any of the ds_days is in the img_path
        if any(day in img_path for day in ds_days):
            idx = img_paths.index(img_path)
            filtered_embeddings.append(embeddings[idx])
            filtered_labels.append(labels[idx])
            filtered_img_paths.append(img_path)

    return np.array(filtered_embeddings), np.array(filtered_labels), filtered_img_paths

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/"
    
    already_have_embeddings = True
    balance_classes = True
    balanced_split = True

    n_samples = None

    pca_dim = 100
    n_proj = 512
    n_perm = 2000

    split_perc = 0.9
    top_ns = cfg["evaluate"]["top_ns"]

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220"
    embeddings_dir = "/runs/exp0_modified_embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classifier/cellpaint_features/hela_mito"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classifier/deepprofiler_features/PTEN_deepprofiler_pooled-clones"

    # DP sensitivity: 0.8315
    # CP sensitivity:

    log_transform = False

    A_label = 30
    A_prime_label = 31
    B_label = 31
    B_prime_label = 34

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    print("Loading pre-extracted embeddings...")
    embeddings = np.load(
        f'{embeddings_dir}/embeddings_raw.npy')
    labels = np.load(
        f'{embeddings_dir}/labels.npy')
    # img_paths = np.loadtxt(f'{embeddings_dir}/image_paths.csv', dtype=str).tolist()

    # Filter the drug label dicts to only include the labels present in the dataset
    unique_labels_in_dataset = set(labels)
    drug_labels_dict = {drug: label for drug, label in drug_labels_dict.items() if label in unique_labels_in_dataset}
    label_drug_dict = {label: drug for label, drug in label_drug_dict.items() if label in unique_labels_in_dataset}

    if args.labels:
        embeddings, labels, drug_labels_dict, label_drug_dict = filter_by_label(args.labels, embeddings, labels, drug_labels_dict, label_drug_dict)

    if balance_classes:
        embeddings, labels = balance_label_counts(embeddings, labels, n_samples=n_samples)
        # balance_label_counts(embeddings, labels)
        print(f"Balanced classes.")

    # Normalize embeddings to unit length using L2 norm numpy
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Rescale to [0, 1] for Wasserstein ground distance stability
    # Using global min/max to preserve the geometric relationship between all points
    # emb_min = embeddings.min()
    # emb_max = embeddings.max()
    # embeddings = (embeddings - emb_min) / (emb_max - emb_min)

    print(f"Min embedding magnitude: {np.min(embeddings):.6f}")
    print(f"Max embedding magnitude: {np.max(embeddings):.6f}")

    # Separate the embeddings according to labels
    A_embeddings = embeddings[labels == A_label]
    # A_imgs = img_paths[labels == A_label]
    A_prime_embeddings = embeddings[labels == A_prime_label]
    # A_prime_imgs = img_paths[labels == A_prime_label]
    B_embeddings = embeddings[labels == B_label]
    # B_imgs = img_paths[labels == B_label]
    B_prime_embeddings = embeddings[labels == B_prime_label]
    # B_prime_imgs = img_paths[labels == B_prime_label]

    w2, beta_w2 = exact_wasserstein_sensitivity(A_embeddings, A_prime_embeddings, log_transform=log_transform)
    print(f"W2 sensitivity: {beta_w2:.4f}")

    # # Calculate the wasserstein distance between the embeddings of A and A'
    # S, p = sw2(A_embeddings,
    #            A_prime_embeddings,
    #            pca_dim=pca_dim,
    #            n_proj=n_proj,
    #            n_perm=n_perm,
    #            seed=1123)
    #
    # print(f"A: {label_drug_dict[A_label]} ({A_label})")
    # print(f"A': {label_drug_dict[A_prime_label]} ({A_prime_label})")
    # print(f"SW2: {S:.6f}")
    # print(f"p-value: {p:.6f}")

    # # Calculate the differential for the perturbation using wasserstein metric
    # SA, SB, T, p = differential_sw2(A_embeddings,
    #                                 A_prime_embeddings,
    #                                 B_embeddings,
    #                                 B_prime_embeddings,
    #                                 pca_dim=pca_dim,
    #                                 n_proj=n_proj,
    #                                 n_perm=n_perm,
    #                                 seed=1123)
    #
    # print(f"=== Params ===")
    # print(f"Distribution Labels:\n"
    #       f"A: {label_drug_dict[A_label]} ({A_label}, n={len(A_embeddings)})\n"
    #       f"A': {label_drug_dict[A_prime_label]} ({A_prime_label}), n={len(A_prime_embeddings)})\n"
    #       f"B: {label_drug_dict[B_label]} ({B_label}, n={len(B_embeddings)})\n"
    #       f"B': {label_drug_dict[B_prime_label]} ({B_prime_label}, n={len(B_prime_embeddings)})")
    # print(f"n_proj: {n_proj}")
    # print(f"n_perm: {n_perm}")
    # print(f"PCA dim: {pca_dim}")
    # print(f"Seed: {1123}\n")
    #
    # print(f"=== Results ===")
    # print(f"SW2(A,A') = {SA:.6f}")
    # print(f"SW2(B,B') = {SB:.6f}")
    # print(f"T = SA - SB = {T:.6f}")
    # print(f"Permutation p-value (two-sided) = {p:.6g}")