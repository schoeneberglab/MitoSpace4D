
import pickle
import random

import numpy as np
import umap
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

def balance_label_counts(embeddings, labels, seed=1123):
    """
    Downsample each class to the same number of samples (the minimum class count),
    and shuffle the balanced set deterministically.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    # Count the number of occurrences of each label
    label_counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
    print(label_counts)

    min_count = min(label_counts.values())

    rng = np.random.default_rng(seed)
    selected_indices = []

    for lbl in unique_labels:
        lbl_indices = np.where(labels == lbl)[0]
        # Randomly select exactly min_count indices for this label
        chosen = rng.choice(lbl_indices, size=min_count, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    # Shuffle so downstream slicing doesn't group by label
    rng.shuffle(selected_indices)

    return embeddings[selected_indices], labels[selected_indices]

def split_dataset(embeddings, labels, split_perc=0.9, per_label=True, seed=1123, shuffle=True):
    if shuffle:
        # Shuffle the dataset before splitting
        np.random.seed(seed)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        embeddings = embeddings[indices]
        labels = labels[indices]
    if per_label:
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



if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/earkfeld/Projects/MitoSpace4D"
    
    already_have_embeddings = True
    balance_classes = False
    split_perc = 0.9
    split_per_label = True

    # pick_datasets = ["20250922-1", "20250922-2", "20250922-3", "20250925-1", "20250925-2", "20250925-3"]
    pick_datasets = None
    pick_labels = [30, 31]

    top_ns = cfg["evaluate"]["top_ns"]

    # embeddings_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20250929_10frames"
    # embeddings_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20250929_10frames_modified_labels_for_eval"
    embeddings_dir = "/home/earkfeld/Desktop/cell_paint_feats/cellpaint_embeddings"
    # embeddings_dir = "/home/earkfeld/Desktop/cell_paint_feats/cellpaint_embeddings_pos_pten_hela"
    # embeddings_dir = "/home/earkfeld/Desktop/cell_paint_feats/2_cellpaint_embeddings_pos_tomm20_hela"
    # embeddings_dir = "/home/earkfeld/Desktop/cell_paint_feats/3_cellpaint_embeddings_pos_a549_hela"

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    if pick_labels is not None:
        # Remove any labels not in pick_labels
        drug_labels_dict = {drug: label for drug, label in drug_labels_dict.items() if label in pick_labels}
        label_drug_dict = {label: drug for label, drug in label_drug_dict.items() if label in pick_labels}

    if already_have_embeddings:
        # embeddings = np.load(f'{embeddings_dir}/embeddings_raw.npy') # original
        embeddings = np.load(f'{embeddings_dir}/embeddings.npy')
        labels = np.load(f'{embeddings_dir}/labels.npy')
        print(f"Embeddings Shape: {embeddings.shape}")
        print(f"Labels Shape: {labels.shape}")

        from sklearn.preprocessing import normalize, StandardScaler

        # Option 1: embeddings or already balanced features
        # X_normalized = normalize(X, norm='l2', axis=1)

        # Option 2: heterogeneous features, standardize first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        # Normalize the along the feature axis (N, D) -> (N, D)
        X_normalized = normalize(X_scaled, norm='max', axis=1)
        embeddings = X_normalized
        print(np.max(embeddings), np.min(embeddings))

        if pick_datasets is not None:
            df_img_paths = pd.read_csv(f'{embeddings_dir}/image_paths.csv', header=None)
            img_paths = df_img_paths[0].tolist()
            
            keep_idxs = []
            for i, path in enumerate(img_paths):
                if any(ds in path for ds in pick_datasets):
                    keep_idxs.append(i)
            embeddings = embeddings[keep_idxs]
            labels = labels[keep_idxs]
        
        if balance_classes:
            embeddings, labels = balance_label_counts(embeddings, labels)
            # balance_label_counts(embeddings, labels)
            print(f"Balanced to {len(np.unique(labels))} classes with {np.bincount(labels)} samples each")

        # shuffle them with seed 1123
        # random.seed(1123)
        # data = list(zip(embeddings, labels))
        # random.shuffle(data)
        # embeddings, labels = zip(*data)
        # embeddings, labels = np.array(embeddings), np.array(labels)

        # len_all_data = round(len(labels) * 1.)
        # train_split = round(len_all_data * split_perc)
        # val_split = round(len_all_data * (1 - split_perc))

        # train_embeddings, eval_embeddings = embeddings[:train_split], embeddings[train_split: train_split + val_split]
        # train_labels, eval_labels = labels[:train_split], labels[train_split: train_split + val_split]

        train_embeddings, train_labels, eval_embeddings, eval_labels = split_dataset(embeddings, labels, split_perc=split_perc, per_label=True)

        # train_embeddings = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings.npy')
        # train_labels = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/labels.npy')
        #
        # eval_embeddings = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings_5/embeddings.npy')
        # eval_labels = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings_5/labels.npy')

        if len(train_embeddings.shape) > 2:
            train_embeddings = train_embeddings[:, -1]  # take only the final time step

        if len(eval_embeddings.shape) > 2:
            eval_embeddings = eval_embeddings[:, -1]  # take only the final time step

    else:
        model = Lightweight3DResNet(embedding_size=2048, 
                                    cfg_aug=cfg['data_params']['transforms'],
                                    apply_aug=False)

        # checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=21-step=14212-val_loss=0.00.ckpt"
        checkpoint_path = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitoSpace4D_resnetbilstm_encoded_normal_eps287.ckpt"
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

        eval_embeddings, eval_images, eval_labels, eval_im_paths = extract_embeddings_from_model(eval_loader, model.model,
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
        preds, correct_preds_idxs, incorrect_preds_idxs = nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs=dist_matrix_idxs)

        with open(f'{proj_dir}/correct_preds_idxs.pkl', 'wb') as f:
            pickle.dump(correct_preds_idxs, f)
        with open(f'{proj_dir}/incorrect_preds_idxs.pkl', 'wb') as f:
            pickle.dump(incorrect_preds_idxs, f)

        # plot confusion matrix
        cm = plot_cm(eval_labels, preds[1], label_drug_dict, verbose=False, vmin=0., vmax=1.)  # top 1 confusion matrix

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
    reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=15, min_dist=0.1, metric='l2')
    all_embeddings = np.concatenate([train_embeddings, eval_embeddings])
    all_embeddings_reduced = reducer.fit_transform(all_embeddings.reshape(all_embeddings.shape[0], -1))
    all_embeddings_reduced = all_embeddings_reduced / np.linalg.norm(all_embeddings_reduced, axis=1)[:, None]

    train_embeddings_reduced = all_embeddings_reduced[:len(train_embeddings)]
    eval_embeddings_reduced = all_embeddings_reduced[len(train_embeddings):]

    # Evaluation on cosine similarity
    if args.dist_metric == 'cosine':
        print("Evaluating umap projected embeddings using cosine distance")
        dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings_reduced, train_embeddings_reduced)
        preds = nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, num_neighbors=[100], dist_matrix_idxs=dist_matrix_idxs)

        # plot confusion matrix
        # cm = plot_cm(eval_labels, preds[1], label_drug_dict)

    # Evaluate on L2 Distance
    if args.dist_metric == 'l2':
        print("Building Tree")
        tree = KDTree(train_embeddings_reduced)
        print("Tree Built")

        num_neighbors = [100]

        accs = []
        for k in num_neighbors:
            dist, nearest_ind = tree.query(eval_embeddings_reduced, k=k)
            predicted_labels = []

            for top_n in top_ns:
                correct = 0
                for i in tqdm(range(len(eval_embeddings_reduced))):
                    eval_lbl = eval_labels[i]
                    k_nearest_nbs = train_labels[nearest_ind[i]]
                    top_most_freq_lbls = topKfrequent(k_nearest_nbs, top_n)
                    correct += 1 if eval_lbl in top_most_freq_lbls else 0

                print(f"------------------Top-{top_n} Evaluation for {k} Neighbors------------------------")
                print(f"Correct: {correct}; Total: {len(eval_embeddings_reduced)}")
                acc = correct * 100. / len(eval_labels)
                accs.append(acc)
                print("Accuracy(%): ", acc)

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
