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
from utils.utils import *
from data_aug.dataset_utils import get_mitospace_data_loaders
import torch
from train_simclr import SimCLRRunner

from utils.vis import plot_cm
from torch.utils.data import DataLoader

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.stats import entropy

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='MitoSpace Evaluation')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--config', default='/tscc/nfs/home/d5agarwal/projects/MitoSpace4D/simclr/config.yaml',
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

                else:
                    preds[top_n].append(top_most_freq_lbls[0])
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

    return preds


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
                                  messup_tmrm=False, visualise_model_layer=True):
    embeddings = []

    images = [] if get_images else None
    labels = [] if get_labels else None

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
            im, lbl = batch
        else:
            im, lbl = batch["images"], batch["classes"]

        if messup_tmrm:
            tmrm_idx = 0
            im[:, tmrm_idx] = (im[:, tmrm_idx] - im[:, tmrm_idx].min()) / (
                    im[:, tmrm_idx].max() - im[:, tmrm_idx].min())

        with torch.no_grad():
            features, _ = model(im.to('cuda'))

        if normalize_embeddings:
            features = F.normalize(features, dim=-1)

        embeddings.append(features.detach().cpu().numpy())

        if get_images:
            images.append(im.detach().cpu().numpy())
        if get_labels:
            labels.append(lbl.detach().cpu().numpy())

        pbar.update(1)

    embeddings = np.concatenate(embeddings)
    # embeddings = np.concatenate(prefinal_activation)

    if get_images:
        images = np.concatenate(images)
    if get_labels:
        labels = np.concatenate(labels)

    return embeddings, images, labels


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


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/"

    model = MitoSpace4D(
        in_channels=cfg['model_params']['in_channels'],
        out_dim=cfg['model_params']['out_dim']).to(device)

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=70-step=8875-val_loss=0.00.ckpt"
    top_ns = cfg["evaluate"]["top_ns"]
    dataset_name = cfg["evaluate"]["dataset"]

    print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )

    loaders_reference = get_mitospace_data_loaders(
        f'{proj_dir}/data/2023_data/',
        shuffle=False, batch_size=8, to_load=["train"],
        timesteps=cfg['data_params']['timesteps'],
        zstacks=cfg['data_params']['zstacks'],
        pick_labels=None)

    loaders_eval = get_mitospace_data_loaders(
        f'{proj_dir}/data/2023_data/',
        shuffle=False, batch_size=8, to_load=["train"],
        timesteps=cfg['data_params']['timesteps'],
        zstacks=cfg['data_params']['zstacks'],
        pick_labels=None
    )

    train_loader, eval_loader = (loaders_reference["train"], loaders_eval["train"])

    train_embeddings, train_images, train_labels = extract_embeddings_from_model(train_loader, model.model,
                                                                                 normalize_embeddings=True,
                                                                                 get_images=False,
                                                                                 get_labels=True,
                                                                                 messup_tmrm=False,
                                                                                 visualise_model_layer=False)

    eval_embeddings, eval_images, eval_labels = extract_embeddings_from_model(eval_loader, model.model,
                                                                              normalize_embeddings=True,
                                                                              get_images=False, get_labels=True,
                                                                              messup_tmrm=False,
                                                                              visualise_model_layer=False)

    # Evaluation on cosine similarity
    if args.dist_metric == 'cosine':
        dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings, train_embeddings, weighted=True,
                                                        temperature=cfg["training"]["loss"]["temperature"])
        preds = nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs)

        # plot confusion matrix
        cm = plot_cm(eval_labels, preds[1], label_drug_dict, verbose=False)  # top 1 confusion matrix

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
        dist_matrix = cosine_distance(eval_embeddings_reduced, train_embeddings_reduced)
        preds = nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, num_neighbors=[1000])

        # plot confusion matrix
        cm = plot_cm(eval_labels, preds[1], label_drug_dict)

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
