import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import get_phenotypic_colors


def compute_intercluster_dist(embedding, labels, label_names):
    """
    Compute the inter-cluster distances for each label
    :param embedding: np.array of shape (n_samples, n_features)
    :param labels: np.array of shape (n_samples,)
    :param label_names: np.array of shape (n_samples,)
    :return: dict of label_name -> inter-cluster distance
    """
    # get the unique labels
    unique_labels = np.unique(labels)

    # find the mean vector for each label
    mean_vectors = {}
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        mean_vectors[label] = np.mean(embedding[idxs], axis=0)
        mean_vectors[label] /= np.linalg.norm(mean_vectors[label])

    # create a dictionary to store the inter-cluster distances
    intercluster_distances = np.zeros((len(unique_labels), len(unique_labels)))

    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            intercluster_distances[i, j] = np.dot(mean_vectors[label1], mean_vectors[label2])
            intercluster_distances[j, i] = intercluster_distances[i, j]

    for label1 in unique_labels:
        for label2 in unique_labels:
            label1_embeddings = embedding[labels == label1]
            label2_embeddings = embedding[labels == label2]

            # compute the pairwise cosine distances between the embeddings
            distances = np.dot(label1_embeddings, label2_embeddings.T).flatten()

            # remove the top and bottom 5% of the distances
            distances = np.sort(distances)
            bottom_5_percentile = np.percentile(distances, 5)
            top_5_percentile = np.percentile(distances, 95)
            distances = distances[(distances > bottom_5_percentile) & (distances < top_5_percentile)]

            intercluster_distances[label1, label2] = np.mean(distances)

    phenotypic_colors = get_phenotypic_colors(intercluster_distances, num_clusters=8)

    # normalize it per row
    for label1 in unique_labels:
        min_ = np.min(intercluster_distances[label1])
        max_ = np.max(intercluster_distances[label1])
        intercluster_distances[label1] = (intercluster_distances[label1] - min_) / (max_ - min_)

    # plot the inter-cluster distances as heatmap with smooth values and values in the cell
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(intercluster_distances, annot=True, ax=ax)
    ax.set_xticklabels(label_names[unique_labels], rotation=90)
    ax.set_yticklabels(label_names[unique_labels], rotation=0)

    plt.show()


if __name__ == '__main__':
    exp_dir = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal'
    embeddings_path = f'{exp_dir}/embeddings/embeddings_static_oligo.npy'
    labels_path = f'{exp_dir}/embeddings/labels_static_oligo.npy'
    label_names_path = f'{exp_dir}/embeddings/label_names.npy'

    embedding = np.load(embeddings_path)
    if len(embedding.shape) == 3:
        # take the last timestep embeddings
        embedding = embedding[:, -1]

    labels = np.load(labels_path)
    label_names = np.load(label_names_path)

    # compute the inter-cluster distances for each label
    inter_cluster_distances = compute_intercluster_dist(embedding, labels, label_names)
