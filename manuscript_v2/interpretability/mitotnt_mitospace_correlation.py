import numpy as np
import os
from interpretability.mitotnt_on_mitospace import get_feats_and_embs, get_identifiers
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mitotnt_features = pd.read_csv('./4d_mito_features_summer.csv')
    umap_emb = np.load("/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings.npy")
    img_file_paths = np.load("/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/image_file_paths.npy")

    identifier = get_identifiers(img_file_paths)

    correlation_matrix = []
    features = ['Node Count', 'Degree', 'Segment Length', 'Fragment Length', 'Fragment Diameter', 'Graph Density', 'Graph Efficiency', 'Clustering Coefficient', 'Node Diffusivity', 'Segment Diffusivity', 'Fragment Diffusivity', 'Node Diffusivity Std', 'Segment Diffusivity Std', 'Fragment Diffusivity Std', 'Fusion Rate', 'Fission Rate', 'Fusion Rate per Node', 'Fission Rate per Node', 'Cell ID']

    data_path_root = '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/'
    folder_to_label = {}
    with open(f"/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            folder_to_label[folder] = int(label)

    for feature in features:
        print(f"Doing {feature}")

        feature_values, embeddings, labels, image_paths = get_feats_and_embs(mitotnt_features, umap_emb, feature, identifier, data_path_root, folder_to_label)
        top_1_percentile = np.percentile(feature_values, 99)
        bottom_1_percentile = np.percentile(feature_values, 1)

        feature_values = np.clip(feature_values, bottom_1_percentile, top_1_percentile)
        feature_values = (feature_values - np.min(feature_values)) / (
                    np.max(feature_values) - np.min(feature_values))
        embeddings = [x[-1] for x in embeddings]  # only take the last timestep
        embeddings = np.array(embeddings)

        # do PCA of the embeddings

        pca = PCA(n_components=10)
        red_embeddings = pca.fit_transform(embeddings)
        print(pca.explained_variance_ratio_)
        # red_embeddings = np.array(embeddings)
        # find the correlation of the feature values with the PCA embeddings

        correlations = []
        for i in range(3):
            corr, _ = pearsonr(feature_values, red_embeddings[:, i])
            # get spearman correlation
            # corr, _ = spearmanr(feature_values, red_embeddings[:, i])
            correlations.append(corr)

        correlation_matrix.append(correlations)

    correlation_matrix = np.array(correlation_matrix)

    # plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, fmt=".2f",
                xticklabels=[f'UMAP{i + 1}' for i in range(correlation_matrix.shape[1])], yticklabels=features)
    plt.title('Correlation Matrix Heatmap')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.show()




