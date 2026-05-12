import numpy as np
import os
from interpretability.mitotnt_on_mitospace import get_feats_and_embs, get_identifiers
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp


def get_feats(features, identifiers, embeddings, feature_to_plot):
    # check there are no nans
    assert features[feature_to_plot].isna().sum() == 0
    identifiers_csv = features['Unnamed: 0']
    identifiers_csv = [str(id) for id in identifiers_csv]

    features_values_to_plot = []
    embeddings_filtered = []
    for embd_idx, id in enumerate(identifiers):
        if id in identifiers_csv:
            idx = identifiers_csv.index(id)
            features_values_to_plot.append(features[feature_to_plot][idx])
            embeddings_filtered.append(embeddings[embd_idx])
        else:
            print(f"Identifier {id} not found in the features csv file.")

    embeddings_filtered = np.array(embeddings_filtered)
    return features_values_to_plot, embeddings_filtered


def preprocess(feature_to_plot):
    feature_to_plot = np.array(feature_to_plot)
    feature_to_plot = (feature_to_plot - feature_to_plot.min()) / (feature_to_plot.max() - feature_to_plot.min())

    # plot the distribution of the feature
    plt.hist(feature_to_plot, bins=50)
    plt.show()

    return feature_to_plot


if __name__ == "__main__":
    root_path = '/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/'
    mitotnt_features = pd.read_csv(osp.join(root_path, 'df_all_bins_20250303.csv'))
    emb = np.load(osp.join(root_path, 'OrganoAgeSpace.npy'))
    identifiers = np.load(osp.join(root_path, 'identifiers.npy'))

    # DO PCA on the embeddings
    pca = PCA(n_components=10)
    umap_emb = pca.fit_transform(emb)
    print(pca.explained_variance_ratio_)

    correlation_matrix = []
    features = ['FLD_ave', 'BLD_ave', 'DFDd_ave', 'DSDd_ave', 'cell volume', 'ave mito to membrane']

    for feature in features:
        print(f"Doing {feature}")

        feature, embeddings = get_feats(mitotnt_features, identifiers, umap_emb, feature)
        top_1_percentile = np.percentile(feature, 99)
        bottom_1_percentile = np.percentile(feature, 1)

        feature_values = np.clip(feature, bottom_1_percentile, top_1_percentile)
        feature_values = preprocess(feature_values)
        # embeddings = [x[-1] for x in embeddings]  # only take the last timestep

        # do PCA of the embeddings

        # pca = PCA(n_components=3)
        # red_embeddings = pca.fit_transform(embeddings)
        red_embeddings = np.array(embeddings)
        # find the correlation of the feature values with the PCA embeddings

        correlations = []
        for i in range(3):
            # corr, _ = pearsonr(feature_values, red_embeddings[:, i])
            # get spearman correlation
            corr, _ = spearmanr(feature_values, red_embeddings[:, i])
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
