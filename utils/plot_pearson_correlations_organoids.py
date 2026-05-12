import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# Cal27
# FEATURES = {
#     'total_node_count_mean': 'Node Count',
#     'fragment_branching_index_mean': 'Node Degree',
#     'segment_length_mean': 'Segment\nLength',
#     'fragment_length_mean': 'Fragment\nLength',
#     'fragment_diameter_mean': 'Fragment\nDiameter',
#     'graph_density_mean': 'Graph\nDensity',
#     'graph_efficiency_mean': 'Graph\nEfficiency',
#     'graph_clustering_coefficient_mean': 'Clustering\nCoefficient',
#     'node_diffusivity_mean': 'Node\nDiffusivity',
#     'segment_diffusivity_mean': 'Segment\nDiffusivity',
#     'fragment_diffusivity_mean': 'Fragment\nDiffusivity',
#     'node_diffusivity_std': 'Node\nDiffusivity Std',
#     'segment_diffusivity_std': 'Segment\nDiffusivity Std',
#     'fragment_diffusivity_std': 'Fragment\nDiffusivity Std',
#     'fusion_rate_mean': 'Fusion Rate',
#     'fission_rate_mean': 'Fission Rate',
# }

# organoid
FEATURES = {
    "mean_branch_length": "Segment\nLength",
    "mean_fragment_length": "Fragment\nLength",
    "mean_fragment_diffusivity": "Fragment\nDiffusivity",
    "cytoplasm_volume": "Cell\nVolume",
    "mean_dist_mito_cell": "Mito-Membrane\nDistance",
}


def compute_pca_feature_correlations(
    parquet_path, features, n_pca_fit=10, n_heatmap_pcs=3
):
    """
    Fit PCA with up to n_pca_fit components (for explained variance on console).
    Heatmap uses Pearson r between first n_heatmap_pcs PCs and MitoTNT features.
    """
    df = pd.read_parquet(parquet_path)

    keep_list = ["SC", "Lde", "L15", "L60", "L130"]
    # Filter the dataframe
    print(f"WARNING: Applying organoid filtering!!!")
    df = df[df["cell_type"].isin(keep_list)].reset_index(drop=True)

    embeddings = np.stack(df["embeddings"].values).astype(np.float64)
    if embeddings.ndim == 3:
        embeddings = embeddings[:, -1, :]

    n_samples, d = embeddings.shape
    n_comp_fit = min(n_pca_fit, n_samples, d)

    pca = PCA(n_components=n_comp_fit)
    pcs = pca.fit_transform(embeddings)
    evr = pca.explained_variance_ratio_

    print("\n--- PCA explained variance (first 10 PCs, fraction of total variance) ---")
    for i in range(min(10, len(evr))):
        cum = evr[: i + 1].sum()
        print(f"  PC{i + 1}: {evr[i]:.6f}  (cumulative PC1..PC{i + 1}: {cum:.6f})")
    if len(evr) < 10:
        print(
            f"  (Only {len(evr)} PC(s) available: n_samples or dim limits components.)"
        )

    feat_cols = list(features.keys())
    feat_labels = list(features.values())
    feat_data = df[feat_cols].values.astype(float)

    valid_mask = ~np.isnan(feat_data).any(axis=1)
    pcs = pcs[valid_mask]
    feat_data = feat_data[valid_mask]
    print(f"Valid samples: {valid_mask.sum()} / {len(valid_mask)}")

    n_h = min(n_heatmap_pcs, pcs.shape[1])
    corr_matrix = np.zeros((n_h, len(feat_cols)))
    for i in range(n_h):
        for j in range(len(feat_cols)):
            r, _ = pearsonr(pcs[:, i], feat_data[:, j])
            corr_matrix[i, j] = r

    return corr_matrix, feat_labels, evr


def plot_heatmap(corr_matrix, feat_labels, explained_var, out_path, desc=None):
    n_components = corr_matrix.shape[0]
    pc_labels = [
        f"PC-{i + 1}\n({explained_var[i] * 100:.1f}%)" for i in range(n_components)
    ]

    cell_size = 1.2
    fig_w = n_components * cell_size + 2.5
    fig_h = len(feat_labels) * cell_size + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        square=True,
        data=corr_matrix.T,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.75,
        vmax=0.75,
        xticklabels=pc_labels,
        yticklabels=feat_labels,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={
            "label": "Pearson r",
            # 'shrink': 0,
            # 'fontsize': 18
        },
        ax=ax,
        annot_kws={
            "fontsize": 18,
            # 'fontweight': 'bold'
        },
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
    ax.tick_params(axis="both", length=0)

    if desc:
        plt.title(
            f"Pearson Correlation of PCA Components vs MitoTNT Features ({desc})",
            fontsize=12,
            pad=12,
        )
    else:
        plt.title(
            f"Pearson Correlation of PCA Components vs MitoTNT Features",
            fontsize=12,
            pad=12,
        )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":

    data_root = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data"
    # parquet_infile = "embeddings+metadata_vis_joined.parquet"
    #
    # # embeddings_dir = "ms4d_2024v3_252eps"
    # embeddings_dir = "ms4d_2024v3_supcon_190eps"
    #
    # parquet_path = os.path.join(data_root, embeddings_dir, parquet_infile)
    # # Console: explained variance for first 10 PCs. Heatmap: PC1–PC3 only.
    # corr_matrix, feat_labels, explained_var = compute_pca_feature_correlations(
    #     parquet_path, FEATURES, n_pca_fit=10, n_heatmap_pcs=3
    # )
    #
    # heatmap_path = os.path.join(data_root, embeddings_dir, 'pca_correlation_heatmap.png')
    # plot_heatmap(corr_matrix, feat_labels, explained_var, heatmap_path, desc=embeddings_dir)

    parquet_infile = "embeddings+metadata.parquet"

    # embeddings_dir = "ms4d_2024v3_252eps"
    # embeddings_dir = "ms4d_2024v3_supcon_190eps"
    embeddings_dir = "ms4d_organoid_embeddings"

    parquet_path = os.path.join(data_root, embeddings_dir, parquet_infile)
    # Console: explained variance for first 10 PCs. Heatmap: PC1–PC3 only.
    corr_matrix, feat_labels, explained_var = compute_pca_feature_correlations(
        parquet_path, FEATURES, n_pca_fit=10, n_heatmap_pcs=3
    )

    heatmap_path = os.path.join(
        data_root, embeddings_dir, "pca_correlation_heatmap.png"
    )
    plot_heatmap(
        corr_matrix, feat_labels, explained_var, heatmap_path, desc=embeddings_dir
    )
