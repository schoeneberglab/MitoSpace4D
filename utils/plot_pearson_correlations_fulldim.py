import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cal27
FEATURES = {
    'fragment_diameter_mean': 'Fragment\nDiameter',
    'fragment_length_mean': 'Fragment\nLength',
    'segment_length_mean': 'Segment\nLength',
    'total_node_count_mean': 'Node Count',
    'fragment_tortuosity_mean': 'Fragment\nTortuosity',
    'fragment_branchpoint_to_endpoint_ratio_mean': 'Branch-\nEnd Ratio',
    'graph_density_mean': 'Graph\nDensity',
    'graph_efficiency_mean': 'Graph\nEfficiency',
    'fragment_diffusivity_mean': 'Fragment\nDiffusivity',
    'segment_diffusivity_mean': 'Segment\nDiffusivity',
    'node_diffusivity_mean': 'Node\nDiffusivity',
    'fusion_rate_mean': 'Fusion Rate',
    'fission_rate_mean': 'Fission Rate',
}

# organoid
# FEATURES = {
#     'mean_branch_length': 'Segment\nLength',
#     'mean_fragment_length': 'Fragment\nLength',
#     'mean_fragment_diffusivity': 'Fragment\nDiffusivity',
#     'cytoplasm_volume': "Cell\nVolume",
#     'mean_dist_mito_cell': "Mito-Membrane\nDistance",
# }


def compute_full_dim_feature_correlations(parquet_path, features, show_n_top=10):
    """
    Pearson r between every embedding dimension (full 2048-d) and each MitoTNT feature.
    For each feature, return the top-`show_n_top` dimensions ranked by |r|.
    """
    df = pd.read_parquet(parquet_path)

    embeddings = np.stack(df['embeddings'].values).astype(np.float64)
    if embeddings.ndim == 3:
        embeddings = embeddings[:, -1, :]

    feat_cols = list(features.keys())
    feat_labels = list(features.values())
    feat_data = df[feat_cols].values.astype(float)

    valid_mask = ~np.isnan(feat_data).any(axis=1)
    embeddings = embeddings[valid_mask]
    feat_data = feat_data[valid_mask]
    print(f'Valid samples: {valid_mask.sum()} / {len(valid_mask)}')

    n_samples, n_dim = embeddings.shape
    n_feat = len(feat_cols)
    print(f'Computing Pearson r over {n_dim} embedding dims × {n_feat} features ({n_samples} samples)...')

    emb_c = embeddings - embeddings.mean(axis=0, keepdims=True)
    feat_c = feat_data - feat_data.mean(axis=0, keepdims=True)
    emb_std = embeddings.std(axis=0, ddof=0)
    feat_std = feat_data.std(axis=0, ddof=0)
    denom = np.outer(emb_std, feat_std) * n_samples
    denom[denom == 0] = np.nan
    full_corr = (emb_c.T @ feat_c) / denom
    full_corr = np.nan_to_num(full_corr, nan=0.0)

    show_n_top = min(show_n_top, n_dim)
    top_dims = np.zeros((show_n_top, n_feat), dtype=int)
    top_corr = np.zeros((show_n_top, n_feat))
    for j in range(n_feat):
        order = np.argsort(-np.abs(full_corr[:, j]))[:show_n_top]
        top_dims[:, j] = order
        top_corr[:, j] = full_corr[order, j]

    return top_corr, top_dims, feat_labels


def plot_heatmap(top_corr, top_dims, feat_labels, out_path, desc=None):
    n_rows, n_cols = top_corr.shape
    rank_labels = [f'Top {i + 1}' for i in range(n_rows)]

    annot = np.empty_like(top_corr, dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            annot[i, j] = f'd{top_dims[i, j]}\n{top_corr[i, j]:.2f}'

    cell_w = 1.0
    cell_h = 0.55
    fig_w = n_cols * cell_w + 2.5
    fig_h = n_rows * cell_h + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        top_corr,
        annot=annot,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        xticklabels=feat_labels,
        yticklabels=rank_labels,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
        ax=ax,
        annot_kws={
            'fontsize': 8,
        },
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.tick_params(axis='both', length=0)
    title = f'Top-{n_rows} Embedding Dimensions per MitoTNT Feature (Pearson r)'
    if desc:
        title = f'{title} ({desc})'
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved heatmap to {out_path}')


if __name__ == '__main__':

    data_root = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data"
    parquet_infile = "embeddings+metadata_vis_joined.parquet"

    embeddings_dir = "ms4d_2024v3_252eps"
    # embeddings_dir = "ms4d_2024v3_random_init"

    # embeddings_dir = "ms4d_2024v3_supcon_190eps"

    show_n_top = 10

    parquet_path = os.path.join(data_root, embeddings_dir, parquet_infile)
    top_corr, top_dims, feat_labels = compute_full_dim_feature_correlations(
        parquet_path, FEATURES, show_n_top=show_n_top
    )

    heatmap_path = os.path.join(
        data_root, embeddings_dir, f'fulldim_correlation_heatmap_top{show_n_top}.png'
    )
    plot_heatmap(top_corr, top_dims, feat_labels, heatmap_path, desc=embeddings_dir)

    # parquet_infile = "embeddings+metadata.parquet"
    #
    # # embeddings_dir = "ms4d_2024v3_252eps"
    # # embeddings_dir = "ms4d_2024v3_supcon_190eps"
    # embeddings_dir = "ms4d_organoid_embeddings"
    #
    # parquet_path = os.path.join(data_root, embeddings_dir, parquet_infile)
    # top_corr, top_dims, feat_labels = compute_full_dim_feature_correlations(
    #     parquet_path, FEATURES, show_n_top=show_n_top
    # )
    #
    # heatmap_path = os.path.join(
    #     data_root, embeddings_dir, f'fulldim_correlation_heatmap_top{show_n_top}.png'
    # )
    # plot_heatmap(top_corr, top_dims, feat_labels, heatmap_path, desc=embeddings_dir)
