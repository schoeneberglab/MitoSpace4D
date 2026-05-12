ROOT_DIR = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data"
PROJ_DIR = "/home/earkfeld/Projects/MitoSpace4D"

"""
Generate comparative figure panels for 4D vs 3D vs 2D self-supervised embeddings.

Produces:
  1. grouped_per_class_accuracy.pdf    – Per-class Top-1 KNN accuracy (grouped bars, sorted by 4D)
  2. delta_accuracy_heatmap.pdf        – Δ accuracy: 4D−3D and 4D−2D per treatment
  3. aggregate_metrics_table.pdf       – Summary table: Top-1 acc, macro-F1, silhouette, Calinski-Harabasz
  4. neighborhood_purity.pdf           – k-NN label purity boxplots
  5. interclass_distance_dendrograms.pdf – Cosine inter-class distance dendrograms
  6. retrieval_precision_at_k.pdf      – Mean precision @ k curves

Usage:
  python generate_comparison_figures.py

Adjust `ROOT_DIR`, `PROJ_DIR`, and `EMBEDDING_DIRS` below to match your setup.
"""

import os
import os.path as osp

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    calinski_harabasz_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

EMBEDDING_DIRS = {
    "4D": "ms4d_2024v3_252eps",
    "3D": "ms3d_2024v3_225eps",
    "2D": "ms2d_2024v3",
}

OUTPUT_DIR = osp.join(PROJ_DIR, "tmp_plotting")

SPLIT_PERC = 0.8
BALANCE_CLASSES = True
BALANCED_SPLIT = True
SEED = 1123
K_NEIGHBORS = 100
PURITY_K_VALUES = [1, 3, 5, 10, 25, 50, 75, 100]
BOXPLOT_K_VALUES = [5, 10, 25, 50, 100]  # subset used for the boxplot figure

# Colour palette for the three dimensionalities
COLORS = {"4D": "#2166ac", "3D": "#66c2a5", "2D": "#fc8d62"}


def load_drug_label_dicts(proj_dir):
    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", "r") as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug
    return drug_labels_dict, label_drug_dict


def balance_label_counts(embeddings, labels, seed=1123):
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    min_count = min((labels == lbl).sum() for lbl in unique_labels)
    indices = []
    for lbl in unique_labels:
        lbl_idx = np.where(labels == lbl)[0]
        indices.append(rng.choice(lbl_idx, size=min_count, replace=False))
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return embeddings[indices], labels[indices]


def split_dataset(embeddings, labels, split_perc=0.8, balanced=True, seed=1123):
    unique_labels = np.unique(labels)
    train_idx, val_idx = [], []
    for lbl in unique_labels:
        lbl_idx = np.where(labels == lbl)[0]
        tr, va = train_test_split(
            lbl_idx, train_size=split_perc, random_state=seed, shuffle=True
        )
        train_idx.extend(tr)
        val_idx.extend(va)
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    return (
        embeddings[train_idx],
        labels[train_idx],
        embeddings[val_idx],
        labels[val_idx],
    )


def topKfrequent(neighbors, distances, k, weighted=True):
    """Return the top-k most frequent labels among neighbors, weighted by distance."""
    freq = {}
    for lbl, d in zip(neighbors, distances):
        w = d if weighted else 1.0
        freq[lbl] = freq.get(lbl, 0) + w
    sorted_lbls = sorted(freq, key=freq.get, reverse=True)
    return sorted_lbls[:k]


def cosine_distance(eval_emb, train_emb):
    dist = eval_emb @ train_emb.T
    idxs = (-dist).argsort(1)
    sorted_dist = np.take_along_axis(dist, idxs, axis=1)
    return sorted_dist, idxs


def knn_predictions(eval_emb, eval_labels, train_emb, train_labels, k=100):
    dist_matrix, dist_idxs = cosine_distance(eval_emb, train_emb)
    dist_matrix = (dist_matrix + 1) / 2  # normalise to [0,1]

    preds = []
    for i in range(len(eval_emb)):
        k_nn_labels = train_labels[dist_idxs[i][:k]]
        k_nn_dists = dist_matrix[i][:k]
        top_lbl = topKfrequent(k_nn_labels, k_nn_dists, 1, weighted=True)
        preds.append(top_lbl[0])
    return np.array(preds)


def per_class_accuracy(eval_labels, preds, unique_labels):
    acc = {}
    for lbl in unique_labels:
        mask = eval_labels == lbl
        if mask.sum() == 0:
            acc[lbl] = 0.0
        else:
            acc[lbl] = (preds[mask] == lbl).sum() / mask.sum() * 100.0
    return acc


def neighborhood_label_purity(embeddings, labels, k_values):
    """For each sample, fraction of k-nearest neighbours sharing the same label."""
    sim = embeddings @ embeddings.T
    purities = {}
    for k in k_values:
        idxs = (-sim).argsort(1)[:, 1 : k + 1]  # exclude self
        nn_labels = labels[idxs]
        match = (nn_labels == labels[:, None]).mean(axis=1)
        purities[k] = match
    return purities


def get_entry_id(path):
    """Extract entry_id as 'parent_folder/filename' from a full path."""
    parts = path.split("/")
    return parts[-2] + "/" + parts[-1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    drug_labels_dict, label_drug_dict = load_drug_label_dicts(PROJ_DIR)

    results = {}

    for dim_label, emb_subdir in EMBEDDING_DIRS.items():
        emb_dir = osp.join(ROOT_DIR, emb_subdir)
        print(f"\n{'='*60}")
        print(f"  Loading {dim_label} embeddings from {emb_dir}")
        print(f"{'='*60}")

        df = pd.read_parquet(osp.join(emb_dir, "embeddings+metadata_vis.parquet"))

        df["entry_id"] = df["image_paths"].apply(get_entry_id)

        exclude_paths = pd.read_parquet(
            osp.join(ROOT_DIR, "2024v3_exclude_paths.parquet")
        )
        print(exclude_paths.columns)
        exclude_paths = exclude_paths["image_paths"].tolist()

        exclude_ids = set(get_entry_id(p) for p in exclude_paths)
        include_mask = ~df["entry_id"].isin(exclude_ids)
        df = df[include_mask]

        embeddings = np.stack(df["embeddings"])
        labels = np.array(df["labels"])

        # Filter dicts to labels in dataset
        ul = set(labels)
        dld = {d: l for d, l in drug_labels_dict.items() if l in ul}
        ldd = {l: d for l, d in label_drug_dict.items() if l in ul}

        if BALANCE_CLASSES:
            embeddings, labels = balance_label_counts(embeddings, labels, seed=SEED)

        train_emb, train_lbl, eval_emb, eval_lbl = split_dataset(
            embeddings,
            labels,
            split_perc=SPLIT_PERC,
            balanced=BALANCED_SPLIT,
            seed=SEED,
        )

        # Handle >2D embeddings (take last timestep)
        if train_emb.ndim > 2:
            train_emb = train_emb[:, -1, :]
        if eval_emb.ndim > 2:
            eval_emb = eval_emb[:, -1, :]

        unique_labels = sorted(ldd.keys())

        # KNN predictions
        preds = knn_predictions(eval_emb, eval_lbl, train_emb, train_lbl, k=K_NEIGHBORS)

        # Per-class accuracy
        pc_acc = per_class_accuracy(eval_lbl, preds, unique_labels)

        # Aggregate metrics
        top1_acc = (preds == eval_lbl).sum() / len(eval_lbl) * 100.0
        macro_f1 = f1_score(eval_lbl, preds, average="macro", zero_division=0) * 100.0

        # Clustering metrics on ALL embeddings (train + eval combined)
        all_emb = np.concatenate([train_emb, eval_emb])
        all_lbl = np.concatenate([train_lbl, eval_lbl])

        # Silhouette on a subsample for speed
        n_sil = min(5000, len(all_emb))
        rng = np.random.default_rng(SEED)
        sil_idx = rng.choice(len(all_emb), n_sil, replace=False)
        sil_score = silhouette_score(
            all_emb[sil_idx], all_lbl[sil_idx], metric="cosine"
        )

        ch_score = calinski_harabasz_score(all_emb, all_lbl)

        # Neighborhood purity
        n_pur = min(5000, len(all_emb))
        pur_idx = rng.choice(len(all_emb), n_pur, replace=False)
        purities = neighborhood_label_purity(
            all_emb[pur_idx], all_lbl[pur_idx], PURITY_K_VALUES
        )

        # Inter-class centroid distance matrix
        centroids = np.array(
            [all_emb[all_lbl == lbl].mean(axis=0) for lbl in unique_labels]
        )
        centroids_normed = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        interclass_cos_sim = centroids_normed @ centroids_normed.T
        interclass_cos_dist = 1 - interclass_cos_sim

        # Confusion matrix
        cm = confusion_matrix(eval_lbl, preds, labels=unique_labels)

        results[dim_label] = {
            "per_class_acc": pc_acc,
            "top1_acc": top1_acc,
            "macro_f1": macro_f1,
            "silhouette": sil_score,
            "calinski_harabasz": ch_score,
            "purities": purities,
            "interclass_dist": interclass_cos_dist,
            "confusion_matrix": cm,
            "unique_labels": unique_labels,
            "label_drug_dict": ldd,
            "preds": preds,
            "eval_labels": eval_lbl,
        }

    # Use 4D label ordering as reference
    ref = results["4D"]
    unique_labels = ref["unique_labels"]
    ldd = ref["label_drug_dict"]
    drug_names = [ldd[l] for l in unique_labels]

    # ──────────────────────────────────────────
    #  PLOT 1: Grouped per-class accuracy bars
    # ──────────────────────────────────────────
    print("\nGenerating grouped per-class accuracy bar chart...")

    acc_4d = [results["4D"]["per_class_acc"].get(l, 0) for l in unique_labels]
    acc_3d = [results["3D"]["per_class_acc"].get(l, 0) for l in unique_labels]
    acc_2d = [results["2D"]["per_class_acc"].get(l, 0) for l in unique_labels]

    # Sort by 4D accuracy (descending)
    sort_idx = np.argsort(acc_4d)[::-1]
    sorted_names = [drug_names[i] for i in sort_idx]
    sorted_4d = [acc_4d[i] for i in sort_idx]
    sorted_3d = [acc_3d[i] for i in sort_idx]
    sorted_2d = [acc_2d[i] for i in sort_idx]

    n_classes = len(unique_labels)
    x = np.arange(n_classes)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        x - bar_w,
        sorted_4d,
        bar_w,
        label="4D",
        color=COLORS["4D"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        sorted_3d,
        bar_w,
        label="3D",
        color=COLORS["3D"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + bar_w,
        sorted_2d,
        bar_w,
        label="2D",
        color=COLORS["2D"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 KNN Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(OUTPUT_DIR, "grouped_per_class_accuracy.pdf"), dpi=300)
    fig.savefig(osp.join(OUTPUT_DIR, "grouped_per_class_accuracy.png"), dpi=300)
    plt.close(fig)
    print(f"  -> Saved grouped_per_class_accuracy.pdf/png")

    # ──────────────────────────────────────────
    #  PLOT 2: Delta accuracy heatmap
    # ──────────────────────────────────────────
    print("Generating delta accuracy heatmap...")

    delta_4d_3d = np.array(acc_4d) - np.array(acc_3d)
    delta_4d_2d = np.array(acc_4d) - np.array(acc_2d)

    # Sort by 4D−2D delta (largest gain first)
    sort_idx_delta = np.argsort(delta_4d_2d)[::-1]
    delta_matrix = np.stack(
        [
            delta_4d_3d[sort_idx_delta],
            delta_4d_2d[sort_idx_delta],
        ],
        axis=1,
    )
    delta_names = [drug_names[i] for i in sort_idx_delta]

    vabs = max(abs(delta_matrix.min()), abs(delta_matrix.max()))

    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(
        delta_matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vabs,
        vmax=vabs,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(delta_names)))
    ax.set_yticklabels(delta_names, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["4D − 3D", "4D − 2D"], fontsize=9)
    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > vabs * 0.6 else "black"
            ax.text(
                j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7, color=color
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Δ Accuracy (pp)", fontsize=9)
    ax.set_title("Accuracy gain of 4D over lower dims", fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(OUTPUT_DIR, "delta_accuracy_heatmap.pdf"), dpi=300)
    fig.savefig(osp.join(OUTPUT_DIR, "delta_accuracy_heatmap.png"), dpi=300)
    plt.close(fig)
    print(f"  -> Saved delta_accuracy_heatmap.pdf/png")

    # ──────────────────────────────────────────
    #  PLOT 3: Aggregate metrics summary table
    # ──────────────────────────────────────────
    print("Generating aggregate metrics table...")

    metrics_df = pd.DataFrame(
        {
            "Dimensionality": ["4D", "3D", "2D"],
            "Top-1 Acc (%)": [results[d]["top1_acc"] for d in ["4D", "3D", "2D"]],
            "Macro F1 (%)": [results[d]["macro_f1"] for d in ["4D", "3D", "2D"]],
            "Silhouette": [results[d]["silhouette"] for d in ["4D", "3D", "2D"]],
            "Calinski-Harabasz": [
                results[d]["calinski_harabasz"] for d in ["4D", "3D", "2D"]
            ],
        }
    )
    metrics_df.to_csv(osp.join(OUTPUT_DIR, "aggregate_metrics.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.axis("off")
    table = ax.table(
        cellText=[
            [
                row["Dimensionality"],
                f'{row["Top-1 Acc (%)"]:.1f}',
                f'{row["Macro F1 (%)"]:.1f}',
                f'{row["Silhouette"]:.3f}',
                f'{row["Calinski-Harabasz"]:.0f}',
            ]
            for _, row in metrics_df.iterrows()
        ],
        colLabels=metrics_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e0e0e0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#cccccc")

    fig.tight_layout()
    fig.savefig(
        osp.join(OUTPUT_DIR, "aggregate_metrics_table.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        osp.join(OUTPUT_DIR, "aggregate_metrics_table.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  -> Saved aggregate_metrics_table.pdf/png + aggregate_metrics.csv")

    # ──────────────────────────────────────────
    #  PLOT 4: Neighborhood label purity boxplots
    # ──────────────────────────────────────────
    print("Generating neighborhood purity boxplots...")

    fig, axes = plt.subplots(
        1, len(BOXPLOT_K_VALUES), figsize=(3 * len(BOXPLOT_K_VALUES), 4), sharey=True
    )
    if len(BOXPLOT_K_VALUES) == 1:
        axes = [axes]

    for ax_i, k in enumerate(BOXPLOT_K_VALUES):
        data_to_plot = []
        labels_to_plot = []
        for dim in ["4D", "3D", "2D"]:
            data_to_plot.append(results[dim]["purities"][k])
            labels_to_plot.append(dim)

        bp = axes[ax_i].boxplot(
            data_to_plot,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, dim in zip(bp["boxes"], ["4D", "3D", "2D"]):
            patch.set_facecolor(COLORS[dim])
            patch.set_alpha(0.7)
        axes[ax_i].set_xticklabels(labels_to_plot)
        axes[ax_i].set_title(f"k = {k}", fontsize=10)
        if ax_i == 0:
            axes[ax_i].set_ylabel("Neighbour label purity")
        axes[ax_i].spines["top"].set_visible(False)
        axes[ax_i].spines["right"].set_visible(False)

    fig.suptitle("Fraction of k-NN sharing same treatment label", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(
        osp.join(OUTPUT_DIR, "neighborhood_purity.pdf"), dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        osp.join(OUTPUT_DIR, "neighborhood_purity.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  -> Saved neighborhood_purity.pdf/png")

    # ──────────────────────────────────────────
    #  PLOT 5: Inter-class distance dendrograms
    # ──────────────────────────────────────────
    print("Generating inter-class distance dendrograms...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_i, dim in enumerate(["4D", "3D", "2D"]):
        dim_unique_labels = results[dim]["unique_labels"]
        dim_ldd = results[dim]["label_drug_dict"]
        dim_drug_names = [dim_ldd[l] for l in dim_unique_labels]

        dist_mat = results[dim]["interclass_dist"].copy()
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0)
        dist_mat = np.clip(dist_mat, 0, None)
        condensed = squareform(dist_mat)
        Z = linkage(condensed, method="average")
        dendrogram(
            Z,
            labels=dim_drug_names,
            ax=axes[ax_i],
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=0,
        )
        axes[ax_i].set_title(f"{dim}", fontsize=12, fontweight="bold")
        axes[ax_i].set_ylabel("Cosine distance")
        axes[ax_i].spines["top"].set_visible(False)
        axes[ax_i].spines["right"].set_visible(False)

    fig.suptitle("Hierarchical clustering of treatment centroids", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(
        osp.join(OUTPUT_DIR, "interclass_distance_dendrograms.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        osp.join(OUTPUT_DIR, "interclass_distance_dendrograms.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  -> Saved interclass_distance_dendrograms.pdf/png")

    # ──────────────────────────────────────────
    #  PLOT 6: Retrieval precision-at-k curves
    # ──────────────────────────────────────────
    print("Generating retrieval precision-at-k curves...")

    k_range = PURITY_K_VALUES

    fig, ax = plt.subplots(figsize=(6, 4))
    for dim in ["4D", "3D", "2D"]:
        prec_at_k = []
        for k in k_range:
            purities_k = results[dim]["purities"].get(k, None)
            if purities_k is not None:
                prec_at_k.append(purities_k.mean() * 100)
            else:
                prec_at_k.append(np.nan)

        valid = [(kv, pv) for kv, pv in zip(k_range, prec_at_k) if not np.isnan(pv)]
        if valid:
            ks, ps = zip(*valid)
            ax.plot(
                ks, ps, "o-", color=COLORS[dim], label=dim, linewidth=2, markersize=5
            )

    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Mean precision @ k (%)")
    ax.set_title("Retrieval precision at k")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(OUTPUT_DIR, "retrieval_precision_at_k.pdf"), dpi=300)
    fig.savefig(osp.join(OUTPUT_DIR, "retrieval_precision_at_k.png"), dpi=300)
    plt.close(fig)
    print(f"  -> Saved retrieval_precision_at_k.pdf/png")

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
