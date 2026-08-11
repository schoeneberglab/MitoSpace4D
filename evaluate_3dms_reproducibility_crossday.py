"""
Leave-one-day-out k-NN evaluation of the MitoSpace-3D reproducibility embeddings.

Mirrors the protocol of `utils/evaluate_reproducibility_crossday.py` (used for the 4D
model) so the numbers are directly comparable: classes are balanced to the smallest
class, each acquisition day is held out in turn, and every held-out cell is classified by
a similarity-weighted vote over its k=100 nearest training neighbours (cosine).

The reproducibility dataset is two dye batches of three days each, so this runs one
3-fold leave-one-day-out cross-validation per batch:
    deepred     20260330, 20260331, 20260401   (MitoTracker DeepRed)
    mitobright  20260407, 20260408, 20260409   (MitoBright)
Each batch contains the same three conditions: control, oligomycin, nocodazole.

Usage:
    python evaluate_3dms_reproducibility_crossday.py --embeddings_dir 3dms_reproducibility
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.utils import topKfrequent

np.random.seed(0)

CM_FONT_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 16,
    "legend.fontsize": 14,
}

DAY_GROUPS = {
    "deepred": ["20260330", "20260331", "20260401"],
    "mitobright": ["20260407", "20260408", "20260409"],
}
GROUP_TITLES = {
    "deepred": "MitoSpace-3D Top-1 kNN (k=100) - MitoTracker DeepRed",
    "mitobright": "MitoSpace-3D Top-1 kNN (k=100) - MitoBright",
}

parser = argparse.ArgumentParser(description="MitoSpace-3D cross-day kNN evaluation")
parser.add_argument(
    "--embeddings_dir",
    default="/home/earkfeld/Projects/MitoSpace4D/3dms_reproducibility",
    help="Directory holding embeddings.npy / labels.npy / metadata.csv.",
)
parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbours.")
parser.add_argument(
    "--top_ns", nargs="+", type=int, default=[1, 3], help="Top-N accuracies to report."
)
parser.add_argument(
    "--balance_classes",
    default=True,
    action="store_true",
    help="Subsample every class to the smallest class count within a batch.",
)
parser.add_argument("--seed", type=int, default=1123)


def cosine_distance(eval_embeddings, train_embeddings):
    """Cosine similarity of L2-normalized embeddings, sorted descending per row."""
    dist_matrix = eval_embeddings @ train_embeddings.T
    dist_matrix_idxs = (-1 * dist_matrix).argsort(1)
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    return dist_matrix_sorted, dist_matrix_idxs


def balance_label_counts(embeddings, labels, dates, seed=1123, shuffle=True):
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    min_count = min(int(np.sum(labels == lbl)) for lbl in unique_labels)

    indices = []
    for lbl in unique_labels:
        lbl_indices = np.where(labels == lbl)[0]
        indices.append(rng.choice(lbl_indices, size=min_count, replace=False))
    indices = np.concatenate(indices)
    if shuffle:
        rng.shuffle(indices)

    return embeddings[indices], labels[indices], dates[indices]


def nearest_neighbor_evaluation(
    eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs, k=100
):
    """Similarity-weighted k-NN vote. Returns (preds, correct_preds_idxs)."""
    # Map cosine similarity from [-1, 1] to [0, 1] so votes are non-negative.
    dist_matrix = (dist_matrix + 1) / 2

    preds = {top_n: [] for top_n in top_ns}
    correct_preds_idxs = {top_n: [] for top_n in top_ns}

    for i in tqdm(range(len(dist_matrix)), leave=False):
        eval_lbl = eval_labels[i]
        k_nearest_nbs = train_labels[dist_matrix_idxs[i][:k]]
        k_nearest_dist = dist_matrix[i][:k]

        for top_n in top_ns:
            top_most_freq_lbls = topKfrequent(
                k_nearest_nbs, k_nearest_dist, top_n, weighted=True
            )
            if eval_lbl in top_most_freq_lbls:
                preds[top_n].append(eval_lbl)
                correct_preds_idxs[top_n].append(i)
            else:
                preds[top_n].append(top_most_freq_lbls[0])

    return preds, correct_preds_idxs


def plot_cm(
    gt_labels,
    pred_labels,
    label_drug_dict,
    show_values=True,
    title="Confusion matrix",
    normalize=True,
    save_path=None,
):
    labels_idx = sorted(label_drug_dict.keys())
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels_idx)

    with plt.rc_context(CM_FONT_RC):
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        label_names = [label_drug_dict[idx] for idx in labels_idx]
        tickmarks = np.arange(len(label_names))
        ax.set_xticks(tickmarks)
        ax.set_xticklabels(label_names, rotation=90)
        ax.set_yticks(tickmarks)
        ax.set_yticklabels(label_names)

        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_plot = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                cm_plot = np.nan_to_num(cm_plot)
            im = ax.imshow(
                cm_plot, cmap=plt.get_cmap("Blues"), interpolation="nearest", vmin=0.0, vmax=1.0
            )
        else:
            cm_plot = cm.copy()
            im = ax.imshow(cm_plot, cmap=plt.get_cmap("Blues"), interpolation="nearest")

        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if normalize:
            fig.colorbar(im, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            fig.colorbar(im, cax=cax)

        if show_values:
            thresh = cm_plot.max() / 2.0 if cm_plot.max() > 0 else 0.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    value = f"{cm_plot[i, j]:.2f}" if normalize else f"{cm[i, j]}"
                    ax.text(
                        j,
                        i,
                        value,
                        horizontalalignment="center",
                        color="white" if cm_plot[i, j] > thresh else "black",
                    )

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

    return cm


def run_fold(
    test_date,
    embeddings,
    labels,
    dates,
    label_drug_dict,
    top_ns,
    k,
    title,
    out_dir,
):
    test_mask = dates == test_date
    train_mask = ~test_mask
    if test_mask.sum() == 0:
        raise ValueError(f"No samples for test_date={test_date!r}.")

    train_embeddings, train_labels = embeddings[train_mask], labels[train_mask]
    eval_embeddings, eval_labels = embeddings[test_mask], labels[test_mask]

    print(f"  Test:  {test_mask.sum()} cells (day={test_date})")
    print(
        f"  Train: {train_mask.sum()} cells (days={sorted(np.unique(dates[train_mask]).tolist())})"
    )

    dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings, train_embeddings)
    preds, correct_preds_idxs = nearest_neighbor_evaluation(
        eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs, k=k
    )

    plot_cm(
        eval_labels,
        preds[1],
        label_drug_dict,
        show_values=True,
        title=f"{title}\nheld-out day {test_date}",
        normalize=True,
        save_path=osp.join(out_dir, f"cm_top1-test_date={test_date}.png"),
    )
    plot_cm(
        eval_labels,
        preds[1],
        label_drug_dict,
        show_values=True,
        title=f"{title}\nheld-out day {test_date}",
        normalize=False,
        save_path=osp.join(out_dir, f"cm_top1_counts-test_date={test_date}.png"),
    )

    fold_rows, per_drug_rows = [], []
    for top_n in top_ns:
        n_correct = len(correct_preds_idxs[top_n])
        n_total = len(eval_labels)
        fold_rows.append(
            {
                "test_date": test_date,
                "dist_metric": "cosine",
                "k": k,
                "top_n": top_n,
                "n_correct": int(n_correct),
                "n_total": int(n_total),
                "accuracy": n_correct * 100.0 / n_total,
            }
        )
        for lbl in np.unique(train_labels):
            total = int(np.sum(eval_labels == lbl))
            correct = int(np.sum(eval_labels[correct_preds_idxs[top_n]] == lbl))
            per_drug_rows.append(
                {
                    "test_date": test_date,
                    "k": k,
                    "top_n": top_n,
                    "drug": label_drug_dict[int(lbl)],
                    "label": int(lbl),
                    "n_correct": correct,
                    "n_total": total,
                    "accuracy": correct * 100.0 / total if total > 0 else 0.0,
                }
            )
        print(f"  Top-{top_n} accuracy: {n_correct * 100.0 / n_total:.2f}%")

    return fold_rows, per_drug_rows, (np.asarray(eval_labels), np.asarray(preds[1]))


def run_group(group, days, embeddings, labels, dates, label_drug_dict, args, crossval_dir):
    print(f"\n########## Batch: {group} ({', '.join(days)}) ##########")
    out_dir = osp.join(crossval_dir, group)
    os.makedirs(out_dir, exist_ok=True)

    mask = np.isin(dates, days)
    g_emb, g_lbl, g_dates = embeddings[mask], labels[mask], dates[mask]
    print(f"{mask.sum()} cells; class counts {dict(zip(*np.unique(g_lbl, return_counts=True)))}")

    if args.balance_classes:
        g_emb, g_lbl, g_dates = balance_label_counts(g_emb, g_lbl, g_dates, seed=args.seed)
        print(
            f"Balanced to {dict(zip(*np.unique(g_lbl, return_counts=True)))} "
            f"({len(g_lbl)} cells total)"
        )

    g_label_drug = {int(l): label_drug_dict[int(l)] for l in np.unique(g_lbl)}
    title = GROUP_TITLES.get(group, f"MitoSpace-3D Top-1 kNN (k={args.k}) - {group}")

    fold_rows, per_drug_rows, top1_gt, top1_preds = [], [], [], []
    for test_date in days:
        print(f"\n===== Fold: held-out day {test_date} =====")
        f_rows, d_rows, (gt, pred) = run_fold(
            test_date,
            g_emb,
            g_lbl,
            g_dates,
            g_label_drug,
            args.top_ns,
            args.k,
            title,
            out_dir,
        )
        fold_rows.extend(f_rows)
        per_drug_rows.extend(d_rows)
        top1_gt.append(gt)
        top1_preds.append(pred)

    gt_concat = np.concatenate(top1_gt)
    preds_concat = np.concatenate(top1_preds)

    # Counts summed over the three folds (every cell is held out exactly once).
    cm_counts = plot_cm(
        gt_concat,
        preds_concat,
        g_label_drug,
        show_values=True,
        title=f"{title}\n3-fold leave-one-day-out (counts)",
        normalize=False,
        save_path=osp.join(out_dir, "cm_top1-sum_across_folds_counts.png"),
    )
    # Row-normalized version of the same pooled matrix.
    plot_cm(
        gt_concat,
        preds_concat,
        g_label_drug,
        show_values=True,
        title=f"{title}\n3-fold leave-one-day-out",
        normalize=True,
        save_path=osp.join(out_dir, "cm_top1-mean_across_folds_normalized.png"),
    )
    np.save(osp.join(out_dir, "cm_top1-sum_across_folds.npy"), cm_counts)
    with open(osp.join(out_dir, "cm_top1-class_order.txt"), "w") as f:
        for lbl in sorted(g_label_drug):
            f.write(f"{lbl}\t{g_label_drug[lbl]}\n")

    for rows in (fold_rows, per_drug_rows):
        for row in rows:
            row["batch"] = group

    return fold_rows, per_drug_rows


def main():
    args = parser.parse_args()
    emb_dir = args.embeddings_dir

    embeddings = np.load(osp.join(emb_dir, "embeddings.npy"))
    labels = np.load(osp.join(emb_dir, "labels.npy"))
    meta = pd.read_csv(osp.join(emb_dir, "metadata.csv"))
    assert len(meta) == len(labels) == len(embeddings), "metadata / embeddings mismatch"
    assert (meta["labels"].values == labels).all(), "metadata labels out of sync"

    dates = meta["date"].astype(str).values
    label_drug_dict = {
        int(l): meta.loc[meta["labels"] == l, "label_names"].iloc[0]
        for l in np.unique(labels)
    }
    print(f"Loaded {embeddings.shape} embeddings")
    print(f"Classes: {label_drug_dict}")
    print(f"Days: {sorted(np.unique(dates).tolist())}")

    crossval_dir = osp.join(emb_dir, "crossval_perday")
    os.makedirs(crossval_dir, exist_ok=True)

    all_fold_rows, all_per_drug_rows = [], []
    for group, days in DAY_GROUPS.items():
        days = [d for d in days if d in set(dates)]
        if not days:
            print(f"Skipping batch {group}: no matching days in the embeddings.")
            continue
        fold_rows, per_drug_rows = run_group(
            group, days, embeddings, labels, dates, label_drug_dict, args, crossval_dir
        )
        all_fold_rows.extend(fold_rows)
        all_per_drug_rows.extend(per_drug_rows)

    fold_df = pd.DataFrame(all_fold_rows)[
        ["batch", "test_date", "dist_metric", "k", "top_n", "n_correct", "n_total", "accuracy"]
    ]
    fold_csv = osp.join(crossval_dir, "knn_accuracy_per_fold.csv")
    fold_df.to_csv(fold_csv, index=False)

    per_drug_df = pd.DataFrame(all_per_drug_rows)[
        ["batch", "test_date", "k", "top_n", "drug", "label", "n_correct", "n_total", "accuracy"]
    ]
    per_drug_csv = osp.join(crossval_dir, "knn_accuracy_per_drug_per_fold.csv")
    per_drug_df.to_csv(per_drug_csv, index=False)

    mean_across = (
        fold_df.groupby(["batch", "dist_metric", "k", "top_n"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_folds"})
    )
    mean_csv = osp.join(crossval_dir, "knn_accuracy_mean_across_folds.csv")
    mean_across.to_csv(mean_csv, index=False)

    mean_per_drug = (
        per_drug_df.groupby(["batch", "top_n", "drug", "label"])
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
        mean_per_drug["n_correct_sum"] / mean_per_drug["n_total_sum"] * 100.0
    )
    mean_per_drug_csv = osp.join(crossval_dir, "knn_accuracy_per_drug_mean_across_folds.csv")
    mean_per_drug.to_csv(mean_per_drug_csv, index=False)

    print("\n================ Per-fold accuracy ================")
    print(fold_df.to_string(index=False))
    print("\n================ Mean across folds ================")
    print(mean_across.to_string(index=False))
    print("\n================ Per drug (mean across folds) ================")
    print(mean_per_drug.to_string(index=False))
    print(f"\nWrote results to {crossval_dir}")


if __name__ == "__main__":
    main()
