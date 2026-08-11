"""
Leave-one-drug-out k-NN evaluation of the MitoSpace-3D reproducibility embeddings,
predicting the *acquisition day* instead of the drug.

This is the transpose of `evaluate_3dms_reproducibility_crossday.py`: the folds are
stratified by drug (one drug held out at a time) and the label being classified is the
acquisition day. It measures how much day-of-acquisition (batch) signal the embedding
carries that generalizes to a drug the neighbour set never contains. Chance is 33.3%.

Same distance and voting rule as the cross-day script: similarity-weighted vote over the
k=100 nearest cosine neighbours, each dye batch evaluated independently.

Day counts are balanced *within each drug*, so both the held-out drug and the two-drug
training set carry an equal number of cells per day.

Usage:
    python evaluate_3dms_reproducibility_crossdrug.py --embeddings_dir 3dms_reproducibility
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from evaluate_3dms_reproducibility_crossday import (
    DAY_GROUPS,
    cosine_distance,
    nearest_neighbor_evaluation,
    plot_cm,
)

np.random.seed(0)

GROUP_TITLES = {
    "deepred": "MitoSpace-3D Top-1 kNN (k=100), day prediction - MitoTracker DeepRed",
    "mitobright": "MitoSpace-3D Top-1 kNN (k=100), day prediction - MitoBright",
}

parser = argparse.ArgumentParser(
    description="MitoSpace-3D leave-one-drug-out kNN, predicting acquisition day"
)
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
    help="Subsample the days within each drug to the smallest day count.",
)
parser.add_argument("--seed", type=int, default=1123)


def balance_days_within_drug(day_labels, drug_labels, seed=1123, shuffle=True):
    """Indices that give every (drug, day) cell the same count within each drug."""
    rng = np.random.default_rng(seed)
    keep = []
    for drug in np.unique(drug_labels):
        drug_idx = np.where(drug_labels == drug)[0]
        days, counts = np.unique(day_labels[drug_idx], return_counts=True)
        min_count = counts.min()
        for day in days:
            day_idx = drug_idx[day_labels[drug_idx] == day]
            keep.append(rng.choice(day_idx, size=min_count, replace=False))
    keep = np.concatenate(keep)
    if shuffle:
        rng.shuffle(keep)
    return keep


def run_fold(
    held_out_drug,
    embeddings,
    day_labels,
    drug_labels,
    day_name_dict,
    drug_name_dict,
    top_ns,
    k,
    title,
    out_dir,
):
    test_mask = drug_labels == held_out_drug
    train_mask = ~test_mask
    drug_name = drug_name_dict[int(held_out_drug)]

    train_embeddings, train_days = embeddings[train_mask], day_labels[train_mask]
    eval_embeddings, eval_days = embeddings[test_mask], day_labels[test_mask]

    train_drugs = sorted(
        drug_name_dict[int(d)] for d in np.unique(drug_labels[train_mask])
    )
    print(f"  Test:  {test_mask.sum()} cells (held-out drug={drug_name})")
    print(f"  Train: {train_mask.sum()} cells (drugs={train_drugs})")

    dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings, train_embeddings)
    preds, correct_preds_idxs = nearest_neighbor_evaluation(
        eval_days, train_days, top_ns, dist_matrix, dist_matrix_idxs, k=k
    )

    plot_cm(
        eval_days,
        preds[1],
        day_name_dict,
        show_values=True,
        title=f"{title}\nheld-out drug {drug_name}",
        normalize=True,
        save_path=osp.join(out_dir, f"cm_top1-heldout_drug={drug_name}.png"),
    )
    plot_cm(
        eval_days,
        preds[1],
        day_name_dict,
        show_values=True,
        title=f"{title}\nheld-out drug {drug_name}",
        normalize=False,
        save_path=osp.join(out_dir, f"cm_top1_counts-heldout_drug={drug_name}.png"),
    )

    fold_rows, per_day_rows = [], []
    for top_n in top_ns:
        n_correct = len(correct_preds_idxs[top_n])
        n_total = len(eval_days)
        fold_rows.append(
            {
                "heldout_drug": drug_name,
                "dist_metric": "cosine",
                "k": k,
                "top_n": top_n,
                "n_correct": int(n_correct),
                "n_total": int(n_total),
                "accuracy": n_correct * 100.0 / n_total,
            }
        )
        for day in np.unique(train_days):
            total = int(np.sum(eval_days == day))
            correct = int(np.sum(eval_days[correct_preds_idxs[top_n]] == day))
            per_day_rows.append(
                {
                    "heldout_drug": drug_name,
                    "k": k,
                    "top_n": top_n,
                    "date": day_name_dict[int(day)],
                    "n_correct": correct,
                    "n_total": total,
                    "accuracy": correct * 100.0 / total if total > 0 else 0.0,
                }
            )
        print(f"  Top-{top_n} accuracy: {n_correct * 100.0 / n_total:.2f}%")

    return fold_rows, per_day_rows, (np.asarray(eval_days), np.asarray(preds[1]))


def run_group(group, days, embeddings, labels, dates, drug_names, args, crossval_dir):
    print(f"\n########## Batch: {group} ({', '.join(days)}) ##########")
    out_dir = osp.join(crossval_dir, group)
    os.makedirs(out_dir, exist_ok=True)

    mask = np.isin(dates, days)
    g_emb, g_drug, g_date = embeddings[mask], labels[mask], dates[mask]

    # Days become the classification target; encode them as 0..n-1 in calendar order.
    day_order = sorted(np.unique(g_date).tolist())
    day_to_idx = {d: i for i, d in enumerate(day_order)}
    g_day = np.array([day_to_idx[d] for d in g_date], dtype=np.int32)
    day_name_dict = {i: d for d, i in day_to_idx.items()}
    drug_name_dict = {int(l): drug_names[int(l)] for l in np.unique(g_drug)}

    counts = pd.crosstab(
        pd.Series([drug_name_dict[int(d)] for d in g_drug], name="drug"),
        pd.Series(g_date, name="date"),
    )
    print(f"{mask.sum()} cells\n{counts.to_string()}")

    if args.balance_classes:
        keep = balance_days_within_drug(g_day, g_drug, seed=args.seed)
        g_emb, g_drug, g_day, g_date = g_emb[keep], g_drug[keep], g_day[keep], g_date[keep]
        counts = pd.crosstab(
            pd.Series([drug_name_dict[int(d)] for d in g_drug], name="drug"),
            pd.Series(g_date, name="date"),
        )
        print(f"Balanced to {len(g_day)} cells\n{counts.to_string()}")

    title = GROUP_TITLES.get(group, f"MitoSpace-3D day prediction (k={args.k}) - {group}")

    fold_rows, per_day_rows, top1_gt, top1_preds = [], [], [], []
    for drug in sorted(np.unique(g_drug)):
        print(f"\n===== Fold: held-out drug {drug_name_dict[int(drug)]} =====")
        f_rows, d_rows, (gt, pred) = run_fold(
            drug,
            g_emb,
            g_day,
            g_drug,
            day_name_dict,
            drug_name_dict,
            args.top_ns,
            args.k,
            title,
            out_dir,
        )
        fold_rows.extend(f_rows)
        per_day_rows.extend(d_rows)
        top1_gt.append(gt)
        top1_preds.append(pred)

    gt_concat = np.concatenate(top1_gt)
    preds_concat = np.concatenate(top1_preds)

    cm_counts = plot_cm(
        gt_concat,
        preds_concat,
        day_name_dict,
        show_values=True,
        title=f"{title}\n3-fold leave-one-drug-out (counts)",
        normalize=False,
        save_path=osp.join(out_dir, "cm_top1-sum_across_folds_counts.png"),
    )
    plot_cm(
        gt_concat,
        preds_concat,
        day_name_dict,
        show_values=True,
        title=f"{title}\n3-fold leave-one-drug-out",
        normalize=True,
        save_path=osp.join(out_dir, "cm_top1-mean_across_folds_normalized.png"),
    )
    np.save(osp.join(out_dir, "cm_top1-sum_across_folds.npy"), cm_counts)
    with open(osp.join(out_dir, "cm_top1-class_order.txt"), "w") as f:
        for idx in sorted(day_name_dict):
            f.write(f"{idx}\t{day_name_dict[idx]}\n")

    for rows in (fold_rows, per_day_rows):
        for row in rows:
            row["batch"] = group

    return fold_rows, per_day_rows


def main():
    args = parser.parse_args()
    emb_dir = args.embeddings_dir

    embeddings = np.load(osp.join(emb_dir, "embeddings.npy"))
    labels = np.load(osp.join(emb_dir, "labels.npy"))
    meta = pd.read_csv(osp.join(emb_dir, "metadata.csv"))
    assert len(meta) == len(labels) == len(embeddings), "metadata / embeddings mismatch"
    assert (meta["labels"].values == labels).all(), "metadata labels out of sync"

    dates = meta["date"].astype(str).values
    drug_names = {
        int(l): meta.loc[meta["labels"] == l, "label_names"].iloc[0]
        for l in np.unique(labels)
    }
    print(f"Loaded {embeddings.shape} embeddings")
    print(f"Drugs (fold strata): {drug_names}")
    print(f"Days (prediction target): {sorted(np.unique(dates).tolist())}")

    crossval_dir = osp.join(emb_dir, "crossval_perdrug")
    os.makedirs(crossval_dir, exist_ok=True)

    all_fold_rows, all_per_day_rows = [], []
    for group, days in DAY_GROUPS.items():
        days = [d for d in days if d in set(dates)]
        if not days:
            print(f"Skipping batch {group}: no matching days in the embeddings.")
            continue
        fold_rows, per_day_rows = run_group(
            group, days, embeddings, labels, dates, drug_names, args, crossval_dir
        )
        all_fold_rows.extend(fold_rows)
        all_per_day_rows.extend(per_day_rows)

    fold_df = pd.DataFrame(all_fold_rows)[
        ["batch", "heldout_drug", "dist_metric", "k", "top_n", "n_correct", "n_total", "accuracy"]
    ]
    fold_df.to_csv(osp.join(crossval_dir, "knn_accuracy_per_fold.csv"), index=False)

    per_day_df = pd.DataFrame(all_per_day_rows)[
        ["batch", "heldout_drug", "k", "top_n", "date", "n_correct", "n_total", "accuracy"]
    ]
    per_day_df.to_csv(
        osp.join(crossval_dir, "knn_accuracy_per_day_per_fold.csv"), index=False
    )

    mean_across = (
        fold_df.groupby(["batch", "dist_metric", "k", "top_n"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_folds"})
    )
    mean_across.to_csv(
        osp.join(crossval_dir, "knn_accuracy_mean_across_folds.csv"), index=False
    )

    mean_per_day = (
        per_day_df.groupby(["batch", "top_n", "date"])
        .agg(
            n_correct_sum=("n_correct", "sum"),
            n_total_sum=("n_total", "sum"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            n_folds=("heldout_drug", "nunique"),
        )
        .reset_index()
    )
    mean_per_day["accuracy_micro"] = (
        mean_per_day["n_correct_sum"] / mean_per_day["n_total_sum"] * 100.0
    )
    mean_per_day.to_csv(
        osp.join(crossval_dir, "knn_accuracy_per_day_mean_across_folds.csv"), index=False
    )

    print("\n================ Per-fold accuracy (chance = 33.3%) ================")
    print(fold_df.to_string(index=False))
    print("\n================ Mean across folds ================")
    print(mean_across.to_string(index=False))
    print("\n================ Per day (mean across folds) ================")
    print(mean_per_day.to_string(index=False))
    print(f"\nWrote results to {crossval_dir}")


if __name__ == "__main__":
    main()
