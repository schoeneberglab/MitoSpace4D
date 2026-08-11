"""
Verify the acquisition-folder -> drug assignment for the reproducibility dataset.

`metadata/drugs_to_labels.txt` assigns drugs by folder suffix and assumes the *same*
order on every imaging day (-1 control, -2 nocodazole, -3 oligomycin). If the imaging
order actually rotated between days, some folders would be mislabelled. Three independent
checks:

  1. cross-day consistency  — classify each folder's cells by k=100 weighted-cosine kNN
     against the other days of the same dye batch only (never the same day). A day whose
     order rotated would conflict with the other two days.
  2. anchor assignment      — score all six one-to-one mappings of {folder suffix} ->
     {drug} against 2024v3 drug centroids (separate 2024 acquisitions, same checkpoint)
     and check the assumed mapping is the argmax. This tests label *identity*, which
     check 1 cannot.
  3. model-free morphometry — threshold raw last frames and measure network
     fragmentation and spatial spread. Each suffix should show the same distinctive
     signature on all six days, with no model involved.

Usage:
    python verify_3dms_reproducibility_labels.py --embeddings_dir 3dms_reproducibility
"""

import argparse
import os
import os.path as osp
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu

DAY_GROUPS = {
    "deepred": ["20260330", "20260331", "20260401"],
    "mitobright": ["20260407", "20260408", "20260409"],
}
ASSUMED = {"1": "control", "2": "nocodazole", "3": "oligomycin"}
DRUGS = ["control", "oligomycin", "nocodazole"]
V3_NAMES = {0: "control", 9: "oligomycin", 17: "nocodazole"}

parser = argparse.ArgumentParser(description="Verify reproducibility drug labels")
parser.add_argument(
    "--embeddings_dir",
    default="/home/earkfeld/Projects/MitoSpace4D/3dms_reproducibility",
)
parser.add_argument(
    "--data_path",
    default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/reproducibility/processed_data",
)
parser.add_argument(
    "--anchor_dir",
    default="/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms3d_2024v3_225eps",
    help="2024v3 embeddings from the same checkpoint, used as the identity anchor.",
)
parser.add_argument("--k", type=int, default=100)
parser.add_argument(
    "--cells_per_folder", type=int, default=40, help="Cells sampled for morphometry."
)
parser.add_argument("--min_vox", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)


def weighted_knn_vote(query_emb, ref_emb, ref_names, k):
    """Fraction of query cells whose weighted k-NN vote lands on each reference class."""
    sims = query_emb @ ref_emb.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    votes = {}
    for i in range(sims.shape[0]):
        w = (sims[i, idx[i]] + 1) / 2
        tally = {}
        for name, wt in zip(ref_names[idx[i]], w):
            tally[name] = tally.get(name, 0.0) + wt
        top = max(tally, key=tally.get)
        votes[top] = votes.get(top, 0) + 1
    n = len(query_emb)
    return {name: cnt / n for name, cnt in votes.items()}


def check_cross_day(emb, meta, k):
    rows = []
    for batch, days in DAY_GROUPS.items():
        bmask = meta["date"].isin(days).values
        b_emb, b_meta = emb[bmask], meta[bmask].reset_index(drop=True)
        for folder in sorted(b_meta["folder"].unique()):
            day = folder.split("-")[0]
            q = (b_meta["folder"] == folder).values
            r = (b_meta["date"] != day).values
            frac = weighted_knn_vote(
                b_emb[q], b_emb[r], b_meta.loc[r, "label_names"].values, k
            )
            modal = max(frac, key=frac.get)
            assigned = b_meta.loc[q, "label_names"].iloc[0]
            rows.append(
                {
                    "batch": batch,
                    "folder": folder,
                    "assigned": assigned,
                    "modal_pred": modal,
                    "modal_frac": round(frac[modal], 3),
                    **{f"pred_{d}": round(frac.get(d, 0.0), 3) for d in DRUGS},
                    "n_cells": int(q.sum()),
                    "agrees": modal == assigned,
                }
            )
    return pd.DataFrame(rows)


def check_anchor_assignment(emb, meta, anchor_dir):
    v3_labels_all = np.load(osp.join(anchor_dir, "labels.npy"))
    keep = np.isin(v3_labels_all, list(V3_NAMES))
    v3_emb = np.asarray(
        np.load(osp.join(anchor_dir, "embeddings.npy"), mmap_mode="r")[keep][:, -1, :],
        dtype=np.float32,
    )
    v3_emb /= np.linalg.norm(v3_emb, axis=1, keepdims=True)
    v3_names = np.array([V3_NAMES[l] for l in v3_labels_all[keep]])

    v3_centroids = {}
    for d in DRUGS:
        c = v3_emb[v3_names == d].mean(0)
        v3_centroids[d] = c / np.linalg.norm(c)

    rows = []
    for batch, days in DAY_GROUPS.items():
        bm = meta["date"].isin(days).values
        suffixes = sorted(meta.loc[bm, "suffix"].unique())

        centroids = {}
        for s in suffixes:
            c = emb[bm & (meta["suffix"] == s).values].mean(0)
            centroids[s] = c / np.linalg.norm(c)

        score = {
            s: {d: float(centroids[s] @ v3_centroids[d]) for d in DRUGS} for s in suffixes
        }
        ranked = sorted(
            (
                (sum(score[s][m] for s, m in zip(suffixes, perm)), dict(zip(suffixes, perm)))
                for perm in permutations(DRUGS)
            ),
            reverse=True,
            key=lambda x: x[0],
        )
        for rank, (total, mapping) in enumerate(ranked, start=1):
            rows.append(
                {
                    "batch": batch,
                    "rank": rank,
                    "score": round(total, 4),
                    "mapping": ", ".join(f"-{s}={mapping[s]}" for s in suffixes),
                    "is_assumed": mapping == {s: ASSUMED[s] for s in suffixes},
                }
            )
    return pd.DataFrame(rows)


def check_morphometry(data_path, cells_per_folder, min_vox, seed):
    rng = np.random.default_rng(seed)
    rows = []
    folders = sorted(
        d for d in os.listdir(data_path) if osp.isdir(osp.join(data_path, d))
    )
    for folder in folders:
        suffix = folder.split("-")[1]
        if suffix not in ASSUMED:
            continue
        files = sorted(
            f for f in os.listdir(osp.join(data_path, folder)) if f.endswith(".npy")
        )
        pick = rng.choice(len(files), size=min(cells_per_folder, len(files)), replace=False)
        for i in pick:
            vol = np.asarray(
                np.load(osp.join(data_path, folder, files[i]), mmap_mode="r")[-1],
                dtype=np.float32,
            )
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
            mask = vol > threshold_otsu(vol)
            if mask.sum() < min_vox:
                continue

            lab, _ = ndimage.label(mask)
            sizes = np.bincount(lab.ravel())[1:]
            sizes = sizes[sizes >= min_vox]
            if sizes.size == 0:
                continue

            coords = np.argwhere(mask).astype(np.float32)
            spread = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(1).mean()))

            rows.append(
                {
                    "folder": folder,
                    "date": folder.split("-")[0],
                    "suffix": suffix,
                    "assigned": ASSUMED[suffix],
                    "n_fragments": int(sizes.size),
                    "mean_fragment_vox": float(sizes.mean()),
                    "largest_fragment_frac": float(sizes.max() / sizes.sum()),
                    "mask_frac": float(mask.mean()),
                    "spread_vox": spread,
                }
            )
    return pd.DataFrame(rows)


def summarize_morphometry(df):
    cols = ["spread_vox", "largest_fragment_frac", "n_fragments", "mean_fragment_vox"]
    med = df.groupby(["date", "suffix"])[cols].median().reset_index()
    rows = []
    for col in cols:
        piv = med.pivot(index="date", columns="suffix", values=col)
        hi, lo = piv.idxmax(axis=1), piv.idxmin(axis=1)
        rows.append(
            {
                "metric": col,
                "highest_folder": (
                    f"-{hi.iloc[0]} ({ASSUMED[hi.iloc[0]]})" if hi.nunique() == 1 else "varies"
                ),
                "highest_same_all_days": bool(hi.nunique() == 1),
                "lowest_folder": (
                    f"-{lo.iloc[0]} ({ASSUMED[lo.iloc[0]]})" if lo.nunique() == 1 else "varies"
                ),
                "lowest_same_all_days": bool(lo.nunique() == 1),
            }
        )
    return med, pd.DataFrame(rows)


def main():
    args = parser.parse_args()
    out_dir = osp.join(args.embeddings_dir, "label_checks")
    os.makedirs(out_dir, exist_ok=True)

    emb = np.load(osp.join(args.embeddings_dir, "embeddings.npy"))
    meta = pd.read_csv(osp.join(args.embeddings_dir, "metadata.csv"))
    meta["date"] = meta["date"].astype(str)
    meta["folder"] = [p.split("/")[-2] for p in meta["image_paths"]]
    meta["suffix"] = [f.split("-")[1] for f in meta["folder"]]

    print("=" * 78)
    print("CHECK 1 — cross-day consistency (each folder vs the other days of its batch)")
    print("=" * 78)
    cross = check_cross_day(emb, meta, args.k)
    print(cross.to_string(index=False))
    disagree = cross[~cross["agrees"]]
    print(
        f"\n  {cross['agrees'].sum()}/{len(cross)} folders agree with their assigned label."
    )
    print("  disagreements: none" if disagree.empty else disagree.to_string(index=False))
    cross.to_csv(osp.join(out_dir, "check1_cross_day_consistency.csv"), index=False)

    print("\n" + "=" * 78)
    print("CHECK 2 — best one-to-one mapping against the 2024v3 anchor")
    print("=" * 78)
    anchor = check_anchor_assignment(emb, meta, args.anchor_dir)
    print(anchor.to_string(index=False))
    for batch, g in anchor.groupby("batch"):
        best = g.sort_values("rank").iloc[0]
        print(
            f"\n  {batch}: best mapping "
            f"{'MATCHES' if best['is_assumed'] else 'DIFFERS FROM'} the assumed one "
            f"(score {best['score']} vs runner-up {g.sort_values('rank').iloc[1]['score']})"
        )
    anchor.to_csv(osp.join(out_dir, "check2_anchor_assignment.csv"), index=False)

    print("\n" + "=" * 78)
    print("CHECK 3 — model-free morphometry from the raw images")
    print("=" * 78)
    morph = check_morphometry(
        args.data_path, args.cells_per_folder, args.min_vox, args.seed
    )
    med, summary = summarize_morphometry(morph)
    print(f"n = {len(morph)} cells sampled\n")
    print("median per day x folder:")
    print(med.to_string(index=False))
    print("\nis the extreme folder on each metric the same on all six days?")
    print(summary.to_string(index=False))
    morph.to_csv(osp.join(out_dir, "check3_morphometrics_per_cell.csv"), index=False)
    med.to_csv(osp.join(out_dir, "check3_morphometrics_median.csv"), index=False)
    summary.to_csv(osp.join(out_dir, "check3_consistency_summary.csv"), index=False)

    print(f"\nWrote check outputs to {out_dir}")


if __name__ == "__main__":
    main()
