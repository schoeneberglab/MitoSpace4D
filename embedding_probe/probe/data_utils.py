"""Load MS4D embeddings and drug labels from project parquets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_label_names(colors_path: str | Path) -> dict[int, str]:
    label_to_name: dict[int, str] = {}
    with open(colors_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            name = parts[1]
            label_id = int(parts[2])
            if label_id not in label_to_name:
                label_to_name[label_id] = name
    return label_to_name


def stack_embeddings(series: pd.Series) -> np.ndarray:
    return np.stack(series.apply(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())


def load_trained_bundle(parquet_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "embeddings" not in df.columns:
        raise ValueError(f"{parquet_path} missing 'embeddings' column")
    return df


def load_aligned_trained_random(
    trained_parquet: str | Path,
    random_parquet: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_trained, X_random, labels, merged_df) on shared cell_ids."""
    x_t, x_r, labels, merged = load_aligned_trained_random_df(trained_parquet, random_parquet)
    return x_t, x_r, labels, merged


def load_aligned_trained_random_df(
    trained_parquet: str | Path,
    random_parquet: str | Path,
    label_names_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    trained_cols = ["cell_id", "labels", "embeddings"]
    trained = pd.read_parquet(trained_parquet, columns=trained_cols)
    random = pd.read_parquet(random_parquet, columns=["cell_id", "embeddings"])
    trained["cell_id"] = trained["cell_id"].astype(str)
    random["cell_id"] = random["cell_id"].astype(str)
    merged = trained.merge(random, on="cell_id", suffixes=("_trained", "_random"))
    if merged.empty:
        raise ValueError("No overlapping cell_id between trained and random parquets")

    full_trained = pd.read_parquet(trained_parquet)
    full_trained["cell_id"] = full_trained["cell_id"].astype(str)
    out = full_trained[full_trained["cell_id"].isin(merged["cell_id"])].copy()
    out = out.set_index("cell_id").loc[merged["cell_id"].tolist()].reset_index()

    x_t = stack_embeddings(merged["embeddings_trained"])
    x_r = stack_embeddings(merged["embeddings_random"])
    labels = merged["labels"].to_numpy(dtype=np.int32)
    return x_t, x_r, labels, out


def pca_center_scale_fit_transform(
    x: np.ndarray,
    calibrator: np.ndarray,
    n_components: int | None = None,
) -> np.ndarray:
    """PCA + StandardScaler fit on calibrator rows, applied to x."""
    n_components = n_components or min(calibrator.shape[0] - 1, calibrator.shape[1])
    n_components = max(1, min(n_components, calibrator.shape[1], calibrator.shape[0] - 1))
    pca = PCA(n_components=n_components).fit(calibrator)
    scaler = StandardScaler().fit(pca.transform(calibrator))
    return scaler.transform(pca.transform(x)).astype(np.float32)


def pick_balanced_indices(
    labels: np.ndarray,
    n_classes: int,
    replicates_per_class: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    eligible = unique[counts >= replicates_per_class]
    if len(eligible) == 0:
        raise ValueError("No class has enough replicates")
    n_classes = min(n_classes, len(eligible))
    chosen = rng.choice(eligible, size=n_classes, replace=False)
    idxs: list[int] = []
    for lab in chosen:
        pool = np.where(labels == lab)[0]
        pick = rng.choice(pool, size=replicates_per_class, replace=False)
        idxs.extend(pick.tolist())
    return np.array(idxs, dtype=np.int64)


def mitotnt_mean_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_mean") and c not in ("embeddings",)]
