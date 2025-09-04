import os
import os.path as osp
from typing import Tuple, Dict, List, Optional

from matplotlib import colors as mpl_colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- 1) Build sample table ----------

def _parse_label_color_file(label_color_file: str) -> tuple[Dict[str, str], Dict[str, List[float]]]:
    """
    Parse a file with lines:
        <dataset> <drug_name> <label> <r> <g> <b>
    where r,g,b are either in [0,1] or [0,255].
    Returns:
        dataset_label_map: dataset -> label
        label_color_map:    label -> [r,g,b] in [0,1]
    """
    dataset_label_map: Dict[str, str] = {}
    label_color_map: Dict[str, List[float]] = {}

    with open(label_color_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            dataset, drug_name, label, r, g, b = parts
            dataset_label_map[dataset] = label

            rf, gf, bf = float(r), float(g), float(b)
            # Normalize to [0,1] if needed
            if rf > 1.0 or gf > 1.0 or bf > 1.0:
                rgb = [rf / 255.0, gf / 255.0, bf / 255.0]
            else:
                rgb = [rf, gf, bf]
            label_color_map[label] = rgb

    return dataset_label_map, label_color_map


def build_sample_table(
    img_path_file: str,
    cell_region_map: str = "",
    label_color_file: str = "/home/earkfeld/Projects/MitoSpace4D/extraction_utils/colors.txt",
    shade_min: float = 0.75,
    shade_max: float = 1.0,
) -> pd.DataFrame:
    """
    Build the samples table from an image-path list and assign region IDs.

    Expects:
      - img_path_file: text file with one image path per line (whitespace-separated OK)
      - cell_region_map: CSV with columns "Data Path", "Cell ID Start", "Region ID"
      - label_color_file: text file mapping dataset -> (label, base RGB)

    Returns:
      DataFrame with columns:
        ['fpath', 'dataset', 'filename', 'cell_id', 'cell_tid', 'region_id',
         'label', 'label_color', 'dataset_color']
    """
    dataset_label_map, label_color_map = _parse_label_color_file(label_color_file)

    df_samples = pd.read_csv(img_path_file, dtype=str, header=None, delimiter=r"\s+")
    df_samples.columns = ["fpath"]

    df_samples["dataset"] = df_samples["fpath"].apply(lambda x: x.split("/")[-2])
    df_samples["filename"]  = df_samples["fpath"].apply(lambda x: x.split("/")[-1].split(".")[0])
    df_samples["cell_id"]   = df_samples["filename"].apply(lambda x: int(x.split("-")[0]))
    df_samples["cell_tid"]  = df_samples["filename"].apply(lambda x: int(x.split("-")[1]))
    df_samples["region_id"] = -1
    df_samples["label"] = -1

    # Map dataset -> label (string) and label -> color
    df_samples["label"] = df_samples["dataset"].map(dataset_label_map)
    df_samples["label_color"] = df_samples["label"].map(label_color_map)

    # Initialize dataset_color column (RGB list per row)
    df_samples["dataset_color"] = None  # dtype object, will store [r,g,b] lists

    
    # -- Region assignment per dataset (vectorized)
    if cell_region_map:
        df_regions = pd.read_csv(cell_region_map)
        for cond in df_samples["dataset"].unique():
            mask = df_samples["dataset"] == cond
            if not mask.any():
                continue

            dr = df_regions[df_regions["Data Path"] == cond].copy()
            if dr.empty:
                continue

            dr = dr.sort_values("Cell ID Start")
            starts = dr["Cell ID Start"].to_numpy(dtype=np.int64)
            region_ids = dr["Region ID"].to_numpy()

            cell_ids = df_samples.loc[mask, "cell_id"].to_numpy(dtype=np.int64)
            pos = np.searchsorted(starts, cell_ids, side="right") - 1
            assigned = np.where(pos >= 0, region_ids[pos], -1)
            df_samples.loc[mask, "region_id"] = assigned.astype(int)

    # -- Assign shades per-dataset within each label
    # For each label, sort its datasets alphabetically and assign scalars in [shade_min, shade_max]
    for label in df_samples["label"].dropna().unique():
        mask_label = df_samples["label"] == label
        if not mask_label.any():
            continue

        # Collect datasets with this label
        datasets = np.sort(df_samples.loc[mask_label, "dataset"].unique())
        n = len(datasets)
        base_rgb = np.array(df_samples.loc[mask_label, "label_color"].dropna().values[0], dtype=float)

        if n == 1:
            scalars = np.array([1.0])
        else:
            scalars = np.linspace(shade_min, shade_max, n)

        # Build dataset -> shaded color map
        dataset_to_color: Dict[str, List[float]] = {
            dataset: (base_rgb * s).clip(0.0, 1.0).tolist() for dataset, s in zip(datasets, scalars)
        }

        # Apply to all rows for those datasets
        df_samples.loc[mask_label, "dataset_color"] = df_samples.loc[mask_label, "dataset"].map(dataset_to_color)

    # Fill any missing label/dataset colors with a neutral placeholder
    # df_samples["label"].fillna("-1", inplace=True)
    df_samples["label_color"] = df_samples["label_color"].apply(
        lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else [0.5, 0.5, 0.5]
    )
    df_samples["dataset_color"] = df_samples["dataset_color"].apply(
        lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else [0.5, 0.5, 0.5]
    )

    return df_samples


# ---------- Normalization helper ----------

def _safe_norm(vmin: int, vmax: int) -> plt.Normalize:
    """Avoid degenerate vmin==vmax cases for Normalize."""
    if vmin == vmax:
        return plt.Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
    return plt.Normalize(vmin=vmin, vmax=vmax)


# ---------- 2) Region colormap (separate routine) ----------

def get_region_colors(
    df_samples: pd.DataFrame,
    cmap: str = "viridis",
    n_frames: int = 20,
    single_frames: bool = False,
) -> np.ndarray:
    """
    Region colormap: color depends only on region_id.

    If single_frames is True, each sample contributes n_frames repeated colors.
    Otherwise, one color per sample.

    Returns:
      np.ndarray of shape (N, 3), dtype float32
    """
    rmin = int(df_samples["region_id"].min())
    rmax = int(df_samples["region_id"].max())

    cmap_fn = plt.get_cmap(cmap)
    region_norm = _safe_norm(rmin, rmax)

    out: List[List[float]] = []
    for _, row in df_samples.iterrows():
        rid = int(row["region_id"])
        rgb = list(cmap_fn(region_norm(rid))[:3])
        if single_frames:
            out.extend([rgb] * n_frames)
        else:
            out.append(rgb)

    return np.asarray(out, dtype=np.float32)


def save_region_colormap(
    df_samples: pd.DataFrame,
    embedding_dir: str,
    filename: str = "cmap_region.npy",
    cmap: str = "viridis",
    n_frames: int = 20,
    single_frames: bool = False,
) -> np.ndarray:
    os.makedirs(embedding_dir, exist_ok=True)
    colors_arr = get_region_colors(df_samples, cmap=cmap, n_frames=n_frames, single_frames=single_frames)
    np.save(osp.join(embedding_dir, filename), colors_arr)
    return colors_arr


# ---------- 3) Temporal colormap (separate routine) ----------

def get_temporal_colors(
    df_samples: pd.DataFrame,
    cmap: str = "viridis",
    n_frames: int = 20,
    single_frames: bool = False,
    region_scale: int = 3,
) -> np.ndarray:
    """
    Temporal colormap: color depends on a synthetic time index:
        time_id = (region_scale * region_id * n_frames) + (cell_tid * n_frames) + frame_id

    If single_frames is True, each sample contributes n_frames colors (varying by frame).
    Otherwise, one color per sample with frame_id = 0.

    Returns:
      np.ndarray of shape (N, 3), dtype float32
    """
    def time_id(region_id: int, cell_tid: int, frame_id: int) -> int:
        return (region_scale * region_id * n_frames) + (cell_tid * n_frames) + frame_id

    rmax = int(df_samples["region_id"].max())
    tid_max = int(df_samples["cell_tid"].max())
    tmin = 0
    tmax = time_id(rmax, tid_max, n_frames - 1)

    cmap_fn = plt.get_cmap(cmap)
    temporal_norm = _safe_norm(tmin, tmax)

    out: List[List[float]] = []
    for _, row in df_samples.iterrows():
        rid = int(row["region_id"])
        tid = int(row["cell_tid"])

        if single_frames:
            for f in range(n_frames):
                out.append(list(cmap_fn(temporal_norm(time_id(rid, tid, f)))[:3]))
        else:
            out.append(list(cmap_fn(temporal_norm(time_id(rid, tid, 0)))[:3]))

    return np.asarray(out, dtype=np.float32)


def save_temporal_colormap(
    df_samples: pd.DataFrame,
    embedding_dir: str,
    filename: str = "cmap_temporal.npy",
    cmap: str = "viridis",
    n_frames: int = 20,
    single_frames: bool = False,
    region_scale: int = 3,
) -> np.ndarray:
    os.makedirs(embedding_dir, exist_ok=True)
    colors_arr = get_temporal_colors(
        df_samples,
        cmap=cmap,
        n_frames=n_frames,
        single_frames=single_frames,
        region_scale=region_scale,
    )
    np.save(osp.join(embedding_dir, filename), colors_arr)
    return colors_arr


# ---------- 4) Dataset colormap (separate routine) ----------

def get_dataset_colors(
    df_samples: pd.DataFrame,
    n_frames: int = 20,
    single_frames: bool = False,
) -> np.ndarray:
    """
    Dataset colormap: uses the per-row 'dataset_color' computed in build_sample_table.
    Colors vary by dataset (shaded within a label group).

    If single_frames is True, each sample contributes n_frames repeated colors.
    Otherwise, one color per sample.

    Returns:
      np.ndarray of shape (N, 3), dtype float32
    """
    out: List[List[float]] = []
    for _, row in df_samples.iterrows():
        rgb = row["dataset_color"]
        if not isinstance(rgb, (list, tuple, np.ndarray)) or len(rgb) != 3:
            rgb = [0.5, 0.5, 0.5]  # fallback

        if single_frames:
            out.extend([list(rgb)] * n_frames)
        else:
            out.append(list(rgb))
    return np.asarray(out, dtype=np.float32)


def save_dataset_colormap(
    df_samples: pd.DataFrame,
    embedding_dir: str,
    n_frames: int = 20,
    single_frames: bool = False,
    filename: str = "cmap_dataset.npy",
) -> np.ndarray:
    assert osp.exists(embedding_dir), f"Embedding directory {embedding_dir} does not exist."
    colors_arr = get_dataset_colors(df_samples, n_frames=n_frames, single_frames=single_frames)
    np.save(osp.join(embedding_dir, filename), colors_arr)
    return colors_arr


# ---------- Optional: one-shot wrapper (backward compatible) ----------

def create_colormap(
    img_path_file: str,
    cell_region_map: str,
    embedding_dir: str = "",
    cmap: str = "viridis",
    n_frames: int = 20,
    single_frames: bool = False,
    region_scale: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1) Build sample table + region IDs.
    2) Compute & save region and temporal colormaps separately.
    Returns:
      (region_colors, temporal_colors)
    """
    df_samples = build_sample_table(img_path_file, cell_region_map)
    region_colors = save_region_colormap(
        df_samples, embedding_dir, "cmap_region.npy", cmap, n_frames, single_frames
    )
    temporal_colors = save_temporal_colormap(
        df_samples, embedding_dir, "cmap_temporal.npy", cmap, n_frames, single_frames, region_scale
    )
    return region_colors, temporal_colors


if __name__ == "__main__":
    embeddings_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_20250828"
    df = build_sample_table(img_path_file=osp.join(embeddings_dir, "image_paths.csv"), 
                            cell_region_map=None)

    # Separate routines
    # region_colors   = save_region_colormap(df, "/path/to/embeddings", cmap="viridis", n_frames=20, single_frames=False)
    # temporal_colors = save_temporal_colormap(df, "/path/to/embeddings", cmap="viridis", n_frames=20, single_frames=False)

    # Dataset map
    dataset_colors = save_dataset_colormap(df, embeddings_dir, n_frames=20, single_frames=False)

    # Or the wrapper (backward compatible)
    # create_colormap("img_paths.txt", "cell_region_map.csv", "/path/to/embeddings", "viridis", 20, False)
