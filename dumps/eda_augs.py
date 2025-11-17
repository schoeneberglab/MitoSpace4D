import os
import random
import numpy as np
from tqdm import tqdm

def compute_mean_std(folder, n=50, seed=42, recurse=False):
    """Compute mean, std, min, and max of intensities from random .npy files in a folder (no subdirs)."""
    # List all .npy files in the folder (no recursion)
    if recurse:
        files = [os.path.join(root, f) for root, dirs, files in os.walk(folder) for f in files if f.endswith(".npy")]
    else:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        raise SystemExit("No .npy files found in the specified folder.")

    random.seed(seed)
    sample_files = random.sample(files, min(n, len(files)))

    sums = sumsqs = counts = None
    mins = maxs = None
    C_ref = None

    for path in tqdm(sample_files):
        arr = np.load(path, mmap_mode="r")  # expected shape: (C, T, D, H, W)
        if arr.ndim != 5:
            raise ValueError(f"{path} has shape {arr.shape}, expected (C,T,D,H,W).")
        C, T, D, H, W = arr.shape
        if C_ref is None:
            C_ref = C
            sums = np.zeros(C, dtype=np.float64)
            sumsqs = np.zeros(C, dtype=np.float64)
            counts = np.zeros(C, dtype=np.int64)
            mins = np.full(C, np.inf, dtype=np.float64)
            maxs = np.full(C, -np.inf, dtype=np.float64)
        elif C != C_ref:
            raise ValueError(f"Channel mismatch: saw C={C} in {path}, expected {C_ref}.")

        x = arr.reshape(C, -1).astype(np.float64, copy=False)
        sums += x.sum(axis=1)
        sumsqs += (x * x).sum(axis=1)
        counts += x.shape[1]
        mins = np.minimum(mins, x.min(axis=1))
        maxs = np.maximum(maxs, x.max(axis=1))

    means = sums / counts
    vars_ = np.clip((sumsqs / counts) - means**2, 0.0, None)
    stds = np.sqrt(vars_)

    print(f"# Sampled {len(sample_files)} of {len(files)} total files")
    print(f"channels: {C_ref}")
    print("mean:", "[" + ", ".join(f"{m:.6f}" for m in means) + "]")
    print("std:  ", "[" + ", ".join(f"{s:.6f}" for s in stds) + "]")
    print("min:  ", "[" + ", ".join(f"{mn:.6f}" for mn in mins) + "]")
    print("max:  ", "[" + ", ".join(f"{mx:.6f}" for mx in maxs) + "]")

    return dict(channels=C_ref, mean=means, std=stds, min=mins, max=maxs, n_sampled=len(sample_files))

if __name__ == "__main__":
    # folder = "/work/nvme/begq/MitoSpace4D/data/2025_data/20250807-2/"  # mitoq
    folder = "/work/nvme/begq/MitoSpace4D/data/2025_data/20250807-1/" # Control
    n = 100
    seed = 1123
    params = compute_mean_std(folder, n=n, seed=seed, recurse=False)