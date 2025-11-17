import os
import random
import numpy as np
from tqdm import tqdm

def compute_gaussian_noise_params(root, alpha=0.10, n=50, seed=42, print_psnr_hint=False):
    """Compute mu and scale for Gaussian noise augmentation from random .npy files.

    Args:
        root (str): Root directory containing subfolders with .npy files.
        alpha (float): Scale factor for noise std: scale = alpha * per-channel std.
        n (int): Number of random .npy samples to use.
        seed (int): Random seed for reproducibility.
        print_psnr_hint (bool): Whether to print PSNR-based noise hints for [0,1] data.

    Returns:
        dict: Dictionary containing per-channel mean, std, mu, scale, and metadata.
    """
    def iter_npy_paths(root):
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".npy"):
                    paths.append(os.path.join(dirpath, f))
        return paths

    npy_paths = iter_npy_paths(root)
    if not npy_paths:
        raise SystemExit("No .npy files found under the specified root directory.")

    random.seed(seed)
    print(f"Found {len(npy_paths)} samples")
    sample_paths = random.sample(npy_paths, min(n, len(npy_paths)))

    sums = sumsqs = counts = None
    C_ref = None

    for path in tqdm(sample_paths):
        arr = np.load(path, mmap_mode="r")  # expected shape: (C, T, D, W, H)
        if arr.ndim != 5:
            raise ValueError(f"{path} has shape {arr.shape}, expected (C,T,D,W,H).")
        C, T, D, W, H = arr.shape
        if C_ref is None:
            C_ref = C
            sums = np.zeros(C, dtype=np.float64)
            sumsqs = np.zeros(C, dtype=np.float64)
            counts = np.zeros(C, dtype=np.int64)
        elif C != C_ref:
            raise ValueError(f"Channel mismatch: saw C={C} in {path}, expected {C_ref}.")

        x = arr.reshape(C, -1).astype(np.float64, copy=False)
        sums += x.sum(axis=1)
        sumsqs += (x * x).sum(axis=1)
        counts += x.shape[1]

    means = sums / counts
    vars_ = np.clip((sumsqs / counts) - means**2, 0.0, None)
    stds = np.sqrt(vars_)
    mu = np.zeros_like(means)
    scale = alpha * stds

    print(f"# Sampled {len(sample_paths)} of {len(npy_paths)} total files")
    print(f"channels: {C_ref}")
    print("data_mean:", "[" + ", ".join(f"{m:.6f}" for m in means) + "]")
    print("data_std: ", "[" + ", ".join(f"{s:.6f}" for s in stds) + "]")
    print()
    print("# Gaussian noise parameters for augmentation")
    print("mu:    ", "[" + ", ".join(f"{m:.6f}" for m in mu) + "]  # zero-mean noise")
    print(f"alpha:  {alpha:.6f}  # scale = alpha * data_std")
    print("scale: ", "[" + ", ".join(f"{s:.6f}" for s in scale) + "]")

    if print_psnr_hint:
        for psnr in (35, 32, 30, 28):
            sigma = 10 ** (-psnr / 20.0)
            print(f"# PSNR {psnr} dB → sigma ≈ {sigma:.6f} (channel-independent)")

    return dict(
        channels=C_ref,
        mean=means,
        std=stds,
        mu=mu,
        scale=scale,
        n_sampled=len(sample_paths)
    )

if __name__ == "__main__":
    root = "/work/nvme/begq/MitoSpace4D/data/2025_data"
    alpha = 0.10      # scale factor relative to per-channel std
    n = 1000           # number of random .npy files to sample
    seed = 1123         # reproducible sampling
    print_psnr_hint = True

    params = compute_gaussian_noise_params(
        root=root,
        alpha=alpha,
        n=n,
        seed=seed,
        print_psnr_hint=print_psnr_hint
    )