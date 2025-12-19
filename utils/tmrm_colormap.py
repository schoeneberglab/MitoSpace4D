import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import threshold_otsu

# def create_tmrm_colormap(embedding_dir: str):
#     """Creates a color map from mean TMRM intensities (Otsu-thresholded) and saves it as .npy."""

#     img_pathfile = osp.join(embedding_dir, 'image_paths.csv')
#     img_paths = np.loadtxt(img_pathfile, dtype=str, delimiter=',')
#     img_paths = np.array([path.strip() for path in img_paths])
#     print("Found image paths for TMRM colormap:", img_paths.shape)

#     tmrm_intensities = np.zeros(len(img_paths), dtype=np.float32)

#     for i, img_path in enumerate(tqdm(img_paths)):
#         # load the image (assumed npy file)
#         img = np.load(img_path, mmap_mode='r')  # shape assumed (C, H, W)
#         channel0 = img[0, ...]

#         # Apply Otsu threshold to remove background
#         thr = threshold_otsu(channel0)
#         mask = channel0 > thr

#         # Handle case where mask is empty (rare)
#         if np.any(mask):
#             tmrm_intensity = channel0[mask].mean()
#         else:
#             tmrm_intensity = channel0.mean()  # fallback

#         tmrm_intensities[i] = tmrm_intensity

#     # Normalize intensities to [0, 1]
#     min_intensity = np.min(tmrm_intensities)
#     max_intensity = np.max(tmrm_intensities)
#     normalized_intensities = (tmrm_intensities - min_intensity) / (max_intensity - min_intensity + 1e-9)

#     np.save(osp.join(embedding_dir, "cmap_tmrm.npy"), normalized_intensities)
#     print(f"Saved TMRM colormap with shape {normalized_intensities.shape} to {osp.join(embedding_dir, 'cmap_tmrm.npy')}")

def create_tmrm_colormap(embedding_dir: str):
    """Creates a color map from mean TMRM intensities (Otsu-thresholded),
    normalized by the number of voxels, and saves it as .npy."""

    img_pathfile = osp.join(embedding_dir, 'image_paths.csv')
    img_paths = np.loadtxt(img_pathfile, dtype=str, delimiter=',')
    img_paths = np.array([path.strip() for path in img_paths])
    print("Found image paths for TMRM colormap:", img_paths.shape)

    tmrm_intensities = np.zeros(len(img_paths), dtype=np.float32)

    for i, img_path in enumerate(tqdm(img_paths)):
        # load the image (assumed npy file)
        img = np.load(img_path, mmap_mode='r')  # shape assumed (C, H, W)
        channel0 = img[0, ...]

        # Apply Otsu threshold to remove background
        thr = threshold_otsu(channel0)
        mask = channel0 > thr

        # Number of voxels in the mask
        voxel_count = mask.sum()

        if voxel_count > 0:
            # Sum of intensities above threshold divided by number of voxels
            tmrm_intensity = channel0[mask].sum() / voxel_count
        else:
            # Fallback to global mean if the mask is empty
            tmrm_intensity = channel0.mean()

        tmrm_intensities[i] = tmrm_intensity

    # Normalize intensities to [0, 1]
    min_intensity = np.min(tmrm_intensities)
    max_intensity = np.max(tmrm_intensities)
    normalized_intensities = (tmrm_intensities - min_intensity) / (max_intensity - min_intensity + 1e-9)

    # Save
    out_path = osp.join(embedding_dir, "cmap_tmrm.npy")
    np.save(out_path, normalized_intensities)
    print(f"Saved TMRM colormap with shape {normalized_intensities.shape} to {out_path}")


if __name__ == "__main__":
    # Example usage
    embedding_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_2024v2_decoupled-tmrm_eps145_r20251118"
    create_tmrm_colormap(embedding_dir)
