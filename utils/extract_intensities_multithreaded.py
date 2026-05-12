import torch
import numpy as np
import os
import random
import yaml
import tqdm

import os.path as osp
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from concurrent.futures import ThreadPoolExecutor, as_completed

np.random.seed(0)
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.multiprocessing.set_sharing_strategy('file_system')

def get_intensities(img, mask_ch=1):
    # Morphology: Ch 1, TMRM: Ch 0; consistent with original channel ordering (tmrm first, morph second)
    # (T, C, Z, Y, Z)
    morph_intensities = []
    tmrm_intensities = []

    for t in range(img.shape[0]):
        # Otsu threshold the mask_ch get a mask
        thr = threshold_otsu(img[t, mask_ch, ...])
        mask = img[t, 1, ...] > thr

        img[t, 0, ...] = img[t, 0, ...] * mask
        img[t, 1, ...] = img[t, 1, ...] * mask

        tmrm_intensities.append(img[t, 0, ...].mean())
        morph_intensities.append(img[t, 1, ...].mean())

    return morph_intensities, tmrm_intensities


def process_image_pair(idx, morph_path, tmrm_path):
    """Worker function to load and process a single pair of images."""
    morph_img = np.load(morph_path)  # (T, Z, Y, X)
    tmrm_img = np.load(tmrm_path)  # (T, Z, Y, X)

    # Concatenate along new channel dimension to get (T, C, Z, Y, X)
    morph_img = np.expand_dims(morph_img, axis=1)  # (T, 1, Z, Y, X)
    tmrm_img = np.expand_dims(tmrm_img, axis=1)  # (T, 1, Z, Y, X)

    # (T, 2, Z, Y, X); consistent with original channel ordering (tmrm first, morph second)
    img = np.concatenate([tmrm_img, morph_img], axis=1)

    morph_intensities, tmrm_intensities = get_intensities(img, mask_ch=1)

    return idx, morph_intensities, tmrm_intensities

if __name__ == '__main__':

    max_threads = 8
    embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_testing'
    os.makedirs(embeddings_dir, exist_ok=True)

    img_pathfile = osp.join(embeddings_dir, 'image_paths.csv')

    # Set up image paths
    df = pd.read_csv(img_pathfile, header=None)
    df.columns = ['morph_path']

    # Keep only entries with paths containing "20240729" (for testing)
    print(len(df))
    df = df[df['morph_path'].str.contains("20240729")].reset_index(drop=True)
    print(len(df))

    # Add a new column "tmrm_path" by replacing the morph_path with the tmrm path
    df['tmrm_path'] = df['morph_path'].apply(lambda x: x.replace('-0-1.npy', '-0-0.npy'))

    # Pre-allocate lists to maintain the exact order of the DataFrame
    results_morph = [None] * len(df)
    results_tmrm = [None] * len(df)

    # Use ThreadPoolExecutor for concurrent I/O and processing
    # You can tweak max_workers; os.cpu_count() is usually a safe default
    print(f"Starting multithreaded processing with {max_threads} workers...")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks to the thread pool
        futures = {
            executor.submit(process_image_pair, i, row['morph_path'], row['tmrm_path']): i
            for i, row in df.iterrows()
        }

        # Process as they complete and update progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(df), desc="Extracting Intensities"):
            idx, morph_int, tmrm_int = future.result()
            results_morph[idx] = morph_int
            results_tmrm[idx] = tmrm_int

    # Assign the correctly ordered results back to the DataFrame
    df['morph_intensities'] = results_morph
    df['tmrm_intensities'] = results_tmrm

    # Save the intensities as numpy arrays
    morph_intensity_path = osp.join(embeddings_dir, 'morph_intensities.npy')
    tmrm_intensity_path = osp.join(embeddings_dir, 'tmrm_intensities.npy')

    morph_intensities_array = np.stack(df['morph_intensities'].values)
    tmrm_intensities_array = np.stack(df['tmrm_intensities'].values)

    np.save(morph_intensity_path, morph_intensities_array)
    np.save(tmrm_intensity_path, tmrm_intensities_array)

    # Save the dataframe with the paths and intensities as a parquet file
    df.to_parquet(osp.join(embeddings_dir, 'mean_intensities.parquet'))
    print("Processing complete and saved!")