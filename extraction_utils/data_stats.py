import os
import numpy as np
from memory_profiler import profile
from tqdm import tqdm


def load_data(fpath):
    return np.load(fpath)


if __name__ == '__main__':
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    data_folder = "2023_data"
    data_dir = f"{proj_dir}/data/{data_folder}"

    drug_folders = os.listdir(f"{data_dir}/processed_data")
    chunk_size = 10  # number of samples to consider per drug for stats calculation

    # find data 99% data for all the data

    # make a dictionary to store the stats like this {0: freq, 10000: freq, 20000: freq, ...}
    stats = {}
    for pixel_vals in range(0, 65536, 5000):
        stats[pixel_vals] = 0

    for drug_folder in drug_folders:
        filenames = os.listdir(f"{data_dir}/processed_data/{drug_folder}")
        pbar = tqdm(filenames, desc=f"Processing {drug_folder}")
        for filename in filenames:
            fpath = f"{data_dir}/processed_data/{drug_folder}/{filename}"
            data = load_data(fpath)
            for pixel_vals in range(0, 65536, 5000):
                stats[pixel_vals] += (np.sum(np.logical_and(pixel_vals < data, data < pixel_vals + 5000)))

            pbar.update(1)

    # convert stats to percentage
    total_pixels = sum(list(stats.values()))
    stats = {k: v / total_pixels for k, v in stats.items()}

    print(stats)


