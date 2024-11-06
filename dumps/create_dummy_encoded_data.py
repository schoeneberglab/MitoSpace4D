import os
import random

import numpy as np
from tqdm import tqdm

import torch
import os.path as osp

if __name__ == '__main__':
    data_dir = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/2024_data/processed_data'
    enc_data_dir = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/2024_data/dummy_encoded_data'
    os.makedirs(enc_data_dir, exist_ok=True)

    drug_folders = os.listdir(data_dir)

    for drug in drug_folders:
        print(f"Processing {drug} ...")
        drug_folder_path = os.path.join(data_dir, drug)
        enc_drug_folder_path = os.path.join(enc_data_dir, drug)
        os.makedirs(enc_drug_folder_path, exist_ok=True)
        pbar = tqdm(os.listdir(drug_folder_path))

        # create dummy encoded data file equivalent to the original data
        for file in os.listdir(drug_folder_path):
            enc = np.random.randn(20, 4, 15, 64, 64).astype(np.float16)
            np.save(osp.join(enc_drug_folder_path, file), enc)
            pbar.update(1)
