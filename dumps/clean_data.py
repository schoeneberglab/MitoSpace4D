import os

import torch
import torch.nn.functional as F
import numpy as np

data_root = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data'

if __name__ == '__main__':
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('.npy'):
                fpath = os.path.join(root, f)

                try:
                    data = np.load(fpath, mmap_mode='r')
                except:
                    print(f"Error loading {fpath}")
                    continue

                if data.shape != (20, 2, 60, 256, 256):
                    print(f"{fpath}")
                    continue