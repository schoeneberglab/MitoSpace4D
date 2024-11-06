import os
import numpy as np

if __name__ == '__main__':
    fpath = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data/20240729/000021.npy'
    data = np.load(fpath)
    save_path = fpath.replace('2024_subdata', '2024_subdata_compressed')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path[:-4], data=data)
    print()