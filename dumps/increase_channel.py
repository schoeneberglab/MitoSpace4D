import os

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2023_data/processed_data/'
    dates = ['20231108', '20231109', '20231113', '20231114', '20231115','20231116']
    for date in dates:
        data_path = os.path.join(data_dir, date)

        pbar = tqdm(os.listdir(data_path))
        try:
            for file in os.listdir(data_path):
                fpath = os.path.join(data_path, file)
                data = np.load(fpath)
                if len(data.shape) == 5:
                    pbar.update(1)
                    continue
                data = data[None].repeat(2, axis=0)
                np.save(fpath, data)
                pbar.update(1)
        except:
            print()
