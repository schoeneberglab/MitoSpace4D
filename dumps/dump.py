import os

import numpy as np

if __name__ == '__main__':
    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2023_data/processed_data'
    for drug in os.listdir(data_dir):
        print(drug)
        for file in os.listdir(os.path.join(data_dir, drug)):
            if file.endswith('.npy'):
                data = np.load(os.path.join(data_dir, drug, file))
                if data.shape[0] > 20:
                    sliced_data = data[:20, 20:80]
                    np.save(os.path.join(data_dir, drug, file), sliced_data)