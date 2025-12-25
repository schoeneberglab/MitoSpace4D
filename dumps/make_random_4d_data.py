import os
import numpy as np

if __name__ == '__main__':
    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_data'
    drug_label_info = '/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt'

    drug_folders = []
    with open(drug_label_info, 'r') as f:
        drug_labels = f.readlines()
        for line in drug_labels:
            folder, drug, label = line.split()
            drug_folders.append(folder)

    for drug_folder in drug_folders:
        os.makedirs(os.path.join(data_dir, drug_folder), exist_ok=True)

        # create 200 npy files for each folder of shape (20, 2, 60, 256, 256)
        for i in range(200):
            data = np.random.rand(20, 2, 60, 256, 256)*20000
            data = data.astype(np.uint16)
            np.save(os.path.join(data_dir, drug_folder, str(i).zfill(6)+'.npy'), data)

