import numpy as np
import os
from tqdm import tqdm

data_root = "/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/"
drug_folders = sorted(os.listdir(os.path.join(data_root, "processed_data")))

# select_folders = ['20250128_Control']
# drug_folders = [folder for folder in drug_folders if folder in select_folders]

for drug_folder in drug_folders:
    filenames = sorted(os.listdir(os.path.join(data_root, "processed_data", drug_folder)))
    filenames = [os.path.join(data_root, "processed_data", drug_folder, file) for file in filenames]

    save_folder = os.path.join(data_root, "2D_MIP_slices", drug_folder)
    os.makedirs(save_folder, exist_ok=True)

    pbar = tqdm(total=len(filenames))
    for file in filenames:
        data = np.load(file)
        slice_2d_data = np.max(data[0], axis=1)

        # save the 2D slice data
        np.save(os.path.join(save_folder, os.path.basename(file)), slice_2d_data)
        pbar.update(1)



