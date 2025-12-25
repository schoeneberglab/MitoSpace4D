import os
import os.path as osp
import h5py
import numpy as np
from tqdm import tqdm
import random
from random import shuffle

def make_h5_data(save_dir, num_samples, all_filenames, all_labels, split='train', start_idx=0, end_idx=0):
    assert num_samples == end_idx - start_idx, "Number of samples do not match the start and end indices."
    with h5py.File(osp.join(save_dir, f'{split}_data.h5'), 'w') as hdf5_file:
        # Create a dataset to store all samples
        dset = hdf5_file.create_dataset('data', (num_samples, *data_shape), dtype='uint16')
        dset_label = hdf5_file.create_dataset('labels', (num_samples, *label_shape), dtype='uint8')

        # Incrementally add data
        pbar = tqdm(total=num_samples)
        for i, filename in enumerate(all_filenames[start_idx: end_idx]):
            sample = np.load(filename)
            sample = sample.transpose(1, 0, 2, 3, 4)
            dset[i] = sample
            dset_label[i] = all_labels[i]
            pbar.update(1)

    print(f"Data saved to HDF5 for {split}.")

if __name__ == '__main__':
    root_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/dummy_data/processed_data'
    save_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/dummy_data'
    drug_label_path = '/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt'
    seed = 1123

    drug_labels = {}
    with open(drug_label_path, 'r') as f:
        drugs_to_labels = f.readlines()
        for line in drugs_to_labels:
            folder, drug, label = line.split()
            drug_labels[folder] = {'drug': drug, 'label': int(label)}
    
    drug_folders = sorted([file for file in os.listdir(root_dir)])

    all_filenames = []
    all_labels = []

    for drug_folder in drug_folders:
        filenames = sorted([file for file in os.listdir(osp.join(root_dir,drug_folder))])
        filenames = [osp.join(root_dir, drug_folder, file) for file in filenames]
        all_filenames.extend(filenames)
        all_labels.extend([drug_labels[drug_folder]['label']] * len(filenames))

    data = list(zip(all_filenames, all_labels))

    random.seed(seed)
    shuffle(data)

    all_filenames, all_labels = zip(*data)
    all_filenames, all_labels = (list(all_filenames), list(all_labels))

    len_all_data = round(len(all_labels) * 1.)

    train_split = round(len_all_data * 0.9)
    val_split = round(len_all_data * 0.1)

    num_train_samples = train_split
    num_val_samples = val_split

    data_shape = (2, 20, 60, 256, 256)  # Shape of your data
    label_shape = ()  # Shape of individual labels (e.g., scalar for classification)

    # Create an empty HDF5 file with the required dataset
    make_h5_data(save_dir, num_train_samples, all_filenames, all_labels, split='train', start_idx=0, end_idx=num_train_samples)
    make_h5_data(save_dir, num_val_samples, all_filenames, all_labels, split='val', start_idx=num_train_samples, end_idx=num_train_samples+num_val_samples)