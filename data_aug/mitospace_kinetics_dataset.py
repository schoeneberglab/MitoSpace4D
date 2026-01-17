import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import os
import random
from random import shuffle
from utils.utils import minus_one_to_one_normalization, agressive_sigmoid, idxs_to_keep, normalize
from torchvision.transforms import Compose
from typing import List, Dict, Union
from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl
import time
import torch


class MitoSpaceDataModule(pl.LightningDataModule):
    def __init__(self, train_datasets: List[Dataset],
                 val_datasets: List[Dataset],
                 batch_size: int,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 prefetch_factor: int = 2) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self):
        # Return dataloader for training
        return DataLoader(ConcatDataset(self.train_datasets), batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last,
                          prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        # Return dataloader for validation
        return DataLoader(ConcatDataset(self.val_datasets), batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last,
                          prefetch_factor=self.prefetch_factor)


class MitoSpaceDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 flag: str = 'train',
                 seed: int = None,
                 pick_labels: List = None,
                 samples_per_drug: int = None,
                 timesteps=None,
                 zstacks=None) -> None:

        self.root_dir = root_dir
        self.timesteps = timesteps
        self.zstacks = zstacks

        self.seed = 1123 if seed is None else seed

        print(f'Loading {flag} Dataset with split seed = {self.seed} ...')

        drug_labels = {}
        with open('/home/earkfeld/Projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt', 'r') as f:
            drugs_to_labels = f.readlines()
            for line in drugs_to_labels:
                folder, drug, label = line.split()
                drug_labels[folder] = {'drug': drug, 'label': int(label)}

        # -- encoded data dir
        # drug_folders = sorted([file for file in os.listdir(osp.join(self.root_dir, 'encoded_data'))])

        # -- All Dirs
        drug_folders = sorted([file for file in os.listdir(self.root_dir) if osp.isdir(osp.join(self.root_dir, file))])

        # -- Kinetic dirs (v1)
        # drug_folders = ['20250722-2', '20250724-1', '20250724-2', '20250725-1', '20250728-1', '20250804-1', '20250804-2', '20250805-1', '20250805-2', '20250806-2', '20250807-1', '20250807-2', '20250813-1', '20250813-2', '20250814-1', '20250814-2']

        # -- Cancer dirs
        # drug_folders = ["20250811-1", "20250811-2", "20250812-1", "20250828-1", "20250828-2", "20250828-3"]
        # drug_folders = ["20250811-1", "20250811-2", "20250812-1"]

        # --Kinetic control dir
        # drug_folders = ["20250807-1"]

        self.all_filenames = []
        self.all_labels = []

        for drug_folder in drug_folders:
            # filenames = sorted([file for file in os.listdir(osp.join(self.root_dir, 'encoded_data', drug_folder)) if osp.isfile(osp.join(self.root_dir, 'encoded_data', drug_folder, file))])
            # filenames = [osp.join(self.root_dir, 'encoded_data', drug_folder, file) for file in filenames]
            filenames = sorted([file for file in os.listdir(osp.join(self.root_dir, drug_folder)) if
                                osp.isfile(osp.join(self.root_dir, drug_folder, file))])
            filenames = [osp.join(self.root_dir, drug_folder, file) for file in filenames]

            if samples_per_drug != 'None' and samples_per_drug is not None:
                print(f"Limiting the number of samples per drug to {samples_per_drug}")
                filenames = filenames[samples_per_drug: samples_per_drug * 2]

            self.all_filenames.extend(filenames)
            self.all_labels.extend([drug_labels[drug_folder]['label']] * len(filenames))

        # Get indices of all sample files which contain "-0.npy" in the filename
        print("Using 60 frames/samples...")
        idxs = [i for i, filename in enumerate(self.all_filenames) if "-0.npy" in filename]
        self.all_filenames = [self.all_filenames[i].removesuffix("-0.npy") for i in idxs]  # remove the "-0.npy" suffix
        self.all_labels = [self.all_labels[i] for i in idxs]

        self.data = list(zip(self.all_filenames, self.all_labels))

        random.seed(self.seed)
        shuffle(self.data)

        self.all_filenames, self.all_labels = zip(*self.data)
        self.all_filenames, self.all_labels = (list(self.all_filenames), list(self.all_labels))

        self.len_all_data = round(len(self.all_labels) * 1.)

        self.train_split = round(self.len_all_data * 0.9)
        self.val_split = round(self.len_all_data * 0.1)
        self.test_split = self.len_all_data - self.train_split - self.val_split

        if flag == "all":
            self.filenames = self.all_filenames
            self.labels = self.all_labels

        elif flag == "train":
            self.filenames = self.all_filenames[:self.train_split]
            self.labels = self.all_labels[:self.train_split]

        elif flag == "val":
            self.filenames = self.all_filenames[self.train_split:self.train_split + self.val_split]
            self.labels = self.all_labels[self.train_split:self.train_split + self.val_split]

        elif flag == "test":
            self.filenames = self.all_filenames[self.train_split + self.val_split:]
            self.labels = self.all_labels[self.train_split + self.val_split:]

        else:
            raise ValueError("Invalid flag")

        if pick_labels is not None:
            # pick only the labels that are in the pick_labels list
            print(f"Filtering the labels to {pick_labels}")
            idxs_to_keep = [i for i, lbl in enumerate(self.labels) if lbl in pick_labels]
            self.filenames = [self.filenames[i] for i in idxs_to_keep]
            self.labels = [self.labels[i] for i in idxs_to_keep]

        print('Loading {} labels, found {} samples ...'.format(len(np.unique(self.labels)), len(self.filenames)))

        # print the number of samples per class
        for lbl in np.unique(self.labels):
            print(f"Class {lbl} has {self.labels.count(lbl)} samples")

    @staticmethod
    def balance_dataset(labels: List[int], samples_per_drug: int) -> List[int]:
        labels_to_idxs = {}
        for i, lbl in enumerate(labels):
            if lbl not in labels_to_idxs:
                labels_to_idxs[lbl] = [i]
            else:
                labels_to_idxs[lbl].append(i)

        # select samples_per_drug samples per drug randomly
        new_idxs = []
        for lbl, idxs in labels_to_idxs.items():
            new_idxs.extend(random.sample(idxs, samples_per_drug))

        return new_idxs

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img_name = self.filenames[idx]

        # Load all time windows for this sample (-0.npy, -1.npy, and -2.npy) and stack along the time axis
        image = np.concatenate([np.load(f"{img_name}-{t}.npy", mmap_mode='r').astype(np.float32) for t in [0, 1, 2]], axis=0)
        print(img_name)
        print(image.shape)

        # image = np.load(img_name, mmap_mode='r').astype(np.float32)

        # normalize if loading the processed_data (not encoded).
        # don't normalize if loading the encoded data (because then its already normalized)
        # image[:, 0] = np.clip(image[:, 0], 0, 25000)/25000.
        # image[:, 1] = np.clip(image[:, 1], 0, 10000)/10000.

        label = self.labels[idx]

        # -- normalize if loading the processed_data (not encoded).
        # don't normalize if loading the encoded data (because then its already normalized)
        # image[:, 0] = np.clip(image[:, 0], 0, 25000)/25000.
        # image[:, 1] = np.clip(image[:, 1], 0, 10000)/10000.

        return {"images": image, "classes": label, "image_paths": f"{img_name}-0.npy"}