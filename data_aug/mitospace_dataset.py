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


class MitoSpaceDataModule(pl.LightningDataModule):
    def __init__(self, train_datasets: List[Dataset], val_datasets: List[Dataset], batch_size: int,
                 num_workers: int = 0, pin_memory: bool = True, drop_last: bool = True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self):
        # Return dataloader for training
        return DataLoader(ConcatDataset(self.train_datasets), batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)

    def val_dataloader(self):
        # Return dataloader for validation
        return DataLoader(ConcatDataset(self.val_datasets), batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)


class MitoSpaceDataset(Dataset):
    def __init__(self, root_dir: str, transform: Union[Compose, torchvision.transforms] = None, flag: str = 'train',
                 seed: int = None, pick_labels: List = None, samples_per_drug: int = None,
                 timesteps=None, zstacks=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.timesteps = timesteps
        self.zstacks = zstacks

        self.seed = 1123 if seed is None else seed

        print(f'Loading {flag} Dataset with split seed = {self.seed} ...')

        drug_labels = {}
        with open('/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt', 'r') as f:
            drugs_to_labels = f.readlines()
            for line in drugs_to_labels:
                folder, drug, label = line.split()
                drug_labels[folder] = {'drug': drug, 'label': int(label)}

        drug_folders = sorted([file for file in os.listdir(osp.join(self.root_dir, 'processed_data'))])

        self.all_filenames = []
        self.all_labels = []

        for drug_folder in drug_folders:
            filenames = sorted([file for file in os.listdir(osp.join(self.root_dir, 'processed_data', drug_folder))])
            filenames = [osp.join(self.root_dir, 'processed_data', drug_folder, file) for file in filenames]
            self.all_filenames.extend(filenames)
            self.all_labels.extend([drug_labels[drug_folder]['label']] * len(filenames))

        self.data = list(zip(self.all_filenames, self.all_labels))

        self.data = self.data * 200  # mimic the original dataset

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
        image = np.load(img_name, mmap_mode='r').astype(np.float32)

        image = image / 65535.

        label = self.labels[idx]

        # if self.transform:
        #     image = self.transform(image)  # image: [time, H, W, Z]
        #     image = minus_one_to_one_normalization(image)
        #     image = idxs_to_keep(image, idxs=None)
        #
        # else:
        #     image = minus_one_to_one_normalization(image)
        #     image = idxs_to_keep(image, idxs=None)

        return {"images": image, "classes": label}
