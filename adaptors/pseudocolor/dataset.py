import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PseudocolorDataset(Dataset):
    def __init__(self, file_paths, embeddings, timepoints_per_file=1):
        self.file_paths = file_paths
        self.embeddings = embeddings
        self.timepoints_per_file = timepoints_per_file
        self.has_time_embeddings = (len(embeddings.shape) == 3)

    def __len__(self):
        return len(self.file_paths) * self.timepoints_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.timepoints_per_file
        time_idx = idx % self.timepoints_per_file
        path = self.file_paths[file_idx]

        if self.has_time_embeddings:
            embedding = self.embeddings[file_idx, time_idx]
        else:
            embedding = self.embeddings[file_idx]

        data_5d = np.load(path, mmap_mode='r')
        data_3d = data_5d[:, time_idx, :, :, :].copy()
        tmrm, morph = data_3d[0], data_3d[1]

        # Normalize
        if morph.max() > morph.min():
            morph = (morph - morph.min()) / (morph.max() - morph.min())
        if tmrm.max() > tmrm.min():
            tmrm = (tmrm - tmrm.min()) / (tmrm.max() - tmrm.min())

        return (torch.FloatTensor(morph).unsqueeze(0),
                torch.FloatTensor(embedding),
                torch.FloatTensor(tmrm).unsqueeze(0))


def get_dataloaders(image_paths, embeddings, batch_size=1, test_split=0.1, seed=42, num_train_samples=1000):
    """
    Centralized split logic to ensure Train and Eval see the same files.
    """
    paths_train, paths_val, emb_train, emb_val = train_test_split(
        image_paths, embeddings, test_size=test_split, random_state=seed
    )

    if num_train_samples is not None:
        indices = np.random.choice(len(paths_train), num_train_samples, replace=False)
        paths_train = paths_train[indices]
        emb_train = emb_train[indices]

    train_ds = PseudocolorDataset(paths_train, emb_train)
    val_ds = PseudocolorDataset(paths_val, emb_val)

    # Pin memory helps with transferring large 3D volumes to GPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader