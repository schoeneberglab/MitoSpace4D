import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MitoSpace3DDataset(Dataset):
    def __init__(self, metadata_file, **kwargs):
        df_metadata = pd.read_csv(metadata_file)
        self.file_paths = df_metadata['file_path'].values
        # self.labels = df_metadata['label'].values

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.file_paths[idx]))


class MitoSpace4DDataset(Dataset):
    def __init__(self, metadata_file):
        self.df_metadata = pd.read_csv(metadata_file)
        self.cell_ids = self.df_metadata['cell_id'].unique()
        # self.labels = df_metadata['label'].values

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        frame_paths = sorted(self.df_metadata[self.df_metadata['cell_id'] == cell_id]['file_path'].values)
        frames = []
        for frame_path in frame_paths:
            frames.append(np.load(frame_path))
        return torch.from_numpy(np.stack(frames, axis=0))

# def get_dataloaders(metadata_file, batch_size=1, test_split=0.1, seed=1123):
#     """
#     Centralized split logic to ensure Train and Eval see the same files.
#     """
#     paths_train, paths_val, emb_train, emb_val = train_test_split(
#         image_paths, embeddings, test_size=test_split, random_state=seed
#     )
#
#     if num_train_samples is not None:
#         indices = np.random.choice(len(paths_train), num_train_samples, replace=False)
#         paths_train = paths_train[indices]
#         emb_train = emb_train[indices]
#
#     train_ds = PseudocolorDataset(paths_train, emb_train)
#     val_ds = PseudocolorDataset(paths_val, emb_val)
#
#     # Pin memory helps with transferring large 3D volumes to GPU
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#
#     return train_loader, val_loader