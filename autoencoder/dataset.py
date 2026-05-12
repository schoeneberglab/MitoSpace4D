import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from glob import glob

class AutoencoderDataset(Dataset):
    """
    Dataset for Autoencoder training.
    Loads 3D images, optionally selects a specific channel, and returns (data, data).
    """

    def __init__(self, dataframe, transform=None, normalize=True, channel_index=None):
        self.meta = dataframe.reset_index(drop=True)
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.meta)

    @torch.no_grad()
    def normalize_channel(self, tensor: torch.Tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)

    def __getitem__(self, idx):
        path = self.meta.iloc[idx]['sample_path']

        try:
            data = np.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        # Normalize
        if self.normalize:
            data = self.normalize_channel(data)

        # pick a random timestep if multiple are present
        data = data[np.random.randint(0, data.shape[0])]

        if data.ndim == 3:
            # Add channel dimension: (D, H, W) -> (1, D, H, W)
            data = data[np.newaxis, ...]

        # pad to 64 if needed
        if data.shape[1] < 64:
            # pad the depth dimension starting from the end to keep the original data at the front
            pad_width = ((0, 0), (0, 64 - data.shape[1]), (0, 0), (0, 0))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)

        # Convert to tensor
        data = torch.from_numpy(data).float()

        # Transform
        if self.transform:
            data = self.transform(data)

        return data


def get_dataloaders(data_root,
                    batch_size=1,
                    val_split=0.1,
                    seed=1123,
                    num_samples=None,
                    num_workers=4,
                    train_transform=None,
                    val_transform=None,
                    channel_index=None,
                    **kwargs):
    """
    Creates train and validation dataloaders from a parquet manifest or image paths array.

    Args:
        manifest_path (str or array-like): Path to manifest.parquet OR array/list of image paths
        train_transform (callable): Transforms for the training set (e.g. flips, blur)
        val_transform (callable): Transforms for the validation set (usually None)
        channel_index (int, optional): The specific channel index to train on.
    """
    # Check if manifest_path is actually an array of paths (backward compatibility)
    # if isinstance(manifest_path, (list, np.ndarray)):
    #     Legacy mode: array of file paths
        # df = pd.DataFrame({'sample_path': manifest_path})
    # else:
        # New mode: Load from parquet manifest
        # df = pd.read_parquet(manifest_path)

    # 3. Optional Subsampling (for debugging)
    # if num_samples is not None and num_samples < len(df):
    #     df = df.sample(n=num_samples, random_state=seed)

    files = glob(data_root + '2024*/*-0-1.npy')
    df = pd.DataFrame({'sample_path': files})

    # 2. Split (Train / Val)
    # This ensures no data leakage between train and val
    df_train, df_val = train_test_split(df, test_size=val_split, random_state=seed)

    print(f"Dataset Split: {len(df_train)} Training, {len(df_val)} Validation")

    # 4. Create Datasets with specific transforms and channel selection
    train_ds = AutoencoderDataset(df_train, transform=train_transform, normalize=True, channel_index=channel_index)
    val_ds = AutoencoderDataset(df_val, transform=val_transform, normalize=True, channel_index=channel_index)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        **kwargs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        **kwargs
    )

    return train_loader, val_loader


if __name__ == "__main__":
    data_root = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v3_data/processed_data/"
    train_loader, val_loader = get_dataloaders(data_root, batch_size=2, num_workers=0, channel_index=0)
    for inputs, targets in train_loader:
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        break  # Just one batch for demo