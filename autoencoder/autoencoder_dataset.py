import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import einops
import pandas as pd

class MitoSpaceAutoEncoderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset for loading .npy files from multiple subfolders within given root directories.
        Args:
            root_dirs (str or list of str): Root directory or list of root directories containing subfolders with .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.data_files = []
        self.transform = transform
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata_flagged.csv'))

        # Set the file paths based on the metadata
        self.metadata['file_path'] = self.metadata.apply(
            lambda row: os.path.join(root_dir, row['sample id'], f"{int(row['cell id']):06d}-{row['movie id']}.npy"), axis=1
        )
        n_start = len(self.metadata)
        print(f"Initial number of entries in metadata: {n_start}")
        # drop any rows where the "frames with stage issue" column has an entry
        self.metadata = self.metadata[self.metadata['frames with stage issue'].isna()]
        n_final = len(self.metadata)
        print(f"Final number of entries in metadata: {n_final} ({n_start - n_final} entries flagged and removed)")

        self.data_files = self.metadata['file_path'].tolist()

        #-- For multiple root dirs (not currently used)
        # Traverse all subfolders and gather npy files
        # for root_dir in root_dirs:
        #     for subfolder in sorted(os.listdir(root_dir)):
        #         subfolder_path = os.path.join(root_dir, subfolder)
        #         if os.path.isdir(subfolder_path):
        #             for file in sorted(os.listdir(subfolder_path)):
        #                 if file.endswith('.npy'):
        #                     file_path = os.path.join(subfolder_path, file)
        #                     self.data_files.append(file_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_name = self.data_files[idx]
        image = np.load(img_name)
        
        if self.transform:
            image = self.transform(image, img_name)
        
        image = einops.rearrange(torch.from_numpy(image).to(torch.float32), 't c z y x -> c t z y x')
        
        return image

class NormalizeChannelsByPath(object):
    """
    Transform that clips and normalizes images by channel if the path string contains a specified substring.

    Args:
        path_substr (str): Substring to look for in the file path.
        max_values (list of float): List of maximum values for each channel to use for
         clipping and normalization.
    """

    def __init__(self, path_substr, max_values):
        self.path_substr = path_substr
        self.max_values = max_values

    def __call__(self, image, path):
        if self.path_substr in path:
            for c, max_val in enumerate(self.max_values):
                if c < image.shape[1]:  # safeguard for channel count
                    image[:, c] = torch.clamp(image[:, c], 0, max_val) / max_val

            image = image.to(dtype=torch.float32)
        return image

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(path_substr='{self.path_substr}', "
            f"max_values={self.max_values})"
        )

def save_random_image(dataset, save_dir='.'):

    # Get a random index
    idx = np.random.randint(0, len(dataset))
    image = dataset[idx]

    # Choose random t and z indices
    t = np.random.randint(0, image.shape[0])  # Random time index
    z = np.random.randint(0, image.shape[2])  # Random depth index

    # Extract the image slice for the selected t and z
    img_slice = image[t, :, z, :, :]  # Shape should be (channels, height, width)

    # Plot the image slices for both channels
    plt.figure(figsize=(10, 5))
    for i in range(img_slice.shape[0]):
        plt.subplot(1, img_slice.shape[0], i + 1)
        plt.imshow(img_slice[i], cmap='viridis')  # Use 'viridis' colormap for better visualization
        plt.title(f'Channel {i}')
        plt.axis('off')

    # Save the figure
    file_name = f'random_image_t{t}_z{z}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.close()  # Close the figure to free memory


if __name__ == '__main__':

    data_dirs = [
        "/mnt/aquila0/others/MitoSpace4D/data/aligned",  # Summer 2024
        "/mnt/aquila0/ssd_processing/Others/MitoSpace4D/summer_2025_new/"
    ]

    ds_transform = NormalizeChannelsByPath(
        path_substr='2024',
        max_values=[25000, 10000]  # TMRM (ch0), MTG (ch1)
    )

    # Create a dataset object
    dataset = MitoSpaceAutoEncoderDataset(
        root_dirs=data_dirs,
        transform=ds_transform
    )
    print("Total samples in dataset:", len(dataset))

    sample_idx = np.random.randint(len(dataset))
    image = dataset[sample_idx]
    print(np.max(image))

    save_random_image(dataset)
