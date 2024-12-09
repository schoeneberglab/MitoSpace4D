import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import load_config

class MitoSpaceAutoEncoderDataset(Dataset):
    def __init__(self, cfg):
        """
        Args:
            root_dir (string): Directory with all the subfolders and npy files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = cfg['data_params']['data_path']
        self.data_files = []
        self.seed = 42
        self.max_value_tmrm = cfg['data_params']['tmrm_clip']
        self.max_value_tracker = cfg['data_params']['mitotracker_clip']

        # Traverse all subfolders and gather npy files
        for subfolder in os.listdir(self.root_dir):
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith('.npy'):
                        file_path = os.path.join(subfolder_path, file)
                        self.data_files.append(file_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_name = self.data_files[idx]
        image = np.load(img_name)
        image = image.astype(np.float32)

        image[:, 0] = np.clip(image[:, 0], 0, self.max_value_tmrm)
        image[:, 0] = image[:, 0] / self.max_value_tmrm

        image[:, 1] = np.clip(image[:, 1], 0, self.max_value_tracker)
        image[:, 1] = image[:, 1] / self.max_value_tracker

        image = image.astype(np.float32)
        return image

def save_random_image(dataset, save_dir='.'):
    # Ensure the save directory exists
    # os.makedirs(save_dir, exist_ok=True)

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
    cfg = load_config('/home/dhruvagarwal/projects/Manav_MitoSpace/MitoSpace4D/autoencoder/config.yaml')
    dataset = MitoSpaceAutoEncoderDataset(cfg)
    print("Total samples in dataset:", len(dataset))

    sample_idx = np.random.randint(len(dataset))
    image = dataset[sample_idx]
    print(np.max(image))

