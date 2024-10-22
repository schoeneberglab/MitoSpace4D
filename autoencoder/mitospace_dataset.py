import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MitoSpaceAutoEncoderDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the subfolders and npy files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.data_files = []
        self.seed = 42

        # Traverse all subfolders and gather npy files
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith('.npy'):
                        file_path = os.path.join(subfolder_path, file)
                        self.data_files.append(file_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_name = self.data_files[idx]
        image = np.load(img_name, mmap_mode='r')

        image = np.clip(image, 0, 20000)
        image = image / 20000
        image = image.astype(np.float16)

        return torch.from_numpy(image)
    
if __name__ == '__main__':
    # Create a dataset object
    dataset = MitoSpaceAutoEncoderDataset(root_dir='data')
    print("Total samples in dataset:", len(dataset))
    
    # Create DataLoader for training
    train_loader = DataLoader(dataset, batch_size=32,
                              shuffle=True, 
                              drop_last=True, 
                              num_workers=6, 
                              pin_memory=True, 
                              prefetch_factor=2
                        )

    # Check the number of batches in train_loader
    print("Number of batches in train_loader:", len(train_loader))

    # Calculate total samples in train_loader using a loop
    for batch in train_loader:
        print(batch.shape)