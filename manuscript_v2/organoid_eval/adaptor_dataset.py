import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random


class AdaptorDataset(Dataset):
    def __init__(self, csv_file, target_column='Label', task_type='regression', transform=None,
                 drugs_to_labels='/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt'):
        """
        Args:
            csv_file (str): Path to the CSV file.
            embeddings_file (str): Path to the .npy file containing embeddings.
            target_column (str): Name of the target column in the CSV file.
            task_type (str): 'regression' or 'classification'. For regression, min-max scaling is applied.
            transform (callable, optional): Optional transform to be applied to the embedding.
        """
        # Load the CSV file

        self.data_df = pd.read_csv(csv_file)
        if self.data_df.columns[0].startswith("Unnamed"):
            self.data_df = self.data_df.drop(columns=[self.data_df.columns[0]])

        # choose the embeddings as the columns with the prefix 'Emb_'
        self.embeddings = self.data_df.filter(regex='Emb_').values

        if len(self.data_df) != len(self.embeddings):
            raise ValueError("The number of rows in the CSV does not match the number of embeddings.")
        self.task_type = task_type.lower()
        self.transform = transform

        nan_mask = self.data_df[target_column].notna()
        self.data_df = self.data_df[nan_mask]
        self.embeddings = self.embeddings[nan_mask]
        self.targets = self.data_df[target_column].values

        if self.task_type == 'regression':
            self.targets = self.targets.astype(np.float32)
            p05 = np.percentile(self.targets, 5)
            p95 = np.percentile(self.targets, 95)
            print(f"Dropping: 5th percentile: {p05}, 95th percentile: {p95}")
            percentile_mask = (self.targets >= p05) & (self.targets <= p95)
            self.data_df = self.data_df[percentile_mask]
            self.embeddings = self.embeddings[percentile_mask]
            self.targets = self.targets[percentile_mask]

            # Adding the log scaling first
            # self.targets = np.log(self.targets)
            # self.targets = self.targets ** (1/3)

            # plotting the distribution of the targets after clipping
            plt.figure(figsize=(10, 6))
            plt.hist(self.targets, bins=50, color='blue', alpha=0.7)
            plt.title(f'Distribution of {target_column} after Clipping')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()

            self.min_target = self.targets.min()
            self.max_target = self.targets.max()
            if self.max_target - self.min_target != 0:
                self.targets = (self.targets - self.min_target) / (self.max_target - self.min_target)
            else:
                self.targets = self.targets - self.min_target
        else:
            self.targets = self.targets.astype(np.int64)
        print(len(self.data_df), len(self.targets), len(self.embeddings))

        assert len(self.data_df) == len(self.targets)
        assert len(self.data_df) == len(self.embeddings)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        # embedding = np.random.rand(2048)
        # embedding = self.embeddings[idx]
        if self.transform:
            embedding = self.transform(embedding)
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)

        if self.task_type == 'regression':
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)

            # # applying scaling to handle the skewness in the target distribution
            # target_tensor = target_tensor ** (1/3)
        else:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.long)

        sample = {'embedding': embedding, 'target': target_tensor}

        return sample

    def get_scaling_params(self):
        return self.min_target, self.max_target

    def save_scaling_parameters(self, file_path):
        """
        Save the min and max scaling parameters to a JSON file.
        Only applicable for regression tasks.
        Args:
            file_path (str): Destination file path for the scaling parameters.
        """
        if self.task_type != 'regression':
            raise ValueError("Scaling parameters are only available for regression tasks.")
        scaling_params = {
            'min': float(self.min_target),
            'max': float(self.max_target)
        }
        with open(file_path, 'w') as f:
            json.dump(scaling_params, f)
        print(f"Scaling parameters saved to {file_path}")


if __name__ == "__main__":
    # Paths to your data files
    base_path = "/home/dhruvagarwal/projects/MitoSpace4D/adaptors/adaptor_embeddings"
    csv_path = f'{base_path}/combined_data.csv'
    embeddings_path = f"{base_path}/combined_embeddings.npy"
    # Create a dataset for a regression task (min-max scaling is applied)
    dataset_reg = AdaptorDataset(csv_file=csv_path,
                                 embeddings_file=embeddings_path,
                                 target_column="Node Diffusivity",
                                 task_type="regression",
                                 data='summer')
    print("Total Samples", len(dataset_reg))
    # Save scaling parameters
    # dataset_reg.save_scaling_parameters("scaling_params.json")
    scaled_targets = dataset_reg.targets
    plt.figure(figsize=(10, 6))
    plt.hist(scaled_targets, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Scaled Node Counts (0-1)')
    plt.xlabel('Scaled Node Count')
    plt.ylabel('Frequency')
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    # Print some statistics about the scaled distribution
    print("\nScaled Distribution Statistics:")
    print(f"Mean: {scaled_targets.mean():.3f}")
    print(f"Std: {scaled_targets.std():.3f}")
    print(f"Min: {scaled_targets.min():.3f}")
    print(f"Max: {scaled_targets.max():.3f}")
    plt.show()
    # Create a dataset for a classification task (no scaling on targets)
    # dataset_clf = AdaptorDataset(csv_file=csv_path,
    #                             embeddings_file=embeddings_path,
    #                             target_column="Label",
    #                             task_type="classification",
    #                              data='summer')
    for sample in dataset_reg:
        continue