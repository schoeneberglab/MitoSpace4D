import torch.utils.data

from data_aug.gaussian_blur import GaussianBlur
from data_aug.random_brightness import RandomBrightness, RandomBrightnessGPU
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
# from exceptions.exceptions import InvalidDatasetSelection
from data_aug.mitospace_dataset import *
from data_aug.random_noise import *
from data_aug.random_rotate import RandomRotation
from data_aug.random_affine import RandomAffine, RandomAffineGPUWrapper
from data_aug.random_crop import RandomResizedCrop, RandomResizedCropGPU
from data_aug.random_exchange_flip import RandomExchangeFlip
from typing import Dict, List


class ContrastiveLearningDataset:
    def __init__(self, root_folder: str, args: Dict) -> None:
        self.root_folder = root_folder
        self.args = args

    def get_dataset(self, name: str, n_views: int, flag: str = 'train', seed: int = None,
                    pick_labels: List = None, samples_per_drug: int = None,
                    timesteps=None, zstacks=None) -> torch.utils.data.Dataset:
        valid_datasets = {'mitospace': lambda: MitoSpaceDataset(self.root_folder,
                                                                flag=flag,
                                                                seed=seed, pick_labels=pick_labels,
                                                                samples_per_drug=samples_per_drug,
                                                                timesteps=timesteps, zstacks=zstacks),
                          'stl10': lambda: datasets.STL10(self.root_folder, split=flag,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=False),
                          'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=(flag == 'train'),
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=False)
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise KeyError()
        else:
            return dataset_fn()


if __name__ == '__main__':
    config_path = '/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml'
    cfg = load_config(config_path)
    dataset = ContrastiveLearningDataset(cfg['data_params']['data_path'], cfg)

    train_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                        cfg['training']['n_views'],
                                        flag='train', seed=None,
                                        pick_labels=None,
                                        samples_per_drug=cfg['data_params']['samples_per_drug'],
                                        timesteps=cfg['data_params']['timesteps'],
                                        zstacks=cfg['data_params']['zstacks'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
               num_workers=cfg['training']['workers'], pin_memory=True, drop_last=True,
               prefetch_factor=cfg['training']['prefetch_factor'])

    # do a dummy loading of the dataset
    pbar = tqdm(train_loader)
    for batch in train_loader:
        pbar.update(1)