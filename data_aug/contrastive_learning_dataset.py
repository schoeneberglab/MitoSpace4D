import torch.utils.data

from .gaussian_blur import GaussianBlur
from .random_brightness import RandomBrightness
from torchvision import transforms, datasets
from .view_generator import ContrastiveLearningViewGenerator
# from exceptions.exceptions import InvalidDatasetSelection
from .mitospace_dataset import *
from .random_noise import *
from .random_rotate import RandomRotation
from .random_affine import RandomAffine, RandomAffineGPUWrapper
from .random_crop import RandomResizedCrop
from .random_exchange_flip import RandomExchangeFlip
from typing import Dict, List


class ContrastiveLearningDataset:
    def __init__(self, root_folder: str, args: Dict) -> None:
        self.root_folder = root_folder
        self.args = args
        cfg = self.args['data_params']['transforms']
        self.mapping = {"Crop": RandomResizedCrop(p=cfg['Crop']['p'], size=cfg['Crop']['size'],
                                                  scale=cfg['Crop']['scale'],
                                                  empty_area_check_idx=cfg['Crop']['empty_area_check_idx']),

                        "HorizontalFlip": transforms.RandomHorizontalFlip(p=cfg['HorizontalFlip']['p']),

                        "VerticalFlip": transforms.RandomVerticalFlip(p=cfg['VerticalFlip']['p']),

                        "Rotation": RandomRotation(p=cfg['Rotation']['p'], degrees=cfg['Rotation']['degrees']),

                        "Brightness": RandomBrightness(p=cfg['Brightness']['p'], factor=cfg['Brightness']['factor'],
                                                       apply_idx=cfg['Brightness']['apply_idx'],
                                                       method=cfg['Brightness']['method']),

                        "GaussianNoise": RandomGaussianNoise(p=cfg['GaussianNoise']['p'], mu=cfg['GaussianNoise']['mu'],
                                                             scale=cfg['GaussianNoise']['scale']),

                        "Affine": RandomAffine(p=cfg['Affine']['p'], degrees=cfg['Affine']['degrees'],
                                                  translate=cfg['Affine']['translate']),

                        "ExchangeFlip": RandomExchangeFlip(p=cfg['ExchangeFlip']['p']),
                        "GaussianBlur": GaussianBlur(p=cfg['GaussianBlur']['p'],
                                                     n_channels=cfg['GaussianBlur']['n_channels'],
                                                     kernel_size=cfg['GaussianBlur']['kernel_size'],
                                                     sigma=cfg['GaussianBlur']['sigma'])
                        }

    @staticmethod
    def get_transforms(self) -> transforms.Compose:
        transform_list = [self.mapping[trans] for trans in
                          self.args['data_params']['transforms']]
        data_transforms = transforms.Compose(transform_list)
        return data_transforms

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size))
                                              ])
        return data_transforms

    def get_dataset(self, name: str, n_views: int, flag: str = 'train', seed: int = None,
                    pick_labels: List = None, samples_per_drug: int = None,
                    transform=None,
                    timesteps=None, zstacks=None) -> torch.utils.data.Dataset:
        valid_datasets = {'mitospace': lambda: MitoSpaceDataset(self.root_folder,
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_transforms(
                                                                        self),
                                                                    n_views) if transform is None else transform,
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
