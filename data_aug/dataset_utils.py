from torchvision import datasets

from data_aug.mitospace_dataset import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mitospace_data_loaders(root_folder, to_load=None, shuffle=False, batch_size=256, pick_labels=None, seed=None,
                               samples_per_drug=None):
    if to_load is None:
        to_load = []
    loaders = {}

    if "train" in to_load:
        train_dataset = MitoSpaceDataset(root_folder,
                                        #  transform=transforms.Compose([transforms.ToTensor(),
                                        #                                transforms.Resize(size=(256, 256))]),
                                         flag='train', seed=seed, pick_labels=pick_labels,
                                         samples_per_drug=samples_per_drug)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["train"] = train_loader

    if "val" in to_load:
        val_dataset = MitoSpaceDataset(root_folder,
                                        #  transform=transforms.Compose([transforms.ToTensor(),
                                        #                                transforms.Resize(size=(256, 256))]),
                                       flag='val', seed=seed, pick_labels=pick_labels,
                                       samples_per_drug=samples_per_drug)

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["val"] = val_loader

    if "test" in to_load:
        test_dataset = MitoSpaceDataset(root_folder,
                                        #  transform=transforms.Compose([transforms.ToTensor(),
                                        #                                transforms.Resize(size=(256, 256))]),
                                        flag='test', seed=seed, pick_labels=pick_labels,
                                        samples_per_drug=samples_per_drug)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["test"] = test_loader

    if "all" in to_load:
        all_dataset = MitoSpaceDataset(root_folder, 
                                    # transform=transforms.Compose([transforms.ToTensor(),
                                                                                #   transforms.Resize(
                                                                                    #   size=(256, 256))]),
                                       flag='all', seed=seed, pick_labels=pick_labels,
                                       samples_per_drug=samples_per_drug)
        all_loader = DataLoader(all_dataset, batch_size=batch_size,
                                num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["all"] = all_loader

    if "stl10_train" in to_load or "stl10_test" in to_load or "stl10_val" in to_load:
        split = to_load[0][6:]
        stl10_dataset = datasets.STL10(root_folder, split=split,
                                    #    transform=transforms.Compose([transforms.ToTensor(),
                                    #                                  transforms.Resize(size=(96, 96))]),
                                       download=False)
        stl10_loader = DataLoader(stl10_dataset, batch_size=batch_size,
                                  num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["stl10"] = stl10_loader

    if "cifar10_train" in to_load or "cifar10_test" in to_load:
        split = to_load[0][8:]
        cifar10_dataset = datasets.CIFAR10(root_folder, train=(split == 'train'),
                                        #    transform=transforms.Compose([transforms.ToTensor(),
                                        #                                  transforms.Resize(size=(32, 32))]),
                                           download=False)
        cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size,
                                    num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["cifar10"] = cifar10_loader

    if "flybrain" in to_load:
        flybrain_dataset = FlyBrainDataset(root_folder,
                                           flag="train",
                                           seed=seed, pick_labels=pick_labels,
                                           samples_per_drug=samples_per_drug,
                                           timesteps=1, zstacks="all")
        flybrain_loader = DataLoader(flybrain_dataset, batch_size=batch_size,
                                    num_workers=32, drop_last=False, shuffle=shuffle)
        loaders["flybrain"] = flybrain_loader
    return loaders
