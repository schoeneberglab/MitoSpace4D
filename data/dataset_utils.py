from data.mitospace_dataset import *
from torch.utils.data import DataLoader


def get_mitospace_data_loaders(root_folder, to_load=None, shuffle=False, batch_size=256, pick_labels=None, seed=None,
                               samples_per_drug=None, timesteps=None, zstacks=None, num_workers=0):
    if to_load is None:
        to_load = []
    loaders = {}

    if "train" in to_load:
        train_dataset = MitoSpaceDataset(root_folder,
                                         flag='train',
                                         seed=seed, pick_labels=pick_labels,
                                         samples_per_drug=samples_per_drug,
                                         timesteps=timesteps, zstacks=zstacks)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=8, drop_last=False, shuffle=shuffle)
        loaders["train"] = train_loader

    if "val" in to_load:
        val_dataset = MitoSpaceDataset(root_folder,
                                       flag='val',
                                       seed=seed, pick_labels=pick_labels,
                                       samples_per_drug=samples_per_drug,
                                       timesteps=timesteps, zstacks=zstacks)

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=8, drop_last=False, shuffle=shuffle)
        loaders["val"] = val_loader

    if "test" in to_load:
        test_dataset = MitoSpaceDataset(root_folder,
                                        flag='test', seed=seed, pick_labels=pick_labels,
                                        samples_per_drug=samples_per_drug)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=8, drop_last=False, shuffle=shuffle)
        loaders["test"] = test_loader

    if "all" in to_load:
        all_dataset = MitoSpaceDataset(root_folder,
                                       flag='all', seed=seed, pick_labels=pick_labels,
                                       samples_per_drug=samples_per_drug)
        all_loader = DataLoader(all_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)
        loaders["all"] = all_loader

    return loaders
