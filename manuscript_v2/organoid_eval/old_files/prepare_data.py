import os
import os.path as osp
from math import ceil

import napari
import tifffile as tiff
import random

import numpy as np


# get conditions from conditions column
def get_age(cell):
    if '20231220' in str(cell):
        age = '67'
    elif '20231221-1' in str(cell):
        age = '68'
    elif '20240221-1' in str(cell):
        age = '131'
    elif '20240221-2' in str(cell):
        age = '15'
    elif '20240222-1' in str(cell):
        age = '132'
    elif '20240222-2' in str(cell):
        age = '16'
    elif '20240223-2' in str(cell):
        age = '17'
    elif '20240226' in str(cell):
        age = '20'
    elif '20240117' in str(cell):
        age = '0'
    elif '20240118' in str(cell):
        age = '1'
    elif '20240119' in str(cell):
        age = '2'
    else:
        age = 'ERROR'
    return age


def napari_it(cell, age):
    viewer = napari.Viewer()

    viewer.add_image(
        cell[:, 0],
        name=f"Image {age}",
        translate=(0, 0),
        colormap='cyan',
        rendering='mip',  # Use maximum intensity projection for 3D
    )

    napari.run()


def prepare_mitospace_data(file_path, age):
    # load the tif file
    try:
        # add .tif extension if not present
        if not file_path.endswith('.tif'):
            fname = file_path.split('/')[-1]
            file_path = osp.join(file_path, f'{fname}.tif')
        tif_image = tiff.imread(file_path)
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        return None

    tif_image = tiff.imread(file_path)

    # Zing the image
    z_stride = ceil(tif_image.shape[0] / 60)
    # select every z_stride-th slice
    tif_image = tif_image[::z_stride]
    # expand it by adding zeros to 60 slices
    tif_image = np.pad(tif_image, ((0, 60 - tif_image.shape[0]), (0, 0), (0, 0)), mode='constant')

    # X and Ying the image
    h, w = tif_image.shape[1], tif_image.shape[2]

    if h < 256:
        tif_image = np.pad(tif_image, ((0, 0), (0, 256 - h), (0, 0)), mode='constant')
    elif h > 256:
        # select the middle 256 slices
        tif_image = tif_image[:, h // 2 - 128: h // 2 + 128, :]
    if w < 256:
        tif_image = np.pad(tif_image, ((0, 0), (0, 0), (0, 256 - w)), mode='constant')
    elif w > 256:
        # select the middle 256 slices
        tif_image = tif_image[:, :, w // 2 - 128: w // 2 + 128]

    tif_image = tif_image[None].repeat(2, axis=0)  # duplicate the image for 2 channels (pseudo tmrm and mitotracker)

    return tif_image


if __name__ == '__main__':
    # root_dir = '/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/tifs_for_mitospace'
    root_dir = '/media/dhruvagarwal/easystore/mitodevXmitospace/Analyzed_Data/MitoDev/single_cells'
    save_dir = '/media/dhruvagarwal/easystore/mitodevXmitospace/processed_data'
    os.makedirs(save_dir, exist_ok=True)

    identifiers = np.load(osp.join(save_dir.split('processed_data')[0], 'identifiers.npy'))
    labels = np.load(osp.join(save_dir.split('processed_data')[0], 'labels.npy'))
    identifiers = list(identifiers)
    labels = list(labels)
    idx = 667
    for dataset in sorted(os.listdir(root_dir)):
        dataset_path = osp.join(root_dir, dataset)
        samples = sorted(os.listdir(dataset_path))
        for sample in samples:
            sample_path = osp.join(dataset_path, sample)
            regions = sorted(os.listdir(sample_path))
            for region in regions:
                region_path = osp.join(sample_path, region)
                cells = sorted(os.listdir(region_path))
                ages = [get_age(cell) for cell in cells]

                for cell, age in zip(cells, ages):
                    cell_path = osp.join(region_path, cell, 'mitograph')
                    # print(cell_path)
                    # files = sorted(os.listdir(cell_path))
                    # cell_4d = []
                    # for file in files:
                    #     file_path = osp.join(cell_path, file)
                    #
                    #     frame = prepare_mitospace_data(file_path, age)
                    #     if frame is None:
                    #         continue
                    #     cell_4d.append(frame)
                    #
                    # cell_4d = np.stack(cell_4d)  # (t, c, z, h, w)
                    # # napari_it(cell_4d, age)
                    #
                    # # save the cell
                    # save_path = osp.join(save_dir, f'{idx:06d}.npy')
                    labels.append(int(age))
                    # np.save(save_path, cell_4d)
                    # idx += 1

                    identifier = cell_path.split('/')[-2].split('cell_')[1]
                    identifiers.append(identifier)

    np.save(osp.join(save_dir.split('processed_data')[0], 'identifiers.npy'), identifiers)
    np.save(osp.join(save_dir.split('processed_data')[0], 'labels.npy'), labels)

