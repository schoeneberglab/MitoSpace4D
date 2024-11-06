import os
import os.path as osp
import napari
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Napari Visualiser')
parser.add_argument('--img_paths', nargs='+', help='Paths to the images to visualise')


def add_to_viewer(viewer, img_path, translate, channel=0):
    drug_name = img_path.split('/')[-2]
    img = np.load(img_path)

    # img = np.clip(img, 0, 5000)
    # img = img / 5000
    # img *= img*5000
    # img = img.astype(np.uint16)

    # visualising mito channel, change the index to 0 for tmrm channel
    viewer.add_image(img[:, channel], name=f"Image {drug_name}", translate=translate, colormap='cyan')


def visualise_samples(img_paths):
    napari_viewer = napari.Viewer()

    # assuming 4 images
    add_to_viewer(napari_viewer, img_paths[0], translate=(0, 0), channel=0)
    add_to_viewer(napari_viewer, img_paths[1], translate=(0, 256 + 10), channel=1)
    add_to_viewer(napari_viewer, img_paths[2], translate=(256 + 10, 0), channel=0)
    add_to_viewer(napari_viewer, img_paths[3], translate=(256 + 10, 256 + 10), channel=1)

    napari.run()


if __name__ == '__main__':
    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data'
    data_dir_2 = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data'

    drug1 = '20240830'
    drug2 = '20240830'

    # args = parser.parse_args()
    # img_paths = args.img_paths

    viewer = napari.Viewer()
    # visualise_samples(img_paths)

    filenames1 = os.listdir(os.path.join(data_dir, drug1))
    filenames2 = os.listdir(os.path.join(data_dir, drug1))

    # # Get two random, non-repeating indices
    # idx1_1, idx1_2 = random.sample(range(len(filenames1)), 2)
    idx1_1, idx1_2 = 0, 0
    idx2_1, idx2_2 = idx1_1, idx1_2

    # # Load and display the first image
    img_path1_1 = osp.join(data_dir, drug1, filenames1[idx1_1])
    img_path1_2 = osp.join(data_dir, drug1, filenames1[idx1_1])
    img_path2_1 = osp.join(data_dir, drug2, filenames2[idx2_1])
    img_path2_2 = osp.join(data_dir, drug2, filenames2[idx2_1])

    add_to_viewer(viewer, img_path1_1, translate=(0, 0), channel=0)
    add_to_viewer(viewer, img_path1_2, translate=(0, 256 + 10), channel=1)
    # add_to_viewer(viewer, img_path2_1, translate=(256 + 10, 0))
    # add_to_viewer(viewer, img_path2_1, translate=(256 + 10, 256 + 10))

    napari.run()
