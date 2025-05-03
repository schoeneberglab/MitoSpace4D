import os
import os.path as osp

import matplotlib.pyplot as plt
import napari
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Napari Visualiser')
parser.add_argument('--img_paths', nargs='+', help='Paths to the images to visualise')


def add_to_viewer(viewer, img_path, translate, channel=0, cmap='cyan', bounding_box_coords_3d=None, box_color='red',
                  box_edge_width=2):
    """
    Adds an image to the napari viewer and overlays 3D bounding boxes if provided.

    Parameters:
    - viewer: napari.Viewer
        The napari viewer instance.
    - img_path: str
        Path to the image file (expects a NumPy `.npy` file).
    - translate: tuple
        Translation to apply to the image in the viewer.
    - channel: int, optional
        The channel index to visualize from the image. Default is 0.
    - cmap: str, optional
        The colormap to use for the image. Default is 'cyan'.
    - bounding_box_coords_3d: list of ndarray, optional
        List of bounding box coordinates in 3D, each as a numpy array
        of shape (8, 3) representing the 8 corners of the box.
        Default is None (no bounding box).
    - box_color: str, optional
        Color of the bounding box edges. Default is 'red'.
    - box_edge_width: int, optional
        Width of the bounding box edges. Default is 2.
    """
    drug_name = img_path.split('/')[-2]
    img = np.load(img_path)

    if img.min() < 0:
        img = (img + 1)/2

    # img[:, 0] = np.clip(img[:, 0], 0, 25000)
    # img[:, 1] = np.clip(img[:, 1], 0, 10000)

    # Add the image to the viewer
    viewer.add_image(
        img[:, channel],
        name=f"Image {drug_name}",
        translate=translate,
        colormap=cmap,
        rendering='mip',  # Use maximum intensity projection for 3D
    )

    # Add 3D bounding boxes if coordinates are provided
    if bounding_box_coords_3d is not None:
        viewer.add_shapes(
            bounding_box_coords_3d,
            shape_type='path',  # Use 'path' to draw edges between points in 3D
            edge_color=box_color,
            edge_width=box_edge_width,
            name=f"3D Bounding Boxes {drug_name}",
        )


def visualise_samples(img_paths):
    napari_viewer = napari.Viewer()

    # assuming 4 images
    add_to_viewer(napari_viewer, img_paths[0], translate=(0, 0), channel=0)
    add_to_viewer(napari_viewer, img_paths[1], translate=(0, 256 + 10), channel=1)
    add_to_viewer(napari_viewer, img_paths[2], translate=(256 + 10, 0), channel=0)
    add_to_viewer(napari_viewer, img_paths[3], translate=(256 + 10, 256 + 10), channel=1)

    napari.run()


if __name__ == '__main__':
    fpaths = ['/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240816/000001.npy',
              '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240905/000001.npy']
    # Replace with the paths to the images you want to visualise

    viewer = napari.Viewer()

    for i, fpath in enumerate(fpaths[:5]):
        if not osp.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        add_to_viewer(viewer, fpath, translate=(i*256, 0), channel=0)
        add_to_viewer(viewer, fpath, translate=(i*256, 256 + 10), channel=1)

    napari.run()
