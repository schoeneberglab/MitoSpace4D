import os
import os.path as osp

import matplotlib.pyplot as plt
import napari
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Visualize 4D movies (.npy) using Napari.')
parser.add_argument('--infiles', '-i', nargs='+', help='Path(s) to the image(s) to visualise')

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
    drug_name = img_path.split('/')[-2] + '/' + img_path.split('/')[-1].split('.')[0]
    img = np.load(img_path)

    if img.min() < 0:
        img = (img + 1)/2

    # img[:, 0] = np.clip(img[:, 0], 0, 25000)
    # img[:, 1] = np.clip(img[:, 1], 0, 10000)

    # Add the image to the viewer
    viewer.add_image(
        # img[:, channel], # for 4d
        img[channel], # for 3d
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


# def visualise_samples(img_paths):
#     napari_viewer = napari.Viewer()

#     # assuming 4 images
#     add_to_viewer(napari_viewer, img_paths[0], translate=(0, 0), channel=0)
#     add_to_viewer(napari_viewer, img_paths[1], translate=(0, 256 + 10), channel=1)
#     add_to_viewer(napari_viewer, img_paths[2], translate=(256 + 10, 0), channel=0)
#     add_to_viewer(napari_viewer, img_paths[3], translate=(256 + 10, 256 + 10), channel=1)

#     napari.run()


if __name__ == '__main__':
    # fpaths = ['/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240816/000001.npy',
    #           '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240905/000001.npy']

    # Replace with the paths to the images you want to visualise
    # fpaths = ["/run/user/1002/gvfs/smb-share:server=aquila0.jslab.ucsd.edu,share=ssd_processing/Others/MitoSpace4D/2025_summer/20250811-1/000005-0.npy"]
    # fpaths = ["/home/earkfeld/Projects/MitoSpace4D/sample_0_decoded.npy"]
    
    # fpaths = ['/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_summer_new/20250724-1/000018-1.npy',
    #           '/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_summer_new/20250724-1/000020-2.npy',
    #           '/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_summer_new/20250724-1/000041-0.npy']

    fpaths = ["/mnt/aquila/SSD_processing/Others/MitoSpace4D/leukemia_drug_resistance_data/20251007-1/000001-0.npy"]

    viewer = napari.Viewer(ndisplay=3)

    for i, fpath in enumerate(fpaths):
        if not osp.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        add_to_viewer(viewer, fpath, translate=(i*256, 0), channel=0)
        add_to_viewer(viewer, fpath, translate=(i*256, 256 + 10), channel=1)

    napari.run()
