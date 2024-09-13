import typing as t
from typing import Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from numpy import ndarray, dtype
from tqdm import tqdm
from torchvision.models import mobilenet_v3_small

import torchvision.transforms as T
# from torchsummary import summary
import os
import os.path as osp
import gc

import pdb

import skimage.io as io
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage import feature

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from glob import glob

from concurrent.futures import ThreadPoolExecutor, as_completed

import sys


def read_image(zip_fname):
    f1, f2 = zip_fname
    tmrm = io.imread(f1)
    mito = io.imread(f2)

    return np.stack([tmrm, mito])


class CellExtractor2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, (3, 3), padding='same')
        self.bbone = mobilenet_v3_small()

        self.label_head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bbone(self.conv1(x))

        labels = self.label_head(x)

        return None, labels


def load_model(device, ckpt_path):
    model = CellExtractor2D()
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def non_maximal_suppression(bboxes, threshold=0.5):
    # Sort bounding boxes by confidence in descending order
    sorted_bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # Initialize a list to store the selected bounding boxes
    selected_bboxes = []

    while len(sorted_bboxes) > 0:
        # Select the bounding box with the highest confidence (top box)
        top_box = sorted_bboxes.pop(0)
        selected_bboxes.append(top_box)

        # Iterate through the remaining bounding boxes
        remaining_bboxes = []
        for bbox in sorted_bboxes:
            # Calculate the intersection over union (IoU) between top_box and bbox
            iou = calculate_iou(top_box, bbox)

            # Only keep bbox if its IoU with top_box is below the threshold
            if iou <= threshold:
                remaining_bboxes.append(bbox)

        # Update sorted_bboxes with the boxes that were not removed
        sorted_bboxes = remaining_bboxes

    return selected_bboxes


def calculate_iou(box1, box2):
    # Extract coordinates from the bounding boxes
    x1_tl, y1_tl, x1_br, y1_br, _ = box1
    x2_tl, y2_tl, x2_br, y2_br, _ = box2

    # Calculate the intersection coordinates
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))

    # Calculate the areas of the bounding boxes and the intersection
    area_box1 = (x1_br - x1_tl) * (y1_br - y1_tl)
    area_box2 = (x2_br - x2_tl) * (y2_br - y2_tl)
    area_overlap = x_overlap * y_overlap

    # Calculate the intersection over union (IoU)
    iou = area_overlap / (area_box1 + area_box2 - area_overlap + 1e-6)

    return iou


def preprocess_img_mito(img):
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return (img * 255.0).astype(np.uint8)


def setup(ckpt_path):
    DEVICE = "cpu"

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"

    model = load_model(DEVICE, ckpt_path)

    return model, DEVICE


def get_bbox_from_2d_patch(filename, model, device, patch_size, stride, zs_of_interest,
                           save_bbox_plot=False, region=None):
    img = io.imread(filename)
    bs = 256

    bboxes = []
    for z in zs_of_interest:
        im = img[z, :, :]  # z-patch to be used for the cell extraction

        patches = []
        bboxes_z = []
        for i in tqdm(range(0, im.shape[0] - patch_size, stride)):
            for j in range(0, im.shape[1] - patch_size, stride):
                patch = im[i:i + patch_size, j:j + patch_size]
                patch = (patch - patch.min()) / (patch.max() - patch.min())
                patch = preprocess_img_mito(patch)
                threshold_value = filters.threshold_otsu(patch)
                patch = patch > threshold_value
                patch = torch.unsqueeze(torch.unsqueeze(torch.Tensor(patch), 0), 0)
                patches.append(patch)
                bboxes_z.append([j, i, j + patch_size, i + patch_size])

        patches = torch.cat(patches, 0)
        probs = []
        for i in range(0, patches.shape[0], bs):
            with torch.no_grad():
                output = model(patches[i:i + bs].to(device).float())
                probs.append(output[1].cpu().numpy())

        probs = np.concatenate(probs, 0)
        bboxes_z = np.array(bboxes_z)

        # keep bboxes with probability > 0.9
        bboxes_z = [list(bbox) + list(prob) for bbox, prob in zip(bboxes_z, probs) if prob > 0.9]
        bboxes.extend(bboxes_z)

    selected_bboxes = non_maximal_suppression(bboxes, threshold=0.1)

    if save_bbox_plot:
        for z in zs_of_interest:
            im = img[z, :, :]
            cm = plt.cm.get_cmap('viridis')
            im = (cm(im)[..., :3] * 255).astype(np.uint8)
            im = Image.fromarray(im)
            draw = ImageDraw.Draw(im)
            for bbox in selected_bboxes:
                left, top, right, bottom, _ = bbox
                draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
            os.makedirs(f'/home/dhruvagarwal/projects/{region}', exist_ok=True)
            im.save(f'/home/dhruvagarwal/projects/{region}/{z}.png')

    return selected_bboxes


if __name__ == '__main__':
    # config
    num_timesteps = 20
    time_of_interest = 10  # chose t=10 to select cell regions
    patch_size = 256
    stride = 10  # stride used for moving the patch in 2D
    zs_of_interest = [60, 80, 100, 120]  # chose zs to select cell regions
    num_threads = 4  # You can adjust this number based on your system's resources
    debug = False
    ckpt_path = "/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/cellfinder_checkpoints/best.ckpt"

    model, device = setup(ckpt_path)

    patch_idx = 0
    inp_dir = sys.argv[1]
    save_dir = sys.argv[2]

    os.makedirs(save_dir, exist_ok=True)

    for region in range(0, 24):  # iterate over the regions (24 is the max number of region in the dataset for now)
        sample_dir = os.path.join(inp_dir, str(region))
        print(sample_dir)

        if not os.path.exists(sample_dir):
            print(f"The {region} doesn't exist")
            continue

        tmrm_files = [filename for filename in sorted(glob(sample_dir + "/*.tif")) if "560nm" in filename]
        mito_files = [filename for filename in sorted(glob(sample_dir + "/*.tif")) if "488nm" in filename]

        assert len(tmrm_files) == len(mito_files), "The number of files in the directories are not equal"
        assert len(tmrm_files) == num_timesteps, f"there should be {num_timesteps} timesteps"

        filename = mito_files[time_of_interest]  # file to be used for the cell extraction
        bbox_from_2d = get_bbox_from_2d_patch(filename, model, device, patch_size, stride, zs_of_interest,
                                              save_bbox_plot=False, region=region)

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Iterate through filenames with tqdm for progress tracking
            futures = {executor.submit(read_image, zip_fname): zip_fname for zip_fname in zip(tmrm_files, mito_files)}

            # Use tqdm to track progress of tasks
            with tqdm(total=len(tmrm_files)) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)  # Update progress bar for each completed task

                    # Retrieve results from futures
            images = [future.result() for future in futures]

        for bbox in tqdm(bbox_from_2d):
            patches = []
            for idx, img in enumerate(images):

                j, i, j_end, i_end, _ = bbox

                if i_end - i != 256 or j_end - j != 256:
                    continue

                patches.append(img[..., i: i_end, j: j_end])

            np.save(os.path.join(save_dir, f"{str(patch_idx).zfill(6)}.npy"), np.array(patches))

            patch_idx = patch_idx + 1

        del images, patches, bbox_from_2d
        gc.collect()
        torch.cuda.empty_cache()
