import typing as t
from typing import Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray, dtype
from tqdm import tqdm
from torchvision.models import mobilenet_v3_small

import torchvision.transforms as T
# from torchsummary import summary
import os
import os.path as osp

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


def read_image(filename):
    return io.imread(filename)


class StarModel(nn.Module):
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


def load_model():
    model = StarModel()
    model.to(DEVICE)
    model.load_state_dict(torch.load("/extraction_utils/cellfinder_checkpoints/best.ckpt",
                                     map_location=DEVICE))
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
        for bbox in sorted_bboxes[:]:
            # Calculate the intersection over union (IoU) between top_box and bbox
            iou = calculate_iou(top_box, bbox)

            # Remove bbox if its IoU with top_box is above the threshold
            if iou > threshold:
                sorted_bboxes.remove(bbox)

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
    #     p2, p98 = np.percentile(img, (10, 90))
    #     img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return (img * 255.0).astype(np.uint8)


DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

labels = [
    0, 1, 2, 3, 4
]

final_dataset = []
final_labels = []

model = load_model()

patch_idx = 0
inp_dir = sys.argv[1]
save_dir = sys.argv[2]

os.makedirs(save_dir, exist_ok=True)
data_dirs = [inp_dir]

for label_idx, data_dir in tqdm(enumerate(data_dirs)):
    for dir_no in range(0, 24):
        sample_dir = os.path.join(data_dir, str(dir_no))
        print(sample_dir)
        if os.path.exists(sample_dir):
            final_files = []

            for filename in sorted(glob(sample_dir + "/*.tif")):
                if "560nm" in filename:
                    final_files.append(filename)

            final_files_mito = []

            for filename in sorted(glob(sample_dir + "/*.tif")):
                if "488nm" in filename:
                    final_files_mito.append(filename)
            if len(final_files) < 30:
                continue
            filename = final_files[10]  # chose t=10 to select cell regions
            img = io.imread(filename)

            im = img[60, :, :]  # choosing cell based on the z=60
            bboxes = []
            patch_size = 256
            for i in tqdm(range(0, im.shape[0] - patch_size, 20)):
                for j in range(0, im.shape[1] - patch_size, 20):
                    with torch.no_grad():
                        patch = im[i:i + patch_size, j:j + patch_size]
                        patch = (patch - patch.min()) / (patch.max() - patch.min())
                        patch = preprocess_img_mito(patch)
                        threshold_value = filters.threshold_otsu(patch)
                        patch = patch > threshold_value
                        patch = torch.unsqueeze(torch.unsqueeze(torch.Tensor(patch), 0), 0)
                        patch = patch.to(DEVICE).float()
                        preds, pred_lb = model(patch)
                        pred_lb = pred_lb.detach().cpu().numpy()
                        if pred_lb > 0.9:
                            bboxes.append([j, i, j + patch_size, i + patch_size, pred_lb])

            selected_bboxes = non_maximal_suppression(bboxes, threshold=0.1)
            images, images_mito = [], []

            num_threads = 32  # You can adjust this number based on your system's resources

            # final_files = final_files[:2]  # for debugging
            # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Iterate through filenames with tqdm for progress tracking
                futures = {executor.submit(read_image, filename): filename for filename in final_files}

                # Use tqdm to track progress of tasks
                with tqdm(total=len(final_files)) as pbar:
                    for future in as_completed(futures):
                        pbar.update(1)  # Update progress bar for each completed task

                        # Retrieve results from futures
                images = [future.result() for future in futures]

            for bbox in tqdm(selected_bboxes):
                patches = []
                for idx, img in enumerate(images):

                    j, i, j_end, i_end, _ = bbox

                    if i_end - i != 256 or j_end - j != 256:
                        continue

                    patches.append(img[:, i: i_end, j: j_end])

                if len(patches) < 30:
                    continue

                np.save(os.path.join(save_dir, f"{str(patch_idx).zfill(6)}.npy"), np.array(patches))

                patch_idx = patch_idx + 1

            del images, images_mito
