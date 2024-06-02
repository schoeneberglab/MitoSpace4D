import random

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torchvision
from utils.utils import normalize
from utils.utils import agressive_sigmoid
from torch import nn
import random

np.random.seed(0)


def create_gaussian_mask(shape, center, sigma):
    """
    Create a Gaussian mask with given shape, centered at specified coordinates,
    and with specified standard deviation (sigma).

    Parameters:
        shape (tuple): Shape of the mask (height, width).
        center (tuple): Coordinates (y, x) of the center.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 2D Gaussian mask.
    """
    h, w = shape
    y, x = center
    y_range = np.arange(h)
    x_range = np.arange(w)
    xx, yy = np.meshgrid(x_range, y_range)

    # Calculate Gaussian mask
    mask = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return mask


class RandomBrightness(object):
    """
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, p=0.5, factor=200, apply_idx=[1], method="aggressive_sigmoid", spread=400):
        self.p = p
        self.factor = factor
        self.spread = spread
        self.apply_idx = apply_idx  # the channel on which to apply the augmentation
        self.method = method

    def __call__(self, sample):
        if np.random.uniform() <= self.p:
            if self.method == "aggressive_sigmoid":
                for idx in self.apply_idx:
                    alpha = random.uniform(3, self.factor + 1)
                    sample[idx] = agressive_sigmoid(sample[idx], alpha)

            elif self.method == "random_gaussian":
                # make a smooth brightness mask with gaussian noise, centered at random pixel locations in 2D
                h, w = sample.shape[-2:]  # Height and width of the image
                center_y = np.random.randint(h)
                center_x = np.random.randint(w)
                center = (center_y, center_x)

                # Define standard deviation of the Gaussian
                sigma = self.spread

                # Create Gaussian mask
                gaussian_mask = create_gaussian_mask((h, w), center, sigma).astype(np.float32)
                gaussian_mask = torch.from_numpy(gaussian_mask)

                if self.apply_idx == [-1]:
                    sample = sample * gaussian_mask
                else:
                    for idx in self.apply_idx:
                        sample[idx] = sample[idx] * gaussian_mask

            elif self.method == "random_pixels":
                # make a smooth brightness mask with random pixel locations in 2D
                h, w = sample.shape[-2:]

                mask = np.random.uniform(1, self.factor, size=(h, w)) * np.random.randint(2, size=(
                    h, w)) * np.random.randint(2, size=(h, w))
                mask = torch.Tensor(mask)

                for idx in self.apply_idx:
                    min_ = sample[idx].min()
                    max_ = sample[idx].max()

                    sample[idx] = sample[idx] * mask
                    new_min_ = sample[idx].min()
                    new_max_ = sample[idx].max()

                    sample[idx] = (sample[idx] - new_min_) / (new_max_ - new_min_) * (max_ - min_) + min_

        return sample


def create_gaussian_mask_gpu(shape, center, sigma):
    """
    Create a Gaussian mask with given shape, centered at specified coordinates,
    and with specified standard deviation (sigma).

    Parameters:
        shape (tuple): Shape of the mask (height, width).
        center (tuple): Coordinates (y, x) of the center.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: 2D Gaussian mask.
    """
    h, w = shape
    y, x = center
    y_range = torch.arange(h, dtype=torch.float32, device='cuda')
    x_range = torch.arange(w, dtype=torch.float32, device='cuda')
    yy, xx = torch.meshgrid(y_range, x_range)

    mask = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return mask


class RandomBrightnessGPU(nn.Module):
    """
    Args:
        p (float): Probability of applying the augmentation.
        factor (int): Factor for brightness adjustment.
        apply_idx (list): List of channels to apply the augmentation.
        method (str): Method of brightness adjustment.
        spread (float): Standard deviation for Gaussian mask.
    """

    def __init__(self, p=0.5, factor=200, apply_idx=[1], method="random_gaussian", spread=400):
        super(RandomBrightnessGPU, self).__init__()
        self.p = p
        self.factor = factor
        self.spread = spread
        self.apply_idx = apply_idx
        self.method = method

    def forward(self, sample):
        if torch.rand(1).item() <= self.p:
            if self.method == "aggressive_sigmoid":
                for idx in self.apply_idx:
                    alpha = random.uniform(3, self.factor + 1)
                    sample[:, idx] = agressive_sigmoid(sample[:, idx], alpha)

            elif self.method == "random_gaussian":
                h, w = sample.size()[-2:]  # Height and width of the image
                center_y = torch.randint(0, h, (1,)).item()
                center_x = torch.randint(0, w, (1,)).item()
                center = (center_y, center_x)

                sigma = self.spread
                gaussian_mask = create_gaussian_mask_gpu((h, w), center, sigma).to(sample.device)

                if self.apply_idx == [-1]:
                    sample = sample * gaussian_mask
                else:
                    for idx in self.apply_idx:
                        sample[:, idx] = sample[:, idx] * gaussian_mask

            elif self.method == "random_pixels":
                h, w = sample.size()[-2:]
                mask = (torch.rand(h, w, device=sample.device) * (self.factor - 1) + 1) * \
                       torch.randint(0, 2, (h, w), device=sample.device) * \
                       torch.randint(0, 2, (h, w), device=sample.device)

                for idx in self.apply_idx:
                    min_ = sample[:, idx].min()
                    max_ = sample[:, idx].max()

                    sample[:, idx] = sample[:, idx] * mask
                    new_min_ = sample[:, idx].min()
                    new_max_ = sample[:, idx].max()

                    sample[:, idx] = (sample[:, idx] - new_min_) / (new_max_ - new_min_) * (max_ - min_) + min_

        return sample
