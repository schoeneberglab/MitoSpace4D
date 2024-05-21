import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class RandomExchangeFlip(object):
    """
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform() < self.p:
            sample = torch.cat((sample[..., sample.shape[1] // 2:, :], sample[..., :sample.shape[1] // 2, :]), dim=2)
        if np.random.uniform() < self.p / 2:
            sample = torch.cat((sample[..., sample.shape[1] // 2:], sample[..., :sample.shape[1] // 2]), dim=3)
            return sample
        else:
            return sample
