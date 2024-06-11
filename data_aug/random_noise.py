import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

np.random.seed(0)


class RandomGaussianNoise(object):
    """
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, mu=[0.04, 0.04], scale=0.05, p=0.5):
        self.scale = scale
        self.mu = mu
        self.p = p

    def __call__(self, sample):
        if np.random.uniform() <= self.p:
            noise = (torch.randn(sample.size()[-2:], dtype=sample.dtype, device=sample.device) * self.scale +
                     self.mu[0])
            sample = sample + noise
            sample = torch.clip(sample, 0, 1)

        return sample


class RandomGaussianNoiseGPU(nn.Module):
    """
    Apply random Gaussian noise to a batch of images.

    Args:
        mu (list): Mean of the Gaussian noise for each channel.
        scale (float): Scale of the Gaussian noise.
        p (float): Probability of applying the noise.
    """

    def __init__(self, mu=[0.04, 0.04], scale=0.05, p=0.5):
        super(RandomGaussianNoiseGPU, self).__init__()
        self.mu = mu
        self.scale = scale
        self.p = p

    def forward(self, sample):
        if torch.rand(1).item() <= self.p:
            noise = torch.normal(mean=torch.tensor(self.mu[0], device=sample.device),
                                 std=torch.tensor(self.mu[0] * 0.2, device=sample.device),
                                 size=sample.shape).float()
            noise = noise * (self.scale * sample.max())
            sample = sample + noise

        return sample