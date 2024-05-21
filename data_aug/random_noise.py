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

            noise = np.concatenate((np.random.normal(np.random.normal(self.mu[0], self.mu[0] * 0.2),
                                                     self.scale * sample.max(), sample[:1].shape),
                                    np.random.normal(np.random.normal(self.mu[1], self.mu[1] * 0.2),
                                                               self.scale * sample.max(), sample[:1].shape)),
                                             axis=0).astype(np.float32)
            sample = sample + noise

        return sample
