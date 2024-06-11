import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from scipy.ndimage import convolve, gaussian_filter
import torch.nn.functional as F
import torchvision.transforms as T


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, p, n_channels, kernel_size, sigma):
        self.p = p
        self.n_channels = n_channels
        self.radius = kernel_size // 2
        self.kernel_size = self.radius * 2 + 1
        self.k = kernel_size
        self.sigma_range = (0.5, sigma)

    def gaussian_kernel(self, sigma):
        x = np.linspace(-self.radius, self.radius+1, self.kernel_size)
        y = np.linspace(-self.radius, self.radius+1, self.kernel_size)
        x, y = np.meshgrid(x, y)

        # Compute the Gaussian kernel
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        return kernel

    def blur_image(self, image, kernel):
        blurred_image = np.zeros_like(image)
        for c in range(self.n_channels):
            blurred_image[c] = convolve(image[c], kernel, mode='constant', cval=0.0)
        return blurred_image

    def __call__(self, sample):
        if np.random.random() < self.p:
            sigma = np.random.uniform(*self.sigma_range)
            kernel = self.gaussian_kernel(sigma)
            sample_np = np.array(sample)

            # if len(sample_np.shape) == 2:  # Grayscale image
            #     sample_np = sample_np[np.newaxis]

            blurred_image = gaussian_filter(sample_np, sigma=(0, 0, sigma, sigma),
                                            mode='constant', cval=0.0)

            sample = torch.from_numpy(blurred_image)

        return sample


class GaussianBlurGPU(nn.Module):
    """Apply Gaussian blur to a batch of images on GPU"""

    def __init__(self, p, n_channels, kernel_size, sigma):
        super(GaussianBlurGPU, self).__init__()
        self.p = p
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.sigma_range = (0.5, sigma)
        self.radius = kernel_size // 2

    def create_gaussian_kernel(self, sigma):
        # Create a 2D Gaussian kernel
        x = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        y = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        x, y = torch.meshgrid(x, y)
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        kernel = kernel / kernel.sum()
        return kernel

    def apply_kernel(self, img, kernel):
        # Apply Gaussian kernel to the image
        kernel = kernel.expand(1, *kernel.shape).to(img.device)
        img = F.conv2d(img.unsqueeze(1), kernel.unsqueeze(1), padding=self.radius, stride=(1, 1))
        return img

    def forward(self, sample):
        if torch.rand(1).item() < self.p:
            sigma = torch.FloatTensor(1).uniform_(*self.sigma_range).item()
            kernel = self.create_gaussian_kernel(sigma).to(sample.device)
            kernel /= kernel.sum()  # Normalize the kernel

            batch_size, time, z, height, width = sample.size()
            sample = sample.view(-1, height, width)

            blurred_images = self.apply_kernel(sample, kernel)

            blurred_images = blurred_images.view(batch_size, time, z, height, width)

            return blurred_images
        return sample
