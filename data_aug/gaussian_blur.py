import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from scipy.ndimage import convolve, gaussian_filter


# class GaussianBlur(object):
#     """blur a single image on CPU"""
#     def __init__(self, p, n_channels, kernel_size):
#         self.p = p
#         self.n_channels = n_channels
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_h = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=self.n_channels)
#         self.blur_v = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=(1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=self.n_channels)
#         self.k = kernel_size
#         self.r = radias
#
#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )
#
#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()
#
#     def __call__(self, sample):
#         if np.random.random() < self.p:
#             sigma = np.random.uniform(0.1, 2.0)
#             x = np.arange(-self.r, self.r + 1)
#             x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#             x = x / x.sum()
#             x = torch.from_numpy(x).view(1, -1).repeat(self.n_channels, 1)
#
#             self.blur_h.weight.data.copy_(x.view(self.n_channels, 1, self.k, 1))
#             self.blur_v.weight.data.copy_(x.view(self.n_channels, 1, 1, self.k))
#
#             with torch.no_grad():
#                 sample = self.blur(sample)
#                 sample = sample.squeeze()
#
#         return sample

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
