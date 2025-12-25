from typing import Tuple, List

import torch
from torch import nn, Tensor
import numpy as np
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomBrightness, \
    RandomGaussianNoise, RandomGaussianBlur, RandomErasing, RandomRotation, RandomAffine, RandomHorizontalFlip3D, \
    RandomVerticalFlip3D, RandomDepthicalFlip3D, RandomRotation3D, RandomAffine3D


class RandomTimeFlip(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.flipper = RandomDepthicalFlip3D(p=self.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swap time dimension with z to apply the z-flipper to time
        # x is (b, t, c, z, h, w)
        b, t, c, z, h, w = x.size()
        x = x.permute(0, 3, 2, 1, 4, 5) # (b, z, c, t, h, w)
        x = x.reshape(b*z, c, t, h, w)
        x = self.flipper(x)
        x = x.view(b, z, c, t, h, w)
        x = x.permute(0, 3, 2, 1, 4, 5) # (b, t, c, z, h, w)

        return x


class RandomExchangeFlip(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x to be (b, c, z, h, w)

        slices = []
        for dim in [2, 3, 4]:  # z, h, w
            if np.random.uniform() < self.p:
                mid = x.shape[dim] // 2
                idx = torch.arange(x.shape[dim], device=x.device)
                idx = torch.cat((idx[mid:], idx[:mid]))
                slices.append((dim, idx))

        for dim, idx in slices:
            x = x.index_select(dim, idx)

        return x


class RandomTimeMask(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p  # p actually doesn't matter here; only when it's zero, we don't apply

        self.time_delay = [16, 14, 12, 10, 8, 6, 4, 2, 0]
        self.probs_time_delay = [0.03, 0.038, 0.047, 0.059, 0.074, 0.092, 0.115, 0.144, 0.401]

        self.clip_len = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
        self.clip_len_probs = [0.501, 0.116,0.092,0.074,0.059,0.047,0.038,0.030,0.024,0.019]


    def forward(self, x: torch.Tensor):
        if self.p == 0:
            return [x, x]

        clip_len = np.random.choice(self.clip_len, p=self.clip_len_probs)

        # select a random continuous window of length clip_len
        start = np.random.randint(0, x.size(1) - clip_len) if x.size(1) > clip_len else 0
        end = start + clip_len
        mask = torch.zeros_like(x, device=x.device)
        mask[:, start:end, :, :, :] = 1

        time_delay = np.random.choice(self.time_delay, p=self.probs_time_delay)
        time_delay = min(time_delay, clip_len)

        mask_2 = mask.clone()
        mask_2[:, start:start + time_delay, :, :, :] = 0
        mask_2[:, end: end + time_delay, :, :, :] = 1

        x_1 = x * mask
        x_2 = x * mask_2

        return [x_1, x_2]

class RandomBrightness(nn.Module):
    def __init__(self, p=0.5, lower=1, upper=1) -> None:
        super().__init__()
        self.p = p
        self.range = (lower, upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.uniform() < self.p:
            factor = np.random.uniform(self.range[0], self.range[1])
            x = x + factor  # that's what kornia's brightness augmentation does
            #x = torch.clamp(x, 0, 1)

        return x


class DataAugmentation(nn.Module):
    def __init__(self, cfg_aug=None, zero_mean_norm=True) -> None:
        super().__init__()

        self.temporal_transform_1 = RandomTimeMask(p=cfg_aug['RandomTimeMask']['p'])
        self.temporal_transform_2 = RandomTimeFlip(p=cfg_aug['RandomTimeFlip']['p'])

        self.transforms_2d = nn.Sequential(
            RandomResizedCrop(p=cfg_aug['ResizedCrop']['p'],
                              size=(cfg_aug['ResizedCrop']['size'], cfg_aug['ResizedCrop']['size']),
                              scale=(cfg_aug['ResizedCrop']['scale'][0], cfg_aug['ResizedCrop']['scale'][1])),
            RandomGaussianBlur(p=cfg_aug['GaussianBlur']['p'],
                               kernel_size=(cfg_aug['GaussianBlur']['kernel_size'],
                                            cfg_aug['GaussianBlur']['kernel_size']),
                               sigma=(cfg_aug['GaussianBlur']['sigma'], cfg_aug['GaussianBlur']['sigma'])),
            RandomErasing(p=cfg_aug['RandomErasing']['p'],
                          scale=(cfg_aug['RandomErasing']['scale'][0], cfg_aug['RandomErasing']['scale'][1]),
                          ratio=(cfg_aug['RandomErasing']['ratio'][0], cfg_aug['RandomErasing']['ratio'][1])),
            RandomBrightness(p=cfg_aug['RandomBrightness']['p'],
                             lower=cfg_aug['RandomBrightness']['lower'],
                             upper=cfg_aug['RandomBrightness']['upper'],
                             ),
            RandomGaussianNoise(p=cfg_aug['GaussianNoise']['p'],
                                mean=cfg_aug['GaussianNoise']['mu'],
                                std=cfg_aug['GaussianNoise']['scale']),
        )

        self.transforms_3d = nn.Sequential(RandomHorizontalFlip3D(p=cfg_aug['HorizontalFlip3D']['p']),
                                           RandomVerticalFlip3D(p=cfg_aug['VerticalFlip3D']['p']),
                                           RandomDepthicalFlip3D(p=cfg_aug['DepthicalFlip3D']['p']),
                                           RandomRotation3D(p=cfg_aug['RandomRotation3D']['p'],
                                                            degrees=cfg_aug['RandomRotation3D']['degrees']),
                                           RandomAffine3D(p=cfg_aug['RandomAffine3D']['p'],
                                                          degrees=cfg_aug['RandomAffine3D']['degrees'],
                                                          translate=(cfg_aug['RandomAffine3D']['translate'][0],
                                                                     cfg_aug['RandomAffine3D']['translate'][1],
                                                                     cfg_aug['RandomAffine3D']['translate'][2])),
                                           RandomExchangeFlip(p=cfg_aug['RandomExchangeFlip']['p'])
                                           )

        self.n_views = 2
        assert self.n_views == 2, "Only two views are supported for now"

        self.zero_mean_norm = zero_mean_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, z, h, w = x.size()

        views = self.temporal_transform_1(x) # (b, t, c, z, h, w), (b, t, c, z, h, w)
        assert len(views) == self.n_views, f"Number of views should be {self.n_views}"

        views = [view.view(b, -1, h, w) for view in views]

        for i in range(self.n_views):
            views[i] = self.transforms_2d(views[i]).view(b, t, c, z, h, w)  # same 2D transforms for all t and z
            views[i] = self.transforms_3d(views[i].view(b, t * c, z, h, w)).view(b, t, c, z, h,
                                                                         w)  # same 3D transforms for all t
            views[i] = self.temporal_transform_2(views[i])  # (b, t, c, z, h, w)

        views = torch.stack(views, dim=0)  # (n_views, b, t, c, z, h, w)
        views = views.view(-1, *views.shape[2:])  # (n_views*b, t, c, z, h, w)

        if self.zero_mean_norm:
            return 2 * views - 1

        return views
