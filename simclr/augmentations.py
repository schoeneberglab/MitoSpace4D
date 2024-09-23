import torch
from torch import nn
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomBrightness, \
    RandomGaussianNoise, RandomGaussianBlur, RandomErasing

from utils.utils import minus_one_to_one_normalization


class DataAugmentation(nn.Module):
    def __init__(self, cfg_aug=None, zero_mean_norm=True) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            RandomResizedCrop(p=cfg_aug['ResizedCrop']['p'],
                              size=(cfg_aug['ResizedCrop']['size'], cfg_aug['ResizedCrop']['size']),
                              scale=(cfg_aug['ResizedCrop']['scale'][0], cfg_aug['ResizedCrop']['scale'][1])),
            RandomHorizontalFlip(p=cfg_aug['HorizontalFlip']['p']),
            RandomVerticalFlip(p=cfg_aug['VerticalFlip']['p']),
            RandomBrightness(p=cfg_aug['Brightness']['p'],
                             brightness=(cfg_aug['Brightness']['val'], cfg_aug['Brightness']['val'])),
            RandomGaussianNoise(p=cfg_aug['GaussianNoise']['p'],
                                mean=cfg_aug['GaussianNoise']['mu'],
                                std=cfg_aug['GaussianNoise']['scale']),
            RandomGaussianBlur(p=cfg_aug['GaussianBlur']['p'],
                                 kernel_size=(cfg_aug['GaussianBlur']['kernel_size'],
                                              cfg_aug['GaussianBlur']['kernel_size']),
                                 sigma=(cfg_aug['GaussianBlur']['sigma'], cfg_aug['GaussianBlur']['sigma'])),
            RandomErasing(p=cfg_aug['RandomErasing']['p'],
                            scale=(cfg_aug['RandomErasing']['scale'][0], cfg_aug['RandomErasing']['scale'][1]),
                            ratio=(cfg_aug['RandomErasing']['ratio'][0], cfg_aug['RandomErasing']['ratio'][1]))
        )

        self.n_views = 2
        self.zero_mean_norm = zero_mean_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, z, h, w = x.size()
        x = x.view(b, -1, h, w)
        views = []
        for i in range(self.n_views):
            views.append(self.transforms(x).view(b, t, c, z, h, w))
        views = torch.stack(views, dim=0)  # (n_views, b, t, c, z, h, w)
        views = views.view(-1, *views.shape[2:])  # (n_views*b, t, c, z, h, w)

        del x
        torch.cuda.empty_cache()

        if self.zero_mean_norm:
            return 2*views - 1

        return views
