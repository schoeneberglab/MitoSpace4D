import torch
from torch import nn
from kornia.augmentation import RandomResizedCrop, RandomGaussianNoise, RandomGaussianBlur,\
    RandomErasing, RandomHorizontalFlip3D, RandomVerticalFlip3D, \
    RandomDepthicalFlip3D, RandomRotation3D, RandomAffine3D

def _disable_kornia_features(module: nn.Module) -> None:
            for m in module.modules():
                if hasattr(m, "disable_features"):
                    try:
                        m.disable_features = True
                    except Exception:
                        pass

class RandomTimeFlip(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.flipper = RandomDepthicalFlip3D(p=self.p)
        _disable_kornia_features(self.flipper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, z, h, w = x.size()
        x = x.permute(0, 3, 2, 1, 4, 5)
        x = x.reshape(b*z, c, t, h, w)
        x = self.flipper(x)
        x = x.view(b, z, c, t, h, w)
        x = x.permute(0, 3, 2, 1, 4, 5)
        return x

class RandomExchangeFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x

        device = x.device
        dims = (2, 3, 4)  # z, h, w

        if self.p >= 1.0:
            sel = torch.tensor([True, True, True], device=device)
        else:
            sel = torch.rand(3, device=device) < self.p

        if not torch.any(sel):
            return x

        chosen_dims = []
        shifts = []
        for i, d in enumerate(dims):
            if sel[i]:
                mid = x.size(d) // 2
                if mid:
                    chosen_dims.append(d)
                    shifts.append(mid)

        if chosen_dims:
            x = torch.roll(x, shifts=tuple(shifts), dims=tuple(chosen_dims))
            
        return x

class RandomTimeMask(nn.Module):
    def __init__(self, p=0.5, n_views=2) -> None:
        super().__init__()
        self.p = p  # p actually doesn't matter here; only when it's zero, we don't apply
        self.n_views = n_views

        self.time_delay = torch.Tensor([16, 14, 12, 10, 8, 6, 4, 2, 0])
        self.probs_time_delay = torch.Tensor([0.03, 0.038, 0.047, 0.059, 0.074, 0.092, 0.115, 0.144, 0.401])

        self.clip_len = torch.Tensor([20, 19, 18, 17, 16, 15, 14, 13, 12, 11])
        self.clip_len_probs = torch.Tensor([0.501, 0.116, 0.092, 0.074, 0.059, 0.047, 0.038, 0.030, 0.024, 0.019])

    def forward(self, x: torch.Tensor):
        if self.p == 0:
            return [x] * self.n_views

        idx = torch.multinomial(self.clip_len_probs, 1)
        clip_len = int(self.clip_len[idx].item())

        if x.size(1) > clip_len:
            start = torch.randint(0, (x.size(1) - clip_len), (1,), device=x.device).item()
        else:
            start = 0
        end = start + clip_len

        # build mask for first clip
        mask = torch.zeros_like(x, device=x.device)
        mask[:, start:end, :, :, :] = 1
        
        x_1 = x * mask

        if self.n_views == 1:
            return [x_1]
        elif self.n_views == 2:

            td_probs = torch.as_tensor(self.probs_time_delay, dtype=torch.float, device=x.device)
            td_vals = torch.as_tensor(self.time_delay, dtype=torch.long, device=x.device)
            td_idx = torch.multinomial(td_probs, 1, replacement=True).item()
            time_delay = int(td_vals[td_idx].item())
            time_delay = min(time_delay, clip_len)

            mask_2 = mask.clone()
            if time_delay > 0:
                mask_2[:, start:start + time_delay, :, :, :] = 0
                mask_2[:, end:end + time_delay, :, :, :] = 1

            x_2 = x * mask_2

            return [x_1, x_2]
        else:
            raise NotImplementedError("Only n_views=1 or 2 is supported for RandomTimeMask.")


class RandomBrightness(nn.Module):
    def __init__(self, 
                 p=0.5, 
                 lower=0.8, 
                 upper=1.2,
                 add_lower=-0.05, 
                 add_upper=0.05, 
                 per_channel=True) -> None:

        super().__init__()
        self.p = float(p)
        self.mult_lower = float(lower)
        self.mult_upper = float(upper)
        self.add_lower = float(add_lower)
        self.add_upper = float(add_upper)
        self.per_channel = bool(per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, c, z, h, w)
        B, T, C = x.shape[:3]

        if self.per_channel:
            shape = (B, 1, C, 1, 1, 1)
        else:
            shape = (B, 1, 1, 1, 1, 1)


        apply_mask = (torch.rand(shape, device=x.device) < self.p).to(x.dtype)
        mult_factors = torch.rand(shape, device=x.device, dtype=x.dtype) * (self.mult_upper - self.mult_lower) + self.mult_lower
        mult_factors = mult_factors * apply_mask + (1.0 * (1 - apply_mask))

        add_factors = torch.rand(shape, device=x.device, dtype=x.dtype) * (self.add_upper - self.add_lower) + self.add_lower
        add_factors = add_factors * apply_mask

        return (x * mult_factors) + add_factors

class DataAugmentation(nn.Module):
    def __init__(self, cfg_aug=None, zero_mean_norm=True, n_views=1) -> None:
        super().__init__()
        
        assert n_views in [1, 2], "Only one or two views are supported for now"
        
        self.n_views = n_views
        self.zero_mean_norm = zero_mean_norm

        self.temporal_transform_1 = RandomTimeMask(p=cfg_aug['RandomTimeMask']['p'], n_views=self.n_views)
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
            RandomGaussianNoise(p=cfg_aug['GaussianNoise']['p'],
                                mean=cfg_aug['GaussianNoise']['mu'],
                                std=cfg_aug['GaussianNoise']['scale']),
        )

        self.transforms_3d = nn.Sequential(
            RandomHorizontalFlip3D(p=cfg_aug['HorizontalFlip3D']['p']),
            RandomVerticalFlip3D(p=cfg_aug['VerticalFlip3D']['p']),
            RandomDepthicalFlip3D(p=cfg_aug['DepthicalFlip3D']['p']),
            RandomRotation3D(p=cfg_aug['RandomRotation3D']['p'],
                             degrees=cfg_aug['RandomRotation3D']['degrees']),
            RandomAffine3D(p=cfg_aug['RandomAffine3D']['p'],
                           degrees=cfg_aug['RandomAffine3D']['degrees'],
                           translate=(cfg_aug['RandomAffine3D']['translate'][0],
                                      cfg_aug['RandomAffine3D']['translate'][1],
                                      cfg_aug['RandomAffine3D']['translate'][2])),
            RandomExchangeFlip(p=cfg_aug['RandomExchangeFlip']['p']),
        )

        self.brightness_3d = RandomBrightness(p=cfg_aug['RandomBrightness']['p'],
                                              lower=cfg_aug['RandomBrightness']['lower'],
                                              upper=cfg_aug['RandomBrightness']['upper'],
                                              per_channel=cfg_aug['RandomBrightness']['per_channel'],
                                              )

        # Disable Kornia image-module features to avoid CPU detaches ¯\_(ツ)_/¯
        _disable_kornia_features(self.transforms_2d)
        _disable_kornia_features(self.transforms_3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b, t, c, z, h, w = x.size()
        views = self.temporal_transform_1(x) # (b, t, c, z, h, w), (b, t, c, z, h, w)
        assert len(views) == self.n_views, f"Number of views should be {self.n_views}"

        views = [view.reshape(b, -1, h, w) for view in views] 

        for i in range(self.n_views):
            views[i] = self.transforms_2d(views[i]).reshape(b, t, c, z, h, w)  

            views[i] = self.transforms_3d(views[i].reshape(b, t * c, z, h, w)).reshape(b, t, c, z, h, w) 
            views[i] = self.brightness_3d(views[i])  
            views[i] = self.temporal_transform_2(views[i])  

        views = torch.stack(views, dim=0)  
        views = views.reshape(-1, *views.shape[2:])  
        
        if self.zero_mean_norm:
            return 2 * views - 1  # scale to [-1, 1]
        
        return views
