import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

try:
    import accimage
except ImportError:
    accimage = None

from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be a sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomAffine(torch.nn.Module):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x-axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x-axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            an x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(
            self,
            p,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
            center=None,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.p = p

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation
        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Affine transformed image.
        """

        if np.random.uniform() > self.p:
            return img

        fill = self.fill

        if len(img.size()) == 4:
            time, z, height, width = img.size()
        else:
            bs, time, z, height, width = img.size()
        # if isinstance(img, Tensor):
        # random_z = random.randint(0, z - 1)

        # we multiply by 1 because right now we only we have either mitotrack or tmrm; if we have more channels (
        # both Mitotracker and TMRM), we need to change this to 2 and have separate fill values for each channel
        # if isinstance(fill, (int, float)):
        #    fill = [float(torch.quantile(img[:, random_z], 0.15))] * 1
        # else:
        #    fill = [float(torch.quantile(img[:, random_z], 0.15))] * 1
        fill = 0

        img_size = [width, height]  # flip for keeping BC on get_params call

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        try:
            # compress dimensions if more than 4
            if len(img.size()) == 4:
                return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
            elif len(img.size()) == 5:
                img = img.view(-1, height, width)
                img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
                return img.view(bs, time, z, height, width)
        except:
            return img

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", scale={self.scale}" if self.scale is not None else ""
        s += f", shear={self.shear}" if self.shear is not None else ""
        s += f", interpolation={self.interpolation.value}" if self.interpolation != InterpolationMode.NEAREST else ""
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += f", center={self.center}" if self.center is not None else ""
        s += ")"

        return s


class RandomAffineGPU(torch.nn.Module):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x-axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x-axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            an x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(
            self,
            p,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
            center=None,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.p = p

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation
        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        if np.random.uniform() > self.p:
            return img

        fill = 0.
        bs, height, width = img.size()

        img_size = [width, height]  # flip for keeping BC on get_params call

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        try:
            return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
        except:
            return img

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", scale={self.scale}" if self.scale is not None else ""
        s += f", shear={self.shear}" if self.shear is not None else ""
        s += f", interpolation={self.interpolation.value}" if self.interpolation != InterpolationMode.NEAREST else ""
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += f", center={self.center}" if self.center is not None else ""
        s += ")"

        return s


class RandomAffineGPUWrapper(RandomAffineGPU):
    def __init__(self, p, degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST,
                 fill=0, center=None):
        super().__init__(p, degrees, translate, scale, shear, interpolation, fill, center)

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Affine transformed image.
        """

        img = img.cuda()
        time, z, height, width = img.size()
        img = img.view(-1, height, width)
        img = super().forward(img)
        img = img.view(time, z, height, width)
        img = img.cpu()

        return img
