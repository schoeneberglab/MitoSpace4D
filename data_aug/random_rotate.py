import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor

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


class RandomRotation(torch.nn.Module):
    """Rotate the image by angle.
	If the image is torch Tensor, it is expected
	to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

	Args:
		degrees (sequence or number): Range of degrees to select from.
			If degrees is a number instead of sequence like (min, max), the range of degrees
			will be (-degrees, +degrees).
		interpolation (InterpolationMode): Desired interpolation enum defined by
			:class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
			If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
			The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
		expand (bool, optional): Optional expansion flag.
			If true, expands the output to make it large enough to hold the entire rotated image.
			If false or omitted, make the output image the same size as the input image.
			Note that the expand flag assumes rotation around the center and no translation.
		center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
			Default is the center of the image.
		fill (sequence or number): Pixel fill value for the area outside the rotated
			image. Default is ``0``. If given a number, the value is used for all bands respectively.

	.. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

	"""

    def __init__(self, p, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        super().__init__()
        _log_api_usage_once(self)

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        self.p = p

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

		Returns:
			float: angle parameter to be passed to ``rotate`` for random rotation.
		"""
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, img):
        """
		Args:
			img (PIL Image or Tensor): Image to be rotated.

		Returns:
			PIL Image or Tensor: Rotated image.
		"""
        if np.random.uniform() > self.p:
            return img
        fill = self.fill
        time, z, _, _ = img.size()
        #if isinstance(img, Tensor):
            #random_z = random.randint(0, z - 1)

            # we multiply by 1 because right now we only we have either mitotrack or tmrm; if we have more channels (
            # both Mitotracker and TMRM), we need to change this to 2 and have separate fill values for each channel
            #fill = [float(torch.quantile(img[:, random_z], 0.05))] * 1
        fill = 0.
        angle = self.get_params(self.degrees)

        try:
            return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill)
        except:
            return img

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += ")"
        return format_string
