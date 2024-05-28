from typing import Tuple

import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models.video import r3d_18
import torch
import torch.nn.functional as F


class ResNetSimCLR3D(nn.Module):
    def __init__(self, out_dim: int, in_channels: int = 100, pretrained: bool = False):
        super(ResNetSimCLR3D, self).__init__()

        self.pre = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1)
        self.net = r3d_18(pretrained=pretrained, num_classes=out_dim)

        # projection head
        self.g = nn.Sequential(nn.Linear(out_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, out_dim, bias=True))

    def forward(self, x):
        x = self.pre(x.permute(0, 2, 1, 3, 4))  # BxCxTxHxW
        x = self.net(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out


if __name__ == "__main__":
    model = ResNetSimCLR3D(base_model='resnet50', out_dim=512)
    x = torch.rand(2, 2, 32, 32)
    print(model(x).shape)
