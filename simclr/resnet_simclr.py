from typing import Tuple

import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models.resnet import resnet50, resnet18
import torch
import torch.nn.functional as F


class ResNetSimCLRv2(nn.Module):
    def __init__(self, base_model: str, out_dim: int, in_channels: int = 2, pretrained: bool = False):
        super(ResNetSimCLRv2, self).__init__()

        self.pre = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1)

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, out_dim, bias=True))

    def forward(self, x):
        x = self.pre(x)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model: str, out_dim: int, in_channels: int = 2, pretrained: bool = False) -> None:
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=pretrained, num_classes=1000 if pretrained else out_dim),
            "resnet50": models.resnet50(pretrained=pretrained, num_classes=1000 if pretrained else out_dim)}

        self.pre = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1)
        self.backbone = self._get_basemodel(base_model)
        feature_dim = self.backbone.fc.out_features

        # projection head
        self.proj = nn.Sequential(nn.Linear(feature_dim, 512, bias=False), nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True), nn.Linear(512, out_dim, bias=True))

    def _get_basemodel(self, model_name: str) -> nn.Module:
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise KeyError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pre(x)
        features = self.backbone(x)
        out = self.proj(features)

        return features, out


if __name__ == "__main__":
    model = ResNetSimCLR(base_model='resnet50', out_dim=512)
    x = torch.rand(2, 2, 32, 32)
    print(model(x).shape)
