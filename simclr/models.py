from typing import Tuple

import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models.video import r3d_18
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), kernel_size=(3, 3, 3)):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Small3DResNetLSTM(nn.Module):
    def __init__(self, in_channels, out_dim, lstm_hidden_size=512, lstm_num_layers=2):
        super(Small3DResNetLSTM, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=(7, 11, 11), stride=(1, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(2, 2, kernel_size=(7, 11, 11), stride=(1, 2, 2), bias=False)
        self.bn2 = nn.BatchNorm3d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(2, 8, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.layer2 = self._make_layer(8, 32, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.layer3 = self._make_layer(32, 256, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.layer4 = self._make_layer(256, 512, stride=(2, 2, 2), kernel_size=(3, 3, 3))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(lstm_hidden_size * 2, out_dim)

        # projection head
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False), nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim, bias=True))

    def _make_layer(self, in_channels, out_channels, stride, kernel_size):
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride, kernel_size=kernel_size))
        layers.append(BasicBlock3D(out_channels, out_channels, kernel_size=(3, 3, 3)))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, seq_length, channels, depth, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, depth, height, width)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = x.view(batch_size, seq_length, -1)  # Reshape to (batch_size, seq_length, feature_dim)

        lstm_out, _ = self.lstm(x)  # (2*N, T, 1024)
        lstm_out = lstm_out[:, -1, :]  # (2*N, 1024)

        feature = self.fc(lstm_out)  # (2*N, 512)
        out = self.proj(feature)  # (2*N, out_dim)

        return feature, out


class MitoSpace4D(nn.Module):
    def __init__(self, in_channels, out_dim, lstm_hidden_size=512, lstm_num_layers=2):
        super(MitoSpace4D, self).__init__()

        self.net = Small3DResNetLSTM(in_channels, out_dim, lstm_hidden_size, lstm_num_layers)

    def forward(self, x):
        x = x.unsqueeze_(2)  # Add a dummy channel dimension for the single-channel 3D data
        x = self.net(x)
        return x


if __name__ == "__main__":
    # Example usage
    in_channels = 1  # Assuming single-channel 3D data
    num_classes = 10  # Number of output classes, adjust as necessary
    model = Small3DResNetLSTM(in_channels=in_channels, out_dim=512)

    # Create a sample input tensor with shape (batch_size, sequence_length, in_channels, depth, height, width)
    input_tensor = torch.randn(2, 20, 1, 60, 32, 32)  # Example input tensor

    # Forward pass
    output = model(input_tensor)
    print(output.shape)  # Should output (batch_size, num_classes)
