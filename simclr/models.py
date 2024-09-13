from typing import Tuple

import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models.video import r3d_18
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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


class InterMixConv3DLSTM(nn.Module):
    def __init__(self, in_channels, out_dim, lstm_hidden_size=512, lstm_num_layers=2):
        super(InterMixConv3DLSTM, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(2, 2, kernel_size=(3, 5, 5), stride=(1, 2, 2), bias=False)
        self.bn2 = nn.BatchNorm3d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(2, 2, kernel_size=(7, 11, 11), stride=(1, 2, 2), bias=False)
        self.bn3 = nn.BatchNorm3d(2)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.resnet_layer1 = self._make_layer(2, 1, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.lstm1 = nn.LSTM(input_size=2925, hidden_size=2925, num_layers=lstm_num_layers, batch_first=True,
                             bidirectional=True)

        self.resnet_layer2 = self._make_layer(1, 1, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.lstm2 = nn.LSTM(input_size=8, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True,
                             bidirectional=True)

        self.resnet_layer3 = self._make_layer(2, 1, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.lstm3 = nn.LSTM(input_size=8, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True,
                             bidirectional=True)

        self.resnet_layer4 = self._make_layer(2, 1, stride=(2, 2, 2), kernel_size=(3, 3, 3))
        self.lstm4 = nn.LSTM(input_size=8, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True,
                             bidirectional=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.lstm_final = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                                  batch_first=True,
                                  bidirectional=True)

        self.fc = nn.Linear(lstm_hidden_size * 2, out_dim)

        # projection head
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False), nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim, bias=True))

    def _make_layer(self, in_channels, out_channels, stride, kernel_size):
        layers = []
        layers.append(BasicBlock3D(in_channels, in_channels * 2, stride, kernel_size=kernel_size))
        layers.append(BasicBlock3D(in_channels * 2, out_channels, kernel_size=(3, 3, 3)))
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

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.maxpool(x)

        x = self.resnet_layer1(x)
        c, z, h, w = x.size()[-4:]

        x = x.view(batch_size, seq_length, -1)

        x, _ = self.lstm1(x)

        x = x.contiguous().view(batch_size * seq_length, -1)
        x = x.view(-1, c, z, h, w)
        x = self.resnet_layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.maxpool(x)

        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = x.view(batch_size, seq_length, -1)

        return x


class Conv3DLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(Conv3DLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in kernel_size)
        self.bias = bias

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device))


class Conv3DLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, conv_reduction_factor, batch_first=False,
                 bias=True, return_all_layers=False):
        super(Conv3DLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        conv_reduction_factor = self._extend_for_multilayer(conv_reduction_factor, num_layers - 1)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.init_conv_out_dim = 1
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.conv_reduction_factor = conv_reduction_factor
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels=input_dim, out_channels=self.init_conv_out_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )

        self.cell_list = nn.ModuleList([Conv3DLSTMCell(input_dim=self.init_conv_out_dim if i == 0 else hidden_dim[i - 1],
                                                       hidden_dim=hidden_dim[i],
                                                       kernel_size=kernel_size[i],
                                                       bias=bias) for i in range(num_layers)])
        kernel_sizes = [(3, 3, 3)]*(num_layers-1)+[(1, 1, 1)]
        paddings = [(1, 1, 1)]*(num_layers-1)+[(0, 0, 0)]
        self.conv_list = nn.ModuleList([nn.Sequential(
            nn.Conv3d(in_channels=hidden_dim[i], out_channels=hidden_dim[i], kernel_size=kernel_sizes[i],
                      stride=conv_reduction_factor[i], padding=paddings[i]),
            nn.BatchNorm3d(hidden_dim[i]),
            nn.ReLU(inplace=True)
        ) for i in range(num_layers)])

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)

        b, t, c, d, h, w = input_tensor.size()
        input_tensor = input_tensor.reshape(b * t, c, d, h, w)
        input_tensor = self.initial_conv(input_tensor)
        _, c, d, h, w = input_tensor.size()

        input_tensor = input_tensor.view(b, -1, c, d, h, w)

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(d, h, w))

        cur_layer_input = input_tensor

        layer_idx = 0
        for conv, convlstmcell in zip(self.conv_list, self.cell_list):
            hid, cell = hidden_state[layer_idx]
            output_inner = []

            for t in range(cur_layer_input.size(1)):
                hid, cell = convlstmcell(cur_layer_input[:, t, :, :, :, :],
                                         [hid, cell])
                output_inner.append(hid)

            cur_layer_input = torch.stack(output_inner, dim=1)

            # downsampling the spatial dimensions; don't downsample for the last layer
            # if layer_idx < self.num_layers - 1:
            b, t, c, d, h, w = cur_layer_input.size()
            cur_layer_input = cur_layer_input.view(b * t, c, d, h, w)
            cur_layer_input = conv(cur_layer_input)
            d, h, w = cur_layer_input.size(2), cur_layer_input.size(3), cur_layer_input.size(4)
            cur_layer_input = cur_layer_input.view(b, t, c, d, h, w)

            layer_idx += 1

        return cur_layer_input

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
            # if i < self.num_layers - 1:
            image_size = tuple((s - 1) // self.conv_reduction_factor[i][j] + 1 for j, s in enumerate(image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (
                isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class MitoSpace4DConvLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=[1, 2, 4, 16, 32, 256], kernel_size=(3, 3, 3), num_layers=6,
                 conv_reduction_factor=[(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)], out_dim=512,
                 feat_dim=2048):
        super(MitoSpace4DConvLSTM, self).__init__()

        self.out_dim = out_dim

        self.net = Conv3DLSTM(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size,
                              num_layers=num_layers,
                              conv_reduction_factor=conv_reduction_factor, batch_first=True, bias=True,
                              return_all_layers=False)

        self.fc = nn.Linear(feat_dim, feat_dim)

        self.proj = nn.Sequential(nn.Linear(feat_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim),
                                  nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim, bias=True))

    def forward(self, x):
        x = x.transpose(1, 2)  # b, c, t, z, h, w -> b, t, c, z, h, w
        x = self.net(x)
        x = x[:, -1].flatten(start_dim=1)
        x = self.fc(x)
        out = self.proj(x)
        return x, out


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
    in_channels = 2  # Assuming single-channel 3D data
    num_classes = 10  # Number of output classes, adjust as necessary
    model = MitoSpace4DConvLSTM(in_channels=in_channels, out_dim=512).cuda()

    # Create a sample input tensor with shape (batch_size, sequence_length, in_channels, depth, height, width)
    input_tensor = torch.randn(6, 2, 20, 60, 256, 256).cuda()  # Example input tensor

    # Forward pass
    output = model(input_tensor)
    print(output.shape)  # Should output (batch_size, num_classes)
