import torch
import torch.nn as nn
from simclr.augmentations import DataAugmentation
from utils.utils import load_config
from autoencoder.models import MitoSpace3DAutoencoder


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
            nn.Conv3d(in_channels=input_dim, out_channels=self.init_conv_out_dim, kernel_size=(1, 3, 3),
                      stride=(1, 2, 2),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )

        self.cell_list = nn.ModuleList(
            [Conv3DLSTMCell(input_dim=self.init_conv_out_dim if i == 0 else hidden_dim[i - 1],
                            hidden_dim=hidden_dim[i],
                            kernel_size=kernel_size[i],
                            bias=bias) for i in range(num_layers)])
        kernel_sizes = [(3, 3, 3)] * (num_layers - 1) + [(1, 1, 1)]
        paddings = [(1, 1, 1)] * (num_layers - 1) + [(0, 0, 0)]
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
            c, d, h, w = cur_layer_input.size(1), cur_layer_input.size(2), cur_layer_input.size(3), cur_layer_input.size(4)
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
                 feat_dim=2048, cfg_aug=None, apply_aug=False):
        super(MitoSpace4DConvLSTM, self).__init__()

        self.out_dim = out_dim
        self.apply_aug = apply_aug

        self.decoder = MitoSpace3DAutoencoder().decoder
        cpkt_path = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/autoencoder/version_3017473/checkpoints/epoch=4-step=43060.ckpt'
        params_dict = torch.load(cpkt_path)
        decoder_param_dict = {k.replace('model.decoder.', ''): v for k, v in params_dict['state_dict'].items() if
                              'model.decoder' in k}
        self.decoder.load_state_dict(decoder_param_dict)

        self.augment_pipeline = DataAugmentation(cfg_aug, zero_mean_norm=True)

        self.net = Conv3DLSTM(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size,
                              num_layers=num_layers,
                              conv_reduction_factor=conv_reduction_factor, batch_first=True, bias=True,
                              return_all_layers=False)

        self.fc = nn.Linear(feat_dim, feat_dim)

        self.proj = nn.Sequential(nn.Linear(feat_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim),
                                  nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.augment_pipeline(x) if self.apply_aug else x  # (b, t, c, d, h, w)
        x = self.net(x)
        x = x[:, -1].flatten(start_dim=1)
        x = self.fc(x)
        out = self.proj(x)
        return x, out


if __name__ == "__main__":
    cfg = load_config("/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml")
    # Example usage
    in_channels = 2  # Assuming single-channel 3D data
    model = MitoSpace4DConvLSTM(in_channels=in_channels, out_dim=512, cfg_aug=cfg['data_params']['transforms']).cuda()

    # Create a sample input tensor with shape (batch_size, sequence_length, in_channels, depth, height, width)
    input_tensor = torch.randn(3, 20, 2, 60, 256, 256).cuda()  # Example input tensor

    # Forward pass
    output, _ = model(input_tensor)
    print(output.shape)  # Should output (batch_size, num_classes)
