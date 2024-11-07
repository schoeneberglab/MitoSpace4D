import os
import random
import numpy as np
import napari
import torch
from torch import nn
import os.path as osp
import torch
import torch.nn as nn
from torchsummary import summary

class MitoSpace3DEncoder(nn.Module):
    def __init__(self, input_dim=2):
        super(MitoSpace3DEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels = input_dim,
                      out_channels = 4,
                      kernel_size = (3, 3, 3),
                      stride = (2, 2, 2),
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels = 4,
                      out_channels = 8,
                      kernel_size = (3, 3, 3),
                      stride = (1, 1, 1),
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels = 8,
                      out_channels = 8,
                      kernel_size = (3, 3, 3),
                      stride = (1, 1, 1),
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels = 8,
                      out_channels = 64,
                      kernel_size = (3, 3, 3),
                      stride = (2, 2, 2),
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels = 64,
                               out_channels = 128,
                               kernel_size = (3, 3, 3),
                               stride = (1, 1, 1),
                               padding = (1, 1, 1),
                               bias=True),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv3d(in_channels=64,
                      out_channels=2,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        assert(len(x.shape) == 6)

        batch_size, timesteps, channels, stacks, height, width = x.shape
        x = x.view(batch_size * timesteps, channels, stacks, height, width)

        noisy_x = torch.nn.functional.interpolate(x,
                                                  scale_factor=(0.25, 0.25, 0.25),
                                                  mode='nearest',
                                                  align_corners=None,
                                                )

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        _, _, new_stacks, new_height, new_width = x.shape
        noisy_x = noisy_x.view(batch_size, timesteps, channels, new_stacks, new_height, new_width)
        x = x.view(batch_size, timesteps, channels, new_stacks, new_height, new_width)

        return torch.cat((x, noisy_x), dim = 2)

class MitoSpace3DDecoder(nn.Module):
    def __init__(self, output_dim=2):
        super(MitoSpace3DDecoder, self).__init__()

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=4,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=128,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=64,
                      out_channels=8,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv6 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=8,
                      out_channels=8,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv7 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=8,
                      out_channels=4,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

        self.deconv8 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=4,
                      out_channels=output_dim,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        assert(len(x.shape) == 6)

        batch_size, timesteps = x.shape[:2]

        x = x.view(batch_size*timesteps, 4, 15, 64, 64)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)

        _, _, new_stacks, new_height, new_width = x.shape
        x = x.view(batch_size, timesteps, 2, new_stacks, new_height, new_width)

        return x

class MitoSpace3DAutoencoder(nn.Module):
    def __init__(self):
        super(MitoSpace3DAutoencoder, self).__init__()
        self.encoder = MitoSpace3DEncoder()
        self.decoder = MitoSpace3DDecoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


if __name__ == '__main__':
    viewer = napari.Viewer()
    cpkt_path = '/home/dhruvagarwal/projects/Manav_MitoSpace/MitoSpace4D/autoencoder/lightning_logs/retrain_old_model_forgetting/lightning_logs/version_0/checkpoints/epoch=25-step=70122.ckpt'
    ae_model = MitoSpace3DAutoencoder()
    decoder = ae_model.decoder
    encoder = ae_model.encoder
    params_dict = torch.load(cpkt_path)
    print(params_dict)

    decoder_param_dict = {k.replace('model.decoder.', ''): v for k, v in params_dict['state_dict'].items() if 'model.decoder' in k}
    encoder_param_dict = {k.replace('model.encoder.', ''): v for k, v in params_dict['state_dict'].items() if 'model.encoder' in k}

    decoder.load_state_dict(decoder_param_dict)
    encoder.load_state_dict(encoder_param_dict)

    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data'
    drug = '20240830'

    filenames = os.listdir(os.path.join(data_dir, drug))
    idx = random.sample(range(len(filenames)), 1)[0]

    img_path = osp.join(data_dir, drug, filenames[idx])

    data = np.load(img_path)
    data = np.clip(data, 0, 5000)
    data = data / 5000
    data = data.astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(0)

    enc = encoder(data)
    dec = decoder(enc)
    print(dec)

    original = (data.squeeze().numpy()*255).astype(np.uint8)
    dec = (dec.squeeze().detach().numpy()*255).astype(np.uint8)

    viewer.add_image(original[:, 0], name=f"Original", translate=(0, 0), colormap='cyan')
    viewer.add_image(original[:, 1], name=f"Original", translate=(0, 256+10), colormap='cyan')
    viewer.add_image(dec[:, 0], name=f"Recon", translate=(256+10, 0), colormap='cyan')
    viewer.add_image(dec[:, 1], name=f"Recon", translate=(256+10, 256+10), colormap='cyan')

    napari.run()