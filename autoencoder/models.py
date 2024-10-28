import torch
import torch.nn as nn

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
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels = 4,
                      out_channels = 8,
                      kernel_size = (3, 3, 3),
                      stride = (1, 1, 1),
                      padding = (1, 1, 1),
                      bias = True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels = 8,
                      out_channels = 8,
                      kernel_size = (3, 3, 3),
                      stride = (1, 1, 1),
                      padding = (1, 1, 1),
                      bias = True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels = 8,
                      out_channels = 64,
                      kernel_size = (3, 3, 3),
                      stride = (2, 2, 2),
                      padding = (1, 1, 1),
                      bias = True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels = 64,
                               out_channels = 128,
                               kernel_size = (3, 3, 3),
                               stride = (1, 1, 1),
                               padding = (1, 1, 1),
                               bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
        )

        self.conv8 = nn.Sequential(
            nn.Conv3d(in_channels=64,
                      out_channels=2,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
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
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=128,
                      out_channels=64,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=64,
                      out_channels=8,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv6 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=8,
                      out_channels=8,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv7 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=8,
                      out_channels=4,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv8 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=4,
                      out_channels=output_dim,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      bias=True)
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
    batch_size = 2
    input_data = torch.randn(batch_size, 20, 2, 60, 256, 256).cuda()
    autoencoder = MitoSpace3DAutoencoder().cuda()

    # print number of parameters
    print(sum(p.numel() for p in autoencoder.parameters()))

    output = autoencoder(input_data)
    assert input_data.shape == output.shape
    print(output.shape)