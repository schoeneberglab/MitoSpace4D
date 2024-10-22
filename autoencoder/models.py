import torch
import torch.nn as nn

class MitoSpace3DEncoder(nn.Module):
    def __init__(self, input_dim=1):
        super(MitoSpace3DEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels = input_dim, 
                      out_channels = 2, 
                      kernel_size = (3, 3, 3), 
                      stride = (1, 2, 2), 
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU(inplace = True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels = 2, 
                      out_channels = 4, 
                      kernel_size = (3, 3, 3), 
                      stride = (2, 2, 2), 
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU(inplace = True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels = 4, 
                      out_channels = 16, 
                      kernel_size = (3, 3, 3), 
                      stride = (2, 2, 2), 
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU(inplace = True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels = 16, 
                      out_channels = 256, 
                      kernel_size = (3, 3, 3), 
                      stride = (1, 1, 1), 
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU(inplace = True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels = 256, 
                      out_channels = 64, 
                      kernel_size = (3, 3, 3), 
                      stride = (1, 1, 1), 
                      padding = (1, 1, 1),
                      bias = True),
            nn.ReLU(inplace = True)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels = 64, 
                               out_channels = 8, 
                               kernel_size = (3, 3, 3), 
                               stride = (1, 2, 2), 
                               padding = (1, 1, 1), 
                               output_padding = (0, 1, 1),
                               bias=True),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        assert(len(x.shape) == 6)
        batch_size, timesteps, channels, stacks, height, width = x.shape
        x = x.view(batch_size * timesteps * channels, stacks, height, width).unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        _, _, new_stacks, new_height, new_width = x.shape
        x = x.view(batch_size, timesteps, channels, -1, new_stacks, new_height, new_width)

        return x

class MitoSpace3DDecoder(nn.Module):
    def __init__(self):
        super(MitoSpace3DDecoder, self).__init__()

        self.deconv1 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16, out_channels=4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=4, out_channels=1, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        assert(len(x.shape) == 7)
        batch_size, timesteps, channels, extra_channel, depth, height, width = x.shape
        x = x.view(batch_size * timesteps * channels, extra_channel, depth, height, width)

        x = self.deconv1(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = x.view(batch_size, 20, 2, -1, 60, 256, 256)

        x = x.squeeze(3)

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
    input_data = torch.randn(2, 20, 2, 60, 256, 256).cuda()
    autoencoder = MitoSpace3DAutoencoder().cuda()

    print(sum(p.numel() for p in autoencoder.parameters() if p.requires_grad))
    output = autoencoder(input_data)
    assert input_data.shape == output.shape
    print('Output shape', output.shape)
