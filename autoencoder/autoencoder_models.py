import torch
import torch.nn as nn
from torchsummary import summary


def _gn(num_channels, num_groups=8):
    return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)


class ResBlock3D(nn.Module):
    """Lightweight residual block for detail refinement."""
    def __init__(self, channels, kernel_size=3, num_groups=8, padding_mode="reflect"):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            _gn(channels, num_groups),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size, padding=pad, padding_mode=padding_mode, bias=False),
            _gn(channels, num_groups),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size, padding=pad, padding_mode=padding_mode, bias=False),
        )
        # Zero-init last conv for stable residual learning
        nn.init.zeros_(self.block[-1].weight)

    def forward(self, x):
        return x + self.block(x)

class ConvBlock3D(nn.Module):
    """Conv -> GN -> SiLU with optional stride for downsampling."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, num_groups=8, padding_mode="reflect"):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, padding_mode=padding_mode, bias=False),
            _gn(out_ch, num_groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock3D(nn.Module):
    """Nearest upsample (anti-blur) + Conv -> GN -> SiLU."""
    def __init__(self, in_ch, out_ch, scale_factor=1, num_groups=8, padding_mode="reflect"):
        super().__init__()
        self.upsample = None
        if isinstance(scale_factor, tuple) or scale_factor != 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # preserves edges better than trilinear
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode, bias=False),
            _gn(out_ch, num_groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        return self.conv(x)


class MitoSpace3DEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=4, num_groups=8):
        super().__init__()

        # Downsample twice (2x total in each spatial dim), same as your original
        self.conv1 = ConvBlock3D(input_dim, 4, stride=2, num_groups=num_groups)
        self.refine1 = ResBlock3D(4, num_groups=num_groups)

        self.conv2 = ConvBlock3D(4, 8, stride=1, num_groups=num_groups)
        self.refine2 = ResBlock3D(8, num_groups=num_groups)

        self.conv3 = ConvBlock3D(8, 8, stride=1, num_groups=num_groups)
        self.refine3 = ResBlock3D(8, num_groups=num_groups)

        self.conv4 = ConvBlock3D(8, 64, stride=2, num_groups=num_groups)  # 2nd downsample
        self.refine4 = ResBlock3D(64, num_groups=num_groups)

        self.conv5 = ConvBlock3D(64, 128, stride=1, num_groups=num_groups)
        self.refine5 = ResBlock3D(128, num_groups=num_groups)

        self.conv6 = ConvBlock3D(128, 64, stride=1, num_groups=num_groups)
        self.refine6 = ResBlock3D(64, num_groups=num_groups)

        self.conv7 = ConvBlock3D(64, 64, stride=1, num_groups=num_groups)
        self.refine7 = ResBlock3D(64, num_groups=num_groups)

        self.conv8 = nn.Sequential(
            nn.Conv3d(64, latent_dim, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(latent_dim, num_groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        assert len(x.shape) == 6
        b, t, c, d, h, w = x.shape
        x = x.view(b * t, c, d, h, w)

        x = self.refine1(self.conv1(x))
        x = self.refine2(self.conv2(x))
        x = self.refine3(self.conv3(x))
        x = self.refine4(self.conv4(x))
        x = self.refine5(self.conv5(x))
        x = self.refine6(self.conv6(x))
        x = self.refine7(self.conv7(x))
        x = self.conv8(x)

        _, ch, d2, h2, w2 = x.shape
        x = x.view(b, t, ch, d2, h2, w2)
        return x


class MitoSpace3DDecoder(nn.Module):
    def __init__(self, latent_dim=4, output_dim=2, num_groups=8):
        super().__init__()
        self.output_dim = output_dim

        # Mirror the encoder’s channel flow; upsample only where encoder downsampled
        self.deconv1 = ConvBlock3D(latent_dim, 64, stride=1, num_groups=num_groups)
        self.refine1 = ResBlock3D(64, num_groups=num_groups)

        self.deconv2 = ConvBlock3D(64, 64, stride=1, num_groups=num_groups)
        self.refine2 = ResBlock3D(64, num_groups=num_groups)

        self.deconv3 = ConvBlock3D(64, 128, stride=1, num_groups=num_groups)
        self.refine3 = ResBlock3D(128, num_groups=num_groups)

        self.deconv4 = ConvBlock3D(128, 64, stride=1, num_groups=num_groups)
        self.refine4 = ResBlock3D(64, num_groups=num_groups)

        self.up5 = UpBlock3D(64, 8, scale_factor=(2, 2, 2), num_groups=num_groups)   # matches encoder conv1 downsample
        self.refine5 = ResBlock3D(8, num_groups=num_groups)

        self.deconv6 = ConvBlock3D(8, 8, stride=1, num_groups=num_groups)
        self.refine6 = ResBlock3D(8, num_groups=num_groups)

        self.deconv7 = ConvBlock3D(8, 4, stride=1, num_groups=num_groups)
        self.refine7 = ResBlock3D(4, num_groups=num_groups)

        self.up8 = UpBlock3D(4, output_dim, scale_factor=(2, 2, 2), num_groups=num_groups)  # matches encoder conv4 downsample
        self.out = nn.Sigmoid()  # keep same output scaling [0,1]

    def forward(self, x):
        assert len(x.shape) == 6
        b, t, c, d, h, w = x.shape
        x = x.view(b * t, c, d, h, w)

        x = self.refine1(self.deconv1(x))
        x = self.refine2(self.deconv2(x))
        x = self.refine3(self.deconv3(x))
        x = self.refine4(self.deconv4(x))
        x = self.refine5(self.up5(x))
        x = self.refine6(self.deconv6(x))
        x = self.refine7(self.deconv7(x))
        x = self.up8(x)
        x = self.out(x)

        _, ch, d2, h2, w2 = x.shape
        x = x.view(b, t, ch, d2, h2, w2)
        return x


class MitoSpace3DAutoencoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=4, output_dim=2, num_groups=8):
        super().__init__()
        self.encoder = MitoSpace3DEncoder(input_dim=input_dim, latent_dim=latent_dim, num_groups=num_groups)
        self.decoder = MitoSpace3DDecoder(latent_dim=latent_dim, output_dim=output_dim, num_groups=num_groups)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


if __name__ == '__main__':
    batch_size = 1
    autoencoder = MitoSpace3DAutoencoder().cuda()
    summary(autoencoder, input_size=(10, 4, 60, 256, 256), batch_size=batch_size)

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    x = torch.randn(batch_size, 20, 2, 60, 256, 256).cuda()
    z = encoder(x)
    y = decoder(z)

    print(f'Input shape: {x.shape}, size: {x.element_size() * x.nelement() / (1024 ** 2):.2f} MB')
    print(f'Latent shape: {z.shape}, size: {z.element_size() * z.nelement() / (1024 ** 2):.2f} MB')
    print(f'Output shape: {y.shape}, size: {y.element_size() * y.nelement() / (1024 ** 2):.2f} MB')
    print(f'Compression Factor: {x.element_size() * x.nelement() / (z.element_size() * z.nelement()):.2f}x')