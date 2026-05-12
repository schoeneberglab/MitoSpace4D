import torch
import torch.nn as nn

def conv3x3(in_c, out_c, stride=1, padding=1, padding_mode="reflect", bias=False):
    return nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=padding, padding_mode=padding_mode, bias=bias)


def conv1x1(in_c, out_c, stride=1, bias=False):
    return nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=bias)

def load_model(checkpoint_path: str, device: torch.device):
    model = MitoSpace3DAutoencoder(input_dim=1, latent_dim=2, output_dim=1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model


class ResDown3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, padding_mode="reflect", norm=nn.BatchNorm3d, negative_slope=0.1):
        super().__init__()
        self.stride = stride
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

        self.conv1 = conv3x3(in_c, out_c, stride=stride, padding=1, padding_mode=padding_mode, bias=False)
        self.bn1 = norm(out_c)
        self.conv2 = conv3x3(out_c, out_c, stride=1, padding=1, padding_mode=padding_mode, bias=False)
        self.bn2 = norm(out_c)

        if stride != 1 or in_c != out_c:
            self.proj = nn.Sequential(
                conv1x1(in_c, out_c, stride=stride, bias=False),
                norm(out_c)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.proj(identity)

        out = out + identity
        out = self.act(out)
        return out


class ResUp3D(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, padding_mode="reflect", norm=nn.BatchNorm3d, negative_slope=0.1):
        super().__init__()
        self.scale = scale_factor
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        self.ups = None
        if self.scale != 1:
            self.ups = nn.Upsample(scale_factor=self.scale, mode='trilinear', align_corners=False)

        self.conv1 = conv3x3(in_c, out_c, stride=1, padding=1, padding_mode=padding_mode, bias=False)
        self.bn1 = norm(out_c)
        self.conv2 = conv3x3(out_c, out_c, stride=1, padding=1, padding_mode=padding_mode, bias=False)
        self.bn2 = norm(out_c)

        proj = []
        if self.scale != 1:
            proj.append(nn.Upsample(scale_factor=self.scale, mode='trilinear', align_corners=False))
        if in_c != out_c:
            proj.append(conv1x1(in_c, out_c, stride=1, bias=False))
            proj.append(norm(out_c))
        self.proj = nn.Sequential(*proj) if proj else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.ups(x) if self.ups is not None else x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.proj(identity)

        out = out + identity
        out = self.act(out)
        return out

class MitoSpace3DEncoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=4):
        super(MitoSpace3DEncoder, self).__init__()

        self.block1 = ResDown3D(input_dim, 4, stride=2)  # downsample
        self.block2 = ResDown3D(4, 8, stride=1)
        self.block3 = ResDown3D(8, 8, stride=1)
        self.block4 = ResDown3D(8, 64, stride=2)  # downsample
        self.block5 = ResDown3D(64, 128, stride=1)
        self.block6 = ResDown3D(128, 64, stride=1)
        self.block7 = ResDown3D(64, 64, stride=1)
        self.block8 = ResDown3D(64, latent_dim, stride=1)

    def forward(self, x):
        # B, C, D, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        return x


class MitoSpace3DDecoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=2):
        super(MitoSpace3DDecoder, self).__init__()
        self.output_dim = output_dim

        self.up1 = ResUp3D(latent_dim, 64, scale_factor=1)
        self.up2 = ResUp3D(64, 64, scale_factor=1)
        self.up3 = ResUp3D(64, 128, scale_factor=1)
        self.up4 = ResUp3D(128, 64, scale_factor=1)

        self.up5 = ResUp3D(64, 8, scale_factor=2)

        self.up6 = ResUp3D(8, 8, scale_factor=1)
        self.up7 = ResUp3D(8, 4, scale_factor=1)

        self.up8 = ResUp3D(4, 4, scale_factor=2)
        self.head = nn.Sequential(
            conv3x3(4, output_dim, stride=1, padding=1, padding_mode="reflect", bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (B, C, D, H, W)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.up8(x)
        x = self.head(x)
        return x


class MitoSpace3DAutoencoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=2, output_dim=1):
        super(MitoSpace3DAutoencoder, self).__init__()
        self.encoder = MitoSpace3DEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = MitoSpace3DDecoder(latent_dim=latent_dim, output_dim=output_dim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


if __name__ == '__main__':
    batch_size = 1
    encoder = MitoSpace3DEncoder(latent_dim=2).cuda()
    decoder = MitoSpace3DDecoder(latent_dim=2, output_dim=1).cuda()

    # (B, C, D, H, W)
    x = torch.randn(batch_size, 1, 64, 256, 256).cuda()

    z = encoder(x)
    print(z.dtype)
    y = decoder(z)

    print(f'Input shape: {x.shape}, size: {x.element_size() * x.nelement() / (1024 ** 2):.2f} MB')
    print(f'Latent shape: {z.shape}, size: {z.element_size() * z.nelement() / (1024 ** 2):.2f} MB')
    print(f'Output shape: {y.shape}, size: {y.element_size() * y.nelement() / (1024 ** 2):.2f} MB')
    print(f'Compression Factor: {x.element_size() * x.nelement() / (z.element_size() * z.nelement()):.2f}x')

    # Print the size of the full autoencoder model
    autoencoder = MitoSpace3DAutoencoder(input_dim=1, latent_dim=2, output_dim=1).cuda()
    # Note: torchsummary has issues with reflect padding, so we use a direct forward pass instead
    print("\n=== Model Summary ===")
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")

    with torch.no_grad():
        test_out = autoencoder(x)
        print(f"Forward pass successful: {x.shape} -> {test_out.shape}")
