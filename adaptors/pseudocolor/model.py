import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer3D(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for 3D volumes.
    Modulates features over (Depth, Height, Width).
    """

    def __init__(self, embedding_dim, num_channels):
        super(FiLMLayer3D, self).__init__()
        self.num_channels = num_channels
        self.adaptor = nn.Sequential(
            nn.Linear(embedding_dim, num_channels * 2),
            nn.ReLU()
        )

    def forward(self, x, embedding):
        # x: [Batch, Channels, Depth, Height, Width]
        # embedding: [Batch, Embedding_Dim]

        params = self.adaptor(embedding)
        gamma, beta = torch.split(params, self.num_channels, dim=1)

        # Reshape for 3D broadcasting: (B, C, 1, 1, 1)
        gamma = gamma.view(-1, self.num_channels, 1, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1, 1)

        return (1 + gamma) * x + beta


class DoubleConv3D(nn.Module):
    """(Conv3d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlockWithFiLM3D(nn.Module):
    """Upscaling then DoubleConv3D, modulated by SSL Embedding"""

    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        # Trilinear upsampling for 3D volumes
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLMLayer3D(embedding_dim, out_channels)
        self.double_conv = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, embedding):
        x1 = self.up(x1)

        # Handle padding for Depth, Height, Width differences
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.film(x, embedding)
        x = self.double_conv(x)
        return x


class ConditionedUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dim=1024):
        super(ConditionedUNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv3D(n_channels, 32)  # Reduced filters for memory safety
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(128, 256))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(256, 512))

        # Decoder
        self.up1 = UpBlockWithFiLM3D(512 + 256, 256, embedding_dim)
        self.up2 = UpBlockWithFiLM3D(256 + 128, 128, embedding_dim)
        self.up3 = UpBlockWithFiLM3D(128 + 64, 64, embedding_dim)
        self.up4 = UpBlockWithFiLM3D(64 + 32, 32, embedding_dim)

        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x, embedding):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)

        x = self.up1(x5, x4, embedding)
        x = self.up2(x, x3, embedding)
        x = self.up3(x, x2, embedding)
        x = self.up4(x, x1, embedding)

        return self.outc(x)