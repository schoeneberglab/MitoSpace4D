import math
import numpy as np
import torch
import torch.nn as nn
from einops import einops

from simclr.augmentations import DataAugmentation
from utils.utils import load_config


def get_sinusoidal_embedding(seq_len, dim, device):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class MitoSpace4DTransformer(nn.Module):
    def __init__(self, out_dim=512, feat_dim=2048, patch_size=(2, 6, 32, 32), hidden_dim=512, nheads=8, num_layers=5,
                 cfg_aug=None, apply_aug=False):
        super(MitoSpace4DTransformer, self).__init__()

        self.patch_size = patch_size  # (t, d, h, w)
        self.out_dim = out_dim
        self.apply_aug = apply_aug

        self.augment_pipeline = DataAugmentation(cfg_aug, zero_mean_norm=True)

        self.conv = nn.Sequential(nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1))

        self.embed = nn.Linear(np.prod(patch_size), hidden_dim)

        num_patches = (20 // self.patch_size[0]) * (60 // self.patch_size[1]) * (256 // self.patch_size[2]) * (
                    256 // self.patch_size[3])
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, feat_dim)
        self.proj = nn.Sequential(nn.Linear(feat_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim),
                                  nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim, bias=True))

    def patchify_and_embed(self, x):
        x = einops.rearrange(x, 'b (t p1) c (d p2) (h p3) (w p4) -> b (t d h w) (p1 p2 p3 p4 c)',
                             p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2], p4=self.patch_size[3])
        return self.embed(x)

    def forward(self, x):
        x = self.augment_pipeline(x) if self.apply_aug else x  # (b, t, c, d, h, w)

        b, t, c, d, h, w = x.size()
        x = x.view(-1, *x.shape[2:])  # (b*t, c, d, h, w)
        x = self.conv(x)
        x = x.view(b, t, *x.shape[1:])  # (b, t, c, d, h, w)
        x = self.patchify_and_embed(x)  # (b, num_tokens, dim)

        x += self.pos_embedding  # (b, num_tokens, dim)

        x = x.permute(1, 0, 2)  # (num_tokens, b, dim)
        x = self.transformer_encoder(x)

        x = x.mean(dim=0)

        x = self.fc(x)
        out = self.proj(x)
        return x, out


if __name__ == "__main__":
    cfg = load_config("/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml")
    # Example usage
    in_channels = 2  # Assuming single-channel 3D data
    model = MitoSpace4DTransformer(cfg_aug=cfg['data_params']['transforms']).cuda()

    # Create a sample input tensor with shape (batch_size, sequence_length, in_channels, depth, height, width)
    input_tensor = torch.randn(6, 20, 2, 60, 256, 256).cuda()  # Example input tensor

    # Forward pass
    output, _ = model(input_tensor)
    print(output.shape)
