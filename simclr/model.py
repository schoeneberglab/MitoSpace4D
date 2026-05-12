import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.model import load_model
from simclr.augmentations import DataAugmentation
from utils.utils import load_config


class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic3DBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = (
            nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class Lightweight3DResNet(nn.Module):
    def __init__(
        self,
        embedding_size=2048,
        cfg=None,
        apply_aug=False,
        decoder_checkpoint_path=None,
    ) -> None:

        super(Lightweight3DResNet, self).__init__()
        self.apply_aug = apply_aug
        self.augment_pipeline = DataAugmentation(
            cfg["data_params"]["transforms"], zero_mean_norm=True
        )
        self._with_decoder = False

        self._channels = cfg["model_params"]["channels"]
        print(f"Using channels: {self._channels} for input.")

        in_channels = len(self._channels)
        self._channels = torch.tensor(self._channels).to(torch.int32)

        if decoder_checkpoint_path:
            self.with_decoder = True

            ae_model = load_model(decoder_checkpoint_path, device="cuda")
            self.decoder = ae_model.decoder
            self.decoder.eval()

            for param in self.decoder.parameters():
                param.requires_grad = False

            del ae_model
            self.decoder.to("cuda")
            print(f"Loaded decoder from: {decoder_checkpoint_path}")

        self.augment_pipeline.to("cuda")

        stem = nn.Sequential(
            nn.Conv3d(
                in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1),
        )

        self.resnet = nn.Sequential(
            stem,
            self._make_layer(16, 32, num_blocks=2, stride=2),
            self._make_layer(32, 64, num_blocks=2, stride=2),
            self._make_layer(64, 128, num_blocks=2, stride=2),
            self._make_layer(128, 512, num_blocks=2, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
            bidirectional=cfg["model_params"]["bidirectional"],
        )

        self.fc = (
            nn.Linear(1024 * 2, embedding_size)
            if cfg["model_params"]["bidirectional"]
            else nn.Linear(1024, embedding_size)
        )

        self.proj = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(Basic3DBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Basic3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def scramble_time(self, x):
        permutation = torch.randperm(x.size(1))
        x = x[:, permutation, ...]
        return x

    def forward(self, x, get_resnet_feats=False):
        if self._with_decoder:
            with torch.no_grad():
                # x: (b, t, c, d, h, w)
                b = x.size(0)
                micro_bs = 2

                decoded_chunks = []
                for i in range(0, b, micro_bs):
                    chunk = x[i : i + micro_bs]
                    out = self.decoder(chunk)
                    decoded_chunks.append(out)
                x = torch.cat(decoded_chunks, dim=0)

        if self.apply_aug:
            with torch.no_grad():
                x = self.augment_pipeline(x)
        else:
            x = 2 * x - 1  # Scale to [-1, 1]

        x = x[:, :, self._channels, ...]

        batch_size, time_steps, channels, depth, height, width = x.size()

        x = x.view(batch_size * time_steps, channels, depth, height, width)

        x = self.resnet(x)
        x_resnet = x.view(batch_size, time_steps, -1)  # Reshape for LSTM

        x, _ = self.lstm(x_resnet)
        # x = x[:, -1, :] # Use the last timestep from LSTM output

        x = x.view(batch_size * time_steps, x.size(-1))  # (b * t, d)
        x = self.fc(x)
        out = self.proj(x)

        x = x.view(batch_size, time_steps, -1)  # (b, t, d)

        if get_resnet_feats:
            return x, x_resnet, out
        else:
            return x, out


if __name__ == "__main__":
    cfg = load_config("/u/earkfeld/MitoSpace4D/simclr/config.yaml")
    model = Lightweight3DResNet(
        embedding_size=2048, cfg_aug=cfg["data_params"]["transforms"], apply_aug=True
    ).cuda()

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    sample_input = torch.randn(1, 20, 2, 30, 256, 256).cuda()  # Example input
    output = model(sample_input)
    print(output.shape)
