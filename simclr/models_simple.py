import torch
import torch.nn as nn
from simclr.augmentations import DataAugmentation
from utils.utils import load_config
import torch.nn.functional as F
from autoencoder.autoencoder_runner import AutoEncoderRunner
from autoencoder.autoencoder_models import MitoSpace3DAutoencoder


class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic3DBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class Lightweight3DResNet(nn.Module):
    def __init__(self, embedding_size=2048, cfg_aug=None, apply_aug=False, decoder_checkpoint_path=None):
        super(Lightweight3DResNet, self).__init__()

        self.apply_aug = apply_aug
        self.augment_pipeline = DataAugmentation(cfg_aug, zero_mean_norm=True)
        # dec_checkpoint_path = "/tscc/nfs/home/d5agarwal/projects/MitoSpace4D/autoencoder/lightning_logs/final_training_sdsc_16_nodes_low_lr_low_gamma/lightning_logs/version_3178623/checkpoints/epoch=8-step=6462.ckpt"
        self.with_decoder = decoder_checkpoint_path is not None
        
        if self.with_decoder:
            decoder_model = MitoSpace3DAutoencoder()
            self.decoder = AutoEncoderRunner.load_from_checkpoint(decoder_checkpoint_path, model=decoder_model)
            self.decoder = self.decoder.model.decoder
            self.decoder.eval()

            # Freeze decoder parameters
            for param in self.decoder.parameters():
                param.requires_grad = False

        # Initial layer: modify for 2-channel input
        self.stem = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        )

        # Define 3D ResNet layers with reduced channels and depth
        self.layer1 = self._make_layer(16, 32, num_blocks=2, stride=2)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 512, num_blocks=2, stride=2)

        # Adaptive average pooling to reduce spatial and depth dimensions
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # BiLSTM for temporal encoding
        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)

        # Final fully connected layer for embedding
        self.fc = nn.Linear(1024 * 2, embedding_size)

        self.proj = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True), nn.Linear(512, 512, bias=True))

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

    def forward(self, x):
    
        with torch.no_grad():
            if self.with_decoder:
                x = self.decoder(x)
            #x = self.scramble_time(x)
            x = self.augment_pipeline(x) if self.apply_aug else 2*x-1  # (b, t, c, d, h, w)

        batch_size, time_steps, channels, depth, height, width = x.size()

        # Reshape for 3D convolution
        x = x.view(batch_size * time_steps, channels, depth, height, width)

        # Forward pass through 3D ResNet layers
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Average pool and reshape
        x = self.avgpool(x)
        x = x.view(batch_size, time_steps, -1)  # Reshape for LSTM

        # Forward pass through BiLSTM
        x, _ = self.lstm(x)

        # Use the last LSTM output
        # x = x[:, -1, :]

        # Pass all the timesteps to the final embedding to get the temporal embeddings
        b, t, d = x.size()
        x = x.reshape(-1, d) # Flatten the time dimension

        # Final embedding
        x = self.fc(x)

        # Projection head
        out = self.proj(x)

        x = x.reshape(b, t, -1)
        out = out.reshape(b, t, -1)[:, -1]

        return x, out


if __name__ == '__main__':
    cfg = load_config("/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml")
    # Initialize model and print the output shape
    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                 apply_aug=True).cuda()
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    sample_input = torch.randn(1, 20, 2, 30, 256, 256).cuda()  # Example input
    output = model(sample_input)
    print(output.shape)  # Should be (batch_size, embedding_size)
