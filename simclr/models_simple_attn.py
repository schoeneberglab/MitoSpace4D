import torch
import torch.nn as nn
from simclr.augmentations import DataAugmentation
from utils.utils import load_config
import torch.nn.functional as F
from autoencoder.autoencoder import AutoEncoderRunner
from autoencoder.models import MitoSpace3DAutoencoder


class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.4):
        super(Basic3DBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(out_channels)  # Replaced GN with IN
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(out_channels)  # Replaced GN with IN
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout layer

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.in1(self.conv1(x)))  # Apply IN after Conv
        out = self.dropout(out)  # Apply dropout after first convolution
        out = self.in2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class Lightweight3DResNet(nn.Module):
    def __init__(self, embedding_size=2048, cfg_aug=None, apply_aug=False, dropout_rate=0.4, decoder_checkpoint_path=None):
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
            nn.InstanceNorm3d(16),  # Replaced GN with IN
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        )

        # Define 3D ResNet layers with reduced channels and depth
        self.layer1 = self._make_layer(16, 32, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(128, 512, num_blocks=2, stride=2, dropout_rate=dropout_rate)

        # Adaptive average pooling to reduce spatial and depth dimensions
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # BiLSTM for temporal encoding
        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)

        # Attention mechanism
        self.attention_fc = nn.Linear(1024 * 2, 1)  # Bidirectional LSTM has hidden_size * 2
        self.softmax = nn.Softmax(dim=1)

        # Final fully connected layer for embedding
        self.fc = nn.Linear(1024 * 2, embedding_size)

        self.proj = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512, bias=True),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(Basic3DBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(Basic3DBlock(out_channels, out_channels, dropout_rate=dropout_rate))
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

        # Attention mechanism
        attention_scores = self.attention_fc(x)  # (batch_size, time_steps, 1)
        attention_scores = self.softmax(attention_scores)  # Normalize scores

        x = torch.sum(attention_scores * x, dim=1)  # (batch_size, hidden_size * 2)

        # Final embedding
        x = self.fc(x)

        # Projection head
        out = self.proj(x)

        return x, out
