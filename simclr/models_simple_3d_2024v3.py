import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from utils.utils import load_config
from simclr.augmentations import DataAugmentation
from autoencoder.autoencoder_runner import AutoEncoderRunner
# from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
from mitospace_autoencoder.utils import load_model

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
    def __init__(self, 
                 embedding_size=2048,
                 cfg=None, 
                 apply_aug=False,
                 decoder_checkpoint_path="/u/earkfeld/MitoSpace4D/mitospace_autoencoder/2024v3_autoencoder.pt",
                 device='cuda'
                #  decoder_checkpoint_path="/u/earkfeld/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
                 ) -> None:
        
        super(Lightweight3DResNet, self).__init__()

        print(f"Training 3D only!")

        self.apply_aug = apply_aug
        # self.augment_pipeline = DataAugmentation(cfg_aug, zero_mean_norm=True)
        self.augment_pipeline = DataAugmentation(cfg['data_params']['transforms'], zero_mean_norm=True)
        self._with_decoder = True if decoder_checkpoint_path is not None else False
        # self._n_channels = cfg['model_params']['in_channels']

        # Get the channels to use from config and convert to tensor for on-device indexing
        self._channels = cfg['model_params']['channels']
        print(f"Using channels: {self._channels} for input.")
        
        in_channels = len(self._channels)
        self._channels = torch.tensor(self._channels).to(torch.int32)
        
        if self._with_decoder:
            dec_checkpoint_path = decoder_checkpoint_path
            # ae_model = MitoSpace3DAutoencoder()
            # ae_model = AutoEncoderRunner.load_from_checkpoint(dec_checkpoint_path, model=ae_model)
            # self.decoder = ae_model.model.decoder

            ae_model = load_model(dec_checkpoint_path, device=device)
            self.decoder = ae_model.decoder
            self.decoder.eval()

            del ae_model # Delete the full model to free up memory

            # Freeze decoder parameters
            for param in self.decoder.parameters():
                param.requires_grad = False

            self.decoder.to(device)
            print(f"Loaded decoder from: {dec_checkpoint_path}")
        self.augment_pipeline.to(device)

        # Initial stem layer
        stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        )

        # 3D ResNet
        self.resnet = nn.Sequential(
            stem,
            self._make_layer(16, 32, num_blocks=2, stride=2),
            self._make_layer(32, 64, num_blocks=2, stride=2),
            self._make_layer(64, 128, num_blocks=2, stride=2),
            self._make_layer(128, 512, num_blocks=2, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Fully connected layer for embeddings
        self.fc = nn.Linear(512, embedding_size)

        # Projection head for SimCLR
        self.proj = nn.Sequential(
            nn.Linear(2048, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True)
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

    def forward(self, x):
        if self._with_decoder:
            with torch.no_grad():
                # # ===== 20-frames scheme =====
                # # x: (b, t, c, d, h, w)
                # b = x.size(0)
                # micro_bs = 2  # decode two batch elements at a time

                # decoded_chunks = []
                # for i in range(0, b, micro_bs):
                #     chunk = x[i:i + micro_bs]          # (micro_bs, t, c, d, h, w)
                #     out   = self.decoder(chunk)        # same shape, (micro_bs, t, c, d, h, w)
                #     decoded_chunks.append(out)

                # # Concatenate all decoded chunks back along the batch dimension
                # x = torch.cat(decoded_chunks, dim=0)   # (b, t, c, d, h, w)

                # # ===== 60-Frame Kinetics scheme =====
                # # x: (B, 60, C, D, H, W)
                # B, T, C, D, H, W = x.shape
                # assert T == 60, f"Expected 60 frames, got {T}"

                # micro_bs = 2
                # chunk_len = 20
                # n_chunks = T // chunk_len  # 3

                # decoded_movies = []

                # # loop over batch in micro-batches
                # for i in range(0, B, micro_bs):
                #     batch_chunk = x[i:i + micro_bs]  # (mb, 60, C, D, H, W)
                #     mb = batch_chunk.size(0)

                #     decoded_time_chunks = []

                #     # split each movie into 3x20 along time
                #     for t in range(n_chunks):
                #         t0 = t * chunk_len
                #         t1 = (t + 1) * chunk_len

                #         # (mb, 20, C, D, H, W)
                #         time_chunk = batch_chunk[:, t0:t1]

                #         # decode
                #         decoded = self.decoder(time_chunk)
                #         decoded_time_chunks.append(decoded)

                #     # reassemble time dimension
                #     # (mb, 60, C, D, H, W)
                #     decoded_movie = torch.cat(decoded_time_chunks, dim=1)
                #     decoded_movies.append(decoded_movie)

                # # reassemble batch
                # x = torch.cat(decoded_movies, dim=0)  # (B, 60, C, D, H, W)
                
                # ===== 20-frames 2024v3 scheme =====
                # x: (b, t, c, d, h, w)
                # print("Decoder")
                # print(f"Input shape to decoder: {x.size()}")
                # Fold time into batch for decoding
                b = x.size(0)
                t = x.size(1)
                # print(f"Reshaped input to decoder: {x.shape}")

                # Keep the last frame only (keep the time dimension)
                x = x[:, -1, ...]  # (b, 1, c, d, h, w)

                # fold time into batch for decoder
                # x = x.view(b * t, *x.shape[2:])  # (b*t, c, d, h, w)
                # print(f"Input shape to decoder after selecting last frame: {x.size()}")
                x = self.decoder(x)  # (b*t, c, d, h, w)
                # print(f"Output shape from decoder: {x.size()}")

                # Add time dimension back (with length 1 since we took only the last frame)
                x = x.view(b, 1, *x.shape[1:])  # (b, 1, c, d, h, w)

                # Reslice depth dimension from 64 back to original 60
                x = x[:, :, :, :60, ...]  # (b, t, c, 60, h, w)

                x = self.augment_pipeline(x) if self.apply_aug else (2 * x - 1)
        
        # Keep only the selected channels
        x = x[:, :, self._channels, ...] # (b, t, c, d, h, w)

        batch_size, time_steps, channels, depth, height, width = x.size()

        # Reshape for 3D convolution
        x = x.view(batch_size * time_steps, channels, depth, height, width)

        # Forward pass through 3D ResNet
        x = self.resnet(x)
        # x = x.view(batch_size, time_steps, -1)  # Reshape for LSTM

        # Forward pass through BiLSTM
        # x, _ = self.lstm(x)
        # x = x[:, -1, :] # Use the last timestep from LSTM output

        # Reshape to (batch_size, feature_size)
        x = x.view(batch_size, -1)

        # feature embedding
        x = self.fc(x)

        # projection head for SimCLR loss eval
        out = self.proj(x)

        return x, out
    

if __name__ == '__main__':
    # cfg = load_config("/u/earkfeld/MitoSpace4D/simclr/config.yaml")
    cfg = load_config("/u/earkfeld/MitoSpace4D/simclr/config_2024v3_3d.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model and print the output shape
    model = Lightweight3DResNet(embedding_size=2048, 
                                cfg=cfg,
                                decoder_checkpoint_path="/u/earkfeld/MitoSpace4D/mitospace_autoencoder/2024v3_autoencoder_sequence-norm.pt",
                                apply_aug=False,
                                device=device).to(device)
    
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    sample_input = torch.randn(2, 2, 2, 16, 64, 64).to(device) # B, T, C, D, H, W
    # sample_input = torch.randn(1, 20, 2, 30, 256, 256).to(device)  # Example input
    output = model(sample_input)
    print(f"embedding size: {output[0].shape}")  # Should be (batch_size, embedding_size)
    print(f"projection size: {output[1].shape}")  # Should be (batch_size, 512)
