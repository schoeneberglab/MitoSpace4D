import torch
import torch.nn as nn
from simclr.augmentations import DataAugmentation
from utils.utils import load_config
import torch.nn.functional as F
from autoencoder.autoencoder_runner import AutoEncoderRunner
from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
import einops

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
                #  cfg_aug=None, 
                 apply_aug=False, 
                 decoder_checkpoint_path="/u/earkfeld/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt",
                 aggregation_method='attn'
                 ) -> None:

        super(Lightweight3DResNet, self).__init__()

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

        # # generate a tensor to test the channel indexing
        # test_tensor = torch.randn(1, 10, 2, 30, 64, 64) # (b, t, c, d, h, w)
        # print(f"Test tensor shape before indexing: {test_tensor.shape}")
        # indexed_tensor = test_tensor[:, :, self._channels, ...]
        # print(f"Test tensor shape after indexing: {indexed_tensor.shape}")

        # dec_checkpoint_path = "/u/earkfeld/MitoSpace4D/autoencoder/lightning_logs/final_training_sdsc_16_nodes_low_lr_low_gamma/lightning_logs/version_3178623/checkpoints/epoch=8-step=6462.ckpt"
        # dec_checkpoint_path = "/u/earkfeld/MitoSpace4D/checkpoints/MitospaceAutoencoder_Summer2024.ckpt"
        # dec_checkpoint_path = "/u/earkfeld/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"
        
        if self._with_decoder:
            dec_checkpoint_path = decoder_checkpoint_path
            ae_model = MitoSpace3DAutoencoder()
            ae_model = AutoEncoderRunner.load_from_checkpoint(dec_checkpoint_path, model=ae_model)
        
            self.decoder = ae_model.model.decoder
            self.decoder.eval()

            del ae_model # Delete the model to free up memory

            # Freeze decoder parameters
            for param in self.decoder.parameters():
                param.requires_grad = False

            self.decoder.to('cuda')
            print(f"Loaded decoder from: {dec_checkpoint_path}")
        self.augment_pipeline.to('cuda')

        # # Initial layer: modify for 2-channel input
        # self.stem = nn.Sequential(
        #     nn.Conv3d(in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        # )

        # # Define 3D ResNet layers with reduced channels and depth
        # self.layer1 = self._make_layer(16, 32, num_blocks=2, stride=2)
        # self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        # self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        # self.layer4 = self._make_layer(128, 512, num_blocks=2, stride=2)

        # # Adaptive average pooling to reduce spatial and depth dimensions
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Initial layer: modify for 2-channel input
        stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        )

        # Define 3D ResNet layers with reduced channels and depth
        self.resnet = nn.Sequential(
            stem,
            self._make_layer(16, 32, num_blocks=2, stride=2),
            self._make_layer(32, 64, num_blocks=2, stride=2),
            self._make_layer(64, 128, num_blocks=2, stride=2),
            self._make_layer(128, 512, num_blocks=2, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # BiLSTM for temporal encoding
        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)

        # Final fully connected layer for embedding
        self.fc = nn.Linear(1024 * 2, embedding_size)

        # Projection head for SimCLR
        self.proj = nn.Sequential(
            nn.Linear(2048, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True)
        )

        self.aggregator = None
        if aggregation_method == 'last':
            self.aggregator = lambda x: x[:, -1, :] # Get the last timestep
        elif aggregation_method == 'mean':
            self.aggregator = lambda x: x.mean(dim=1) # Mean over time
        elif aggregation_method == 'attn':
            self.aggregator = AttnPool1D(d=embedding_size, d_attn=256) # Attention pooling
        elif aggregation_method == 'None' or aggregation_method is None:
            self.aggregator = lambda x: x # No aggregation
        else:
            raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

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
                # x: (b, t, c, d, h, w)
                b = x.size(0)
                micro_bs = 2  # decode two batch elements at a time

                decoded_chunks = []
                for i in range(0, b, micro_bs):
                    chunk = x[i:i + micro_bs]          # (micro_bs, t, c, d, h, w)
                    out   = self.decoder(chunk)        # same shape, (micro_bs, t, c, d, h, w)
                    decoded_chunks.append(out)

                # Concatenate all decoded chunks back along the batch dimension
                x = torch.cat(decoded_chunks, dim=0)   # (b, t, c, d, h, w)

        # print(f"Input shape before augment/normalize: {x.size()}")
        # Augment or normalize
        x = self.augment_pipeline(x) if self.apply_aug else (2 * x - 1)
        # print(f"Input shape after augment/normalize: {x.size()}")
        # keep only the specified channels while retaining the channel dimension
        x = x[:, :, self._channels, ...] # (b, t, c, d, h, w)
        # print(f"Input shape after channel indexing: {x.size()}")

        batch_size, time_steps, channels, depth, height, width = x.size()

        # Reshape for 3D convolution
        x = x.view(batch_size * time_steps, channels, depth, height, width)

        # # Forward pass through 3D ResNet layers
        # x = self.stem(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # # Average pool and reshape
        # x = self.avgpool(x)

        # Forward pass through 3D ResNet layers
        x = self.resnet(x)
        
        x = x.view(batch_size, time_steps, -1)  # Reshape for LSTM

        # Forward pass through BiLSTM
        x, _ = self.lstm(x)

        # Aggregate embeddings
        x = self.aggregator(x) # (b, d) or (b, t, d) depending on aggregator

        #-- Aggregated embeddings (b, d)
        if x.dim() == 2:
            x = self.fc(x)
            out = self.proj(x)
            return x, out
        
        #-- Sequence-level embeddings (b, t, d)
        # Reshape for final fc and projection
        b, t, d = x.size()
        x = x.reshape(-1, d)
        
        # Final embedding & projection
        x = self.fc(x)
        out = self.proj(x)

        # Reshape back to (b, t, d)
        x = x.reshape(b, t, -1)
        out = out.reshape(b, t, -1)[:, -1]
        return x, out

class AttnPool1D(torch.nn.Module):
    def __init__(self, d, d_attn=128):
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Linear(d, d_attn, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(d_attn, 1, bias=False)
        )

    def forward(self, H, mask=None):
        # mask: (B, T) with 0 for pad, 1 for valid (optional)
        s = self.score(H).squeeze(-1)               # (B, T)
        if mask is not None: s = s.masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(s, dim=1)                 # (B, T)
        z = torch.einsum("bt, btd -> bd", w, H)     # (B, D)
        return z

if __name__ == '__main__':
    cfg = load_config("/u/earkfeld/MitoSpace4D/simclr/config.yaml")
    # Initialize model and print the output shape
    model = Lightweight3DResNet(embedding_size=2048, 
                                cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=True).cuda()
    
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    sample_input = torch.randn(1, 20, 2, 30, 256, 256).cuda()  # Example input
    output = model(sample_input)
    print(output.shape)  # Should be (batch_size, embedding_size)
