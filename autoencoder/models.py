import torch
import torch.nn as nn

class MitoSpace3DDecoder(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=[1, 4, 16, 4, 1]):
        super(MitoSpace3DDecoder, self).__init__()

        # Define the entire decoder as a single sequential block
        self.decoder = nn.Sequential(
            # Layer 1: Transpose Conv, BatchNorm, ReLU (with bias)
            nn.ConvTranspose3d(in_channels=hidden_dim[4], out_channels=hidden_dim[3], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(hidden_dim[3]),
            nn.ReLU(inplace=True),

            # Layer 2: Transpose Conv, BatchNorm, ReLU (upsampling, with bias)
            nn.ConvTranspose3d(in_channels=hidden_dim[3], out_channels=hidden_dim[2], kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(hidden_dim[2]),
            nn.ReLU(inplace=True),

            # Layer 3: Transpose Conv, BatchNorm, ReLU (upsampling, with bias)
            nn.ConvTranspose3d(in_channels=hidden_dim[2], out_channels=hidden_dim[1], kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(hidden_dim[1]),
            nn.ReLU(inplace=True),

            # Layer 4: Transpose Conv, BatchNorm, ReLU (with bias)
            nn.ConvTranspose3d(in_channels=hidden_dim[1], out_channels=hidden_dim[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(hidden_dim[0]),
            nn.ReLU(inplace=True),

            # Final output layer (with bias)
            nn.ConvTranspose3d(in_channels=hidden_dim[0], out_channels=output_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        )

    def forward(self, x):
        # Apply the decoder to upsample
        batch_size, timesteps, channels, depth, height, width = x.shape
        x = x.view(batch_size * timesteps * channels, depth, height, width).unsqueeze(1)  # Merge batch and channel dimensions

        # Forward pass through the decoder
        x = self.decoder(x)

        # Step 3: Reshape back to [batch_size, timesteps, channels, depth, height, width]
        _, _, new_stacks, new_height, new_width = x.shape
        x = x.view(batch_size, timesteps, channels, new_stacks, new_height, new_width)  # Split batch and channel dimensions
        print(f"Final output shape: {x.shape}")

        return x

import torch
import torch.nn as nn

class MitoSpace3DEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=[1, 2, 4, 8, 32, 256, 1024]):
        super(MitoSpace3DEncoder, self).__init__()
        layers = []
        
        # Upsampling part (1, 2, 4, 8, 32, 256, 1024)
        in_channels = input_dim
        for out_channels in hidden_dim:
            layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update for next layer

        # Downsampling part (1024 -> 256 -> 8)
        downsample_dims = [1024, 256, 8]
        for i in range(len(downsample_dims) - 1):
            layers.append(nn.Conv3d(in_channels=downsample_dims[i], out_channels=downsample_dims[i+1], 
                                    kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True))
            layers.append(nn.BatchNorm3d(downsample_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        assert(len(x.shape) == 6)
        # Step 1: Reshape from [20, 2, 60, 256, 256] to [40, 60, 256, 256]
        batch_size, timesteps, channels, stacks, height, width = x.shape
        x = x.view(batch_size * timesteps * channels, stacks, height, width).unsqueeze(1)  # Merge batch and channel dimensions

        # Step 2: Apply the encoder to downsample
        x = self.encoder(x)

        # Step 3: Reshape back to [20, 2, 15, 64, 64]
        _, _, new_stacks, new_height, new_width = x.shape
        x = x.view(batch_size, timesteps, channels, new_stacks, new_height, new_width)  # Split batch and channel dimensions

        return x

class MitoSpace3DAutoencoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=[1, 4, 16, 4, 1]):
        super(MitoSpace3DAutoencoder, self).__init__()
        self.encoder = MitoSpace3DEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = MitoSpace3DDecoder(output_dim=output_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        # Pass through the encoder
        x = self.encoder(x)
        # Pass through the decoder
        return self.decoder(x)

if __name__ == '__main__':
    # Example usage:
    input_data = torch.randn(3, 5, 2, 60, 256, 256)

    # Using default input_dim and hidden_dim
    autoencoder = MitoSpace3DEncoder()
    output = autoencoder(input_data)
    print('Output shape', output.shape)