import torch
import torch.nn as nn
from torchsummary import summary

class BasicResNet3DDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor = 1):
        super(BasicResNet3DDecoderBlock, self).__init__()

        # Define the upsampling layer if out_channels > in_channels
        self.upsample = nn.Sequential()
        if upsample_factor != 1:
            self.upsample = nn.Upsample(scale_factor=(upsample_factor, upsample_factor, upsample_factor), mode='trilinear', align_corners=True)

        self.block1 = nn.Sequential(
            self.upsample,
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )        

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                self.upsample,
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            )
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        identity = self.shortcut(x)  # Adjust input to match output channels
        out = self.block1(x)        # Pass through first block
        out = self.block2(out)        # Pass through second block
        out += identity                # Add shortcut connection
        out = self.relu(out)          # Apply ReLU activation
        return out

class MitoSpace3DResNetDecoder(nn.Module):
    def __init__(self, output_dim=2):
        super(MitoSpace3DResNetDecoder, self).__init__()

        # Adjust the input/output channels according to the original architecture
        self.deconv1 = BasicResNet3DDecoderBlock(in_channels=4, out_channels=64, upsample_factor=1)
        self.deconv2 = BasicResNet3DDecoderBlock(in_channels=64, out_channels=64, upsample_factor=1)
        self.deconv3 = BasicResNet3DDecoderBlock(in_channels=64, out_channels=128, upsample_factor=1)
        self.deconv4 = BasicResNet3DDecoderBlock(in_channels=128, out_channels=64, upsample_factor=1)
        self.deconv5 = BasicResNet3DDecoderBlock(in_channels=64, out_channels=8, upsample_factor=2)
        self.deconv6 = BasicResNet3DDecoderBlock(in_channels=8, out_channels=8, upsample_factor=1)
        self.deconv7 = BasicResNet3DDecoderBlock(in_channels=8, out_channels=4, upsample_factor=1)
        self.deconv8 = BasicResNet3DDecoderBlock(in_channels=4, out_channels=output_dim, upsample_factor=2)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        batch_size, timesteps, channels, stacks, height, width = x.shape
        x = x.view(batch_size * timesteps, channels, stacks, height, width)
        
        x = self.deconv1(x)  # First decoding block
        x = self.deconv2(x)  # Second decoding block
        x = self.deconv3(x)  # Third decoding block
        x = self.deconv4(x)  # Fourth decoding block
        x = self.deconv5(x)  # Fifth decoding block
        x = self.deconv6(x)  # Sixth decoding block
        x = self.deconv7(x)  # Seventh decoding block
        x = self.deconv8(x)  # Eighth decoding block
        x = self.relu(x)

        _, _, new_stacks, new_height, new_width = x.shape
        x = x.view(batch_size, timesteps, -1, new_stacks, new_height, new_width)

        return x

def main():
    # Initialize the decoder
    output_dim = 2  # Set the desired output dimension
    decoder = MitoSpace3DResNetDecoder(output_dim=output_dim)
    
    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)

    # Print the model architecture using torchsummary
    # Assuming the input shape is (batch_size, in_channels, stacks, height, width)
    batch_size = 2
    in_channels = 4  # Adjust according to your specific input
    timesteps = 5
    stacks = 16
    height = 64
    width = 64
    summary(decoder, (timesteps, in_channels, stacks, height, width), batch_size=batch_size)

if __name__ == "__main__":
    main()


