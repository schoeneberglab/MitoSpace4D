# import torch
# import torch.nn as nn
# from torchsummary import summary

# # Define the BasicResNet3DEncoderBlock (if not already defined)
# class BasicResNet3DEncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicResNet3DEncoderBlock, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.LeakyReLU(negative_slope=0.2)
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#             )
#         self.relu = nn.LeakyReLU(negative_slope=0.2)

#     def forward(self, x):
#         identity = self.shortcut(x)
#         out = self.block1(x)
#         out = self.block2(out)
#         out += identity
#         out = self.relu(out)
#         return out

# # Updated MitoSpace3DEncoder class
# class MitoSpace3DResNetEncoder(nn.Module):
#     def __init__(self, input_dim=2):
#         super(MitoSpace3DResNetEncoder, self).__init__()

#         # Using BasicResNet3DEncoderBlock instead of Conv3D
#         self.conv1 = BasicResNet3DEncoderBlock(input_dim, 4, stride=2)
#         self.conv2 = BasicResNet3DEncoderBlock(4, 8, stride=1)
#         self.conv3 = BasicResNet3DEncoderBlock(8, 8, stride=1)
#         self.conv4 = BasicResNet3DEncoderBlock(8, 64, stride=2)
#         self.conv5 = BasicResNet3DEncoderBlock(64, 128, stride=1)
#         self.conv6 = BasicResNet3DEncoderBlock(128, 64, stride=1)
#         self.conv7 = BasicResNet3DEncoderBlock(64, 64, stride=1)
#         self.conv8 = nn.Sequential(
#             nn.Conv3d(64, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
#         )

#     def forward(self, x):
#         assert len(x.shape) == 6
#         batch_size, timesteps, channels, stacks, height, width = x.shape
#         x = x.view(batch_size * timesteps, channels, stacks, height, width)

#         noisy_x = torch.nn.functional.interpolate(
#             x,
#             scale_factor=(0.25, 0.25, 0.25),
#             mode='nearest',
#             align_corners=None
#         )

#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)

#         _, _, new_stacks, new_height, new_width = x.shape
#         noisy_x = noisy_x.view(batch_size, timesteps, channels, new_stacks, new_height, new_width)
#         x = x.view(batch_size, timesteps, channels, new_stacks, new_height, new_width)

#         return torch.cat((x, noisy_x), dim=2)

# def main():
#     # Initialize the encoder
#     encoder = MitoSpace3DResNetEncoder(input_dim=2)
    
#     # Move the model to the appropriate device (CPU or GPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     encoder.to(device)

#     # Create a random input tensor with shape (batch_size, timesteps, channels, stacks, height, width)
#     # Example dimensions: batch_size = 2, timesteps = 5, channels = 2, stacks = 16, height = 64, width = 64
#     batch_size = 2
#     timesteps = 5
#     channels = 2
#     stacks = 60
#     height = 256
#     width = 256

#     x = torch.randn(batch_size, timesteps, channels, stacks, height, width).to(device)

#     # Pass the input tensor through the encoder to get the output
#     output = encoder(x)

#     # Print the shapes of the input and output tensors
#     print("Input shape:", x.shape)
#     print("Output shape:", output.shape)

#     # Print a summary of the model
#     summary(encoder, (timesteps, channels, stacks, height, width), batch_size=batch_size)

# if __name__ == "__main__":
#     main()
