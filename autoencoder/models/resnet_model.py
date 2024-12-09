import torch
import torch.nn as nn
from torchsummary import summary
from autoencoder.models.resnet_encoder import MitoSpace3DResNetEncoder  # Adjust the import paths as needed
from autoencoder.models.resnet_decoder import MitoSpace3DResNetDecoder  # Adjust the import paths as needed

class MitoSpace3ResNetAutoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(MitoSpace3ResNetAutoEncoder, self).__init__()
        self.encoder = MitoSpace3DResNetEncoder(input_dim=input_dim)
        self.decoder = MitoSpace3DResNetDecoder(output_dim=output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def test_autoencoder():
    # Define the input dimensions
    batch_size = 1  # Set to 1 for summary purposes
    timesteps = 20
    channels = 2
    stacks = 16
    height = 64
    width = 64

    # Create a random input tensor with the shape (batch, timesteps, channels, stacks, height, width)
    input_tensor = torch.rand(batch_size, timesteps, channels, stacks, height, width)

    # Initialize the autoencoder
    autoencoder = MitoSpace3ResNetAutoEncoder(input_dim=channels, output_dim=channels).to("cuda")

    # Print model summary
    print("Autoencoder Summary:")
    summary(autoencoder, input_size=(timesteps, channels, stacks, height, width), batch_size=batch_size)

# Run the test function
if __name__ == "__main__":
    test_autoencoder()
