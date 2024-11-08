import torch
from torch import Tensor
from pytorch_msssim import SSIM, MS_SSIM
# Assuming the SSIM class from the provided code is available

def main():
    # Create two random 3D tensors (e.g., simulating two 3D images with shape (N, C, D, H, W))
    N, C, D, H, W = 1, 2, 60, 256, 256  # Batch size, channels, depth, height, width
    tensor1 = torch.rand((N, C, D, H, W))
    tensor2 = torch.rand((N, C, D, H, W))

    # Instantiate the MS_SSIM class for 3D images with 2 channels
    msssim_module = MS_SSIM(
        data_range=1.0,      # Assuming the random tensors are between 0 and 1
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=C,           # Number of channels is 2
        spatial_dims=3       # Set spatial dimensions to 3 for 3D images
    )

    # Calculate MSSSIM between the two 3D tensors
    msssim_value = msssim_module(tensor1, tensor2)

    # Print the MSSSIM value
    print(f"MSSSIM between the two random 3D tensors: {msssim_value.item()}")

if __name__ == "__main__":
    main()
