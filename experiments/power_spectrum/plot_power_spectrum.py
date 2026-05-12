import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def check_striping(image_slice):
    """
    Computes and displays the log-magnitude power spectrum of an image.
    Args:
        image_slice: 2D numpy array (single frame from your 4D stack)
    """
    # 1. Compute 2D FFT
    f_transform = fftpack.fft2(image_slice)

    # 2. Shift zero frequency to center
    f_shift = fftpack.fftshift(f_transform)

    # 3. Compute Magnitude Spectrum (Log scale for visibility)
    # Adding 1 to avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # 4. Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='inferno')
    plt.title('FFT Magnitude Spectrum')
    plt.axis('off')

    plt.show()

# Example usage:
# check_striping(my_4d_array[0, 10, :, :]) # Time 0, Slice 10

if __name__ == '__main__':
