import os
import numpy as np

if __name__ == '__main__':
    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data/20240729'
    for f in os.listdir(data_dir):
        if f.endswith('.npy'):
            data = np.load(os.path.join(data_dir, f))

            # pixel intensity graph
            total_pixels = data.size
            # find number of pixels in the range 0-255
            pixels_0_255 = np.sum((data >= 0) & (data <= 255)) / total_pixels * 100
            # find number of pixels in the range 256-511
            pixels_256_511 = np.sum((data >= 256) & (data <= 511)) / total_pixels * 100
            # find number of pixels in the range 512-767
            pixels_512_767 = np.sum((data >= 512) & (data <= 767)) / total_pixels * 100
            # find number of pixels in the range 768-1023
            pixels_768_1023 = np.sum((data >= 768) & (data <= 1023)) / total_pixels * 100
            # find number of pixels in the range 1024-all
            pixels_1024_all = np.sum(data > 1023) / total_pixels * 100

            print(pixels_0_255, pixels_256_511, pixels_512_767, pixels_768_1023, pixels_1024_all)
