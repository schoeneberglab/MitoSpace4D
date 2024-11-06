import tifffile
import os
import numpy as np

if __name__ == '__main__':
    fpath = "/home/dhruvagarwal/projects/MitoSpace4D/dumps/Control_488nm_stack0000_0000000msec_processed.tif"
    img = tifffile.imread(fpath)

    mip_img = np.max(img, axis=0)

    print()