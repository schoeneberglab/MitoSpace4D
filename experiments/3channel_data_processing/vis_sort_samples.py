import numpy as np
import napari
import os

data_dir = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/andre_3color_cancer/20260129-0/000007"

files = sorted(os.listdir(data_dir))

data = []
for f in files:
    infile = os.path.join(data_dir, f)
    data.append(np.load(infile))

data = np.array(np.stack(data, axis=0))

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data)

napari.run()