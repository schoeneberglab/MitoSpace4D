import os
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn.functional as F
import numpy as np

data_root = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/2024_data/processed_data'
compressed_data_root = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/2024_8bit_30z_data/processed_data'

def process_file(f, root):
    try:
        if not f.endswith('.npy'):
            return

        print(f'Processing {f} ...')
        fpath = os.path.join(root, f)
        data = np.load(fpath).astype(np.float32)
        data = torch.tensor(data)

        # Apply trilinear interpolation to reduce the third dimension
        resized_data = F.interpolate(data, size=(30, 256, 256), mode='trilinear', align_corners=False)
        data = resized_data.numpy().astype(np.uint16)

        # Clip and scale to 8-bit range
        max_value = 5000
        data = np.clip(data, 0, max_value)
        data = (data / max_value * 255).astype(np.uint8)

        # Prepare output path
        output_path = fpath.replace(data_root, compressed_data_root)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, data)

    except Exception as e:
        print(f'Error processing file {f}: {e}')

def process_drug_folder(drug):
    print(f'Processing {drug} ...')
    root = os.path.join(data_root, drug)
    files = sorted(os.listdir(root))
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, files, [root] * len(files))

if __name__ == '__main__':
    folders = sorted(['20240802', '20240904', '20240912'])

    # Process each folder in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(process_drug_folder, folders)
