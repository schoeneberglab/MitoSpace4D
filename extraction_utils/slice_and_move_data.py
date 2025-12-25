import os
import shutil
import numpy as np
from tqdm import tqdm
import argparse


def slice_and_move(src_dir, dest_dir, timestep_start, timestep_end, z_start, z_end, drugs=['nothing']):
    for drug in drugs:
        print(drug)
        os.makedirs(os.path.join(dest_dir, drug), exist_ok=True)
        pbar = tqdm(total=len(os.listdir(os.path.join(src_dir, drug))))

        for file in os.listdir(os.path.join(src_dir, drug)):
            if file.endswith('.npy'):
                data = np.load(os.path.join(src_dir, drug, file))
                sliced_data = data[timestep_start: timestep_end, z_start: z_end]
                np.save(os.path.join(dest_dir, drug, file), sliced_data)
            pbar.update(1)


if __name__ == '__main__':
    """ The file loads the complete 4D data, slices the z and time dimensions and move it to scratch space in SDSC"""

    parser = argparse.ArgumentParser(description="Process 4D data for different drugs")
    parser.add_argument('--drugs', type=str, nargs='+', required=True, help="List of drugs")
    args = parser.parse_args()

    src_dir = "/tscc/nfs/home/d5agarwal/projects/MitoSpace4D/data/2023_data/processed_data"
    dest_dir = "/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/2023_data/processed_data"
    os.makedirs(dest_dir, exist_ok=True)

    timestep_start = 0
    timestep_end = 20

    z_start = 20
    z_end = 80

    slice_and_move(src_dir, dest_dir, timestep_start, timestep_end, z_start, z_end, args.drugs)
