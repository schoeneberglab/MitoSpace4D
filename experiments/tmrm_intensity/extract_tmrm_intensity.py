# import os.path as osp
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from skimage.filters import threshold_otsu
# from autoencoder.ae_util import AEUtil
#
# def extract_tmrm_intensities(img_paths, outfile="tmrm_intensities.csv", encoded=False):
#     """Extracts TMRM intensities on a per-frame basis for regression task."""
#
#     # img_paths = np.loadtxt(img_path_file, dtype=str, delimiter=',')
#     img_paths = np.array([path.strip() for path in img_paths])
#     print(f"Extracting TMRM intensities for {len(img_paths)} images.")
#     intensity_data = []
#
#     if encoded:
#         ae = AEUtil()
#
#     for i, img_path in enumerate(tqdm(img_paths)):
#
#         if encoded:
#             # Decode and detach from device
#             img = ae.load(path=img_path) # (C, T, D, H, W)
#         else:
#             img = np.load(img_path)  # (C, T, D, H, W)
#
#         raw_mean_intensities = np.zeros(img.shape[1])
#         otsu_mean_intensities = np.zeros(img.shape[1])
#         for t in range(img.shape[1]):
#
#             # Otsu threshold the morphology channel get a mask
#             thr = threshold_otsu(img[1, t, ...])
#             mask = img[1, t, ...] > thr
#
#             # Raw mean intensity of the tmrm channel
#             raw_mean_intensities[t] = img[0, t, ...].mean()
#
#             # Masked mean intensity of the tmrm channel
#             img[0, t, ...] = img[0, t, ...] * mask
#             otsu_mean_intensities[t] = img[0, t, ...].mean()
#
#         intensity_data.append({
#             "image_path": img_path,
#             "raw_mean_intensities": raw_mean_intensities,
#             "otsu_mean_intensities": otsu_mean_intensities,
#         })
#
#         if i % 100 == 0:
#             intensity_df = pd.DataFrame(intensity_data)
#             intensity_df.to_csv(outfile, index=False)
#
#     intensity_df = pd.DataFrame(intensity_data)
#     intensity_df.to_csv(outfile, index=False)

import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_otsu
from autoencoder.ae_util import AEUtil
from concurrent.futures import ThreadPoolExecutor  # ADDED
import pickle as pkl

def extract_tmrm_intensities(img_paths, outfile="tmrm_intensities.csv", encoded=False):
    """Extracts TMRM intensities on a per-frame basis for regression task."""

    # img_paths = np.loadtxt(img_path_file, dtype=str, delimiter=',')
    img_paths = np.array([path.strip() for path in img_paths])
    print(f"Extracting TMRM intensities for {len(img_paths)} images.")
    intensity_data = []

    if encoded:
        ae = AEUtil()

    for i, img_path in enumerate(tqdm(img_paths)):

        if encoded:
            # Decode and detach from device
            img = ae.load(path=img_path) # (C, T, D, H, W)
        else:
            img = np.load(img_path)  # (C, T, D, H, W)

        raw_mean_intensities = np.zeros(img.shape[1])
        otsu_mean_intensities = np.zeros(img.shape[1])

        def _process_frame(t):
            # Otsu threshold the morphology channel get a mask
            thr = threshold_otsu(img[1, t, ...])
            mask = img[1, t, ...] > thr

            # Raw mean intensity of the tmrm channel
            raw_mean = img[0, t, ...].mean()

            # Masked mean intensity of the tmrm channel
            masked = img[0, t, ...] * mask
            otsu_mean = masked.mean()

            return t, raw_mean, otsu_mean

        with ThreadPoolExecutor() as ex:  # ADDED
            for t, raw_mean, otsu_mean in ex.map(_process_frame, range(img.shape[1])):  # ADDED
                raw_mean_intensities[t] = raw_mean
                otsu_mean_intensities[t] = otsu_mean

        intensity_data.append({
            "image_path": img_path,
            "raw_mean_intensities": raw_mean_intensities,
            "otsu_mean_intensities": otsu_mean_intensities,
        })

        if i % 100 == 0:
            intensity_df = pd.DataFrame(intensity_data)

            # Write to pkl file
            with open(outfile, 'wb') as f:
                pkl.dump(intensity_data, f)

    intensity_df = pd.DataFrame(intensity_data)
    # intensity_df.to_csv(outfile, index=False)
    # Write to pkl file
    with open(outfile, 'wb') as f:
        pkl.dump(intensity_data, f)

if __name__ == "__main__":
    # Example usage
    # img_path = [
    #     # "/home/earkfeld/Projects/MitoSpace4D/data/2024v2_encoded_data/20240729-1/000000-0.npy"
    #     "/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v2_data/processed_data/20240729-1/000000-0.npy"
    # ]

    # infile = "/home/earkfeld/Projects/MitoSpace4D/runs/20260113_2024v2-embeddings_2024v2-model_all/image_paths.csv"
    # outfile = "/home/earkfeld/Projects/MitoSpace4D/runs/20260113_2024v2-embeddings_2024v2-model_all/tmrm_intensities.csv"

    infile = "/home/earkfeld/Projects/MitoSpace4D/runs/20260108_kinetics_morphology_resnet_embeddings_all/image_paths.csv"
    outfile = "/home/earkfeld/Projects/MitoSpace4D/runs/20260108_kinetics_morphology_resnet_embeddings_all/tmrm_intensities_kinetics.pkl"

    img_paths = np.loadtxt(infile, dtype=str, delimiter=',').tolist()
    print(f"Extracting TMRM intensities for {len(img_paths)} images.")
    print(f"infile: {infile}, \noutfile: {outfile}")

    extract_tmrm_intensities(img_paths, outfile=outfile, encoded=True)
