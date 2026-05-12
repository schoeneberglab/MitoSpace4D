import glob
import os
import pickle as pkl

import numpy as np
from skimage.filters import threshold_otsu
from tqdm import tqdm

np.random.seed(1123)


def degrade_llsm_to_confocal(image_3d):
    """Transforms LLSM to low-quality Cell Painting style."""

    img_morph = image_3d[1]
    img_tmrm = image_3d[0]

    thresh = threshold_otsu(img_morph)
    mask = img_morph > thresh
    img_tmrm_masked = img_tmrm * mask

    # Axial blur: average N planes to simulate thick confocal slice
    # mid_z = image_3d.shape[0] // 2
    z_index = np.random.rand()
    mid_z = 20 + int((40 - 20) * z_index)
    img = np.mean(
        img_morph[mid_z - 5 : mid_z + 5, :, :], axis=0
    )  # use entire z-stack due to thick optical sectioning in cell painting

    # get the same slice from the masked tmrm channel
    img_tmrm_masked_slice = img_tmrm_masked[mid_z - 5 : mid_z + 5 :, :, :]
    mean_tmrm = np.mean(img_tmrm_masked_slice)

    # Get the mean tmrm image for the same slice (for later use in evaluation)
    img_tmrm_mean = np.mean(img_tmrm_masked_slice, axis=0)

    # Reduce contrast
    img /= 3
    bg_level = 0.01
    img += bg_level

    # Add background read noise
    img += np.random.normal(0, 0.0015, img.shape)

    int_low = 0.0
    int_high = float(np.percentile(img, 99.99))
    if int_high <= int_low:  # guard against flat patches
        int_high = int_low + 1.0
    img = np.clip(img, int_low, int_high)
    image_norm = (img - int_low) / (int_high - int_low)
    image_norm = np.clip(image_norm, 0.0, 1.0)

    return image_norm, img_tmrm_mean, mean_tmrm


if __name__ == "__main__":
    root_path = (
        "/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v3_data/processed_data/"
    )
    # root_path = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/liver_drugs/"
    all_sample_path = sorted(glob.glob(root_path + "2024*"))
    frame_index = -1

    save_root_path = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v3_llsmtoconfocal_data/processed_data_seed1123/"
    # save_root_path = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/liver_drugs_llsmtoconfocal_data/processed_data_seed1123/"

    os.makedirs(save_root_path, exist_ok=True)

    tmrm_intensity_dict = {}

    for sample_path in all_sample_path:
        all_files = sorted(glob.glob(sample_path + "/*-0-1.npy"))

        print("Processing", sample_path)
        for cell_id, morph_file in tqdm(enumerate(all_files)):
            # zstack = np.load(file)[1, 0]
            zstack = np.load(morph_file)[frame_index]  # ch, frame 0
            tmrm_zstack = np.load(morph_file.replace("-0-1.npy", "-0-0.npy"))[
                frame_index
            ]  # ch, frame 0

            zstack = np.stack(
                [tmrm_zstack, zstack], axis=0
            )  # ch, z, x, y; tmrm is now channel 0, morph is channel 1
            img_morph, img_tmrm, mean_tmrm = degrade_llsm_to_confocal(zstack)

            save_path = save_root_path + sample_path.split("/")[-1] + "/"
            os.makedirs(save_path, exist_ok=True)

            filename = morph_file.split("/")[-1]

            outfile = save_path + filename
            tmrm_intensity_dict[outfile] = img_morph
            img_out = np.stack([img_tmrm, img_morph], axis=0)
            np.save(outfile, img_out)

    # pickle the tmrm intensity dict
    with open(f"{save_root_path}tmrm_intensity_dict.pkl", "wb") as f:
        pkl.dump(tmrm_intensity_dict, f)

    print("Processing complete. Saved degraded images and TMRM intensity dictionary.")
