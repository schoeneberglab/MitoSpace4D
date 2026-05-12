import os
import os.path as osp
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter  # Added for better off-screen rendering
import imageio
from PIL import Image
from PIL.ImageOps import grayscale
from matplotlib.colors import LinearSegmentedColormap
import json
import argparse
import torch
from utils.utils import load_config, get_fpaths
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Function to select uniform indices
def uniform_choose_indices(labels, num):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 26]  # Removing metformin
    chosen_idxs = []
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        if len(idxs) > num:
            np.random.seed(0)  # Set seed for reproducibility
            chosen_idxs.extend(np.random.choice(idxs, num, replace=False))
        else:
            chosen_idxs.extend(idxs)
    return np.array(chosen_idxs)


# Function to apply colormap
def get_rgb(image, cmap):
    img = 255 - image  # Invert the image
    rgb_vol = np.zeros_like(image, dtype=np.uint8)
    img_gray = grayscale(Image.fromarray(img))
    img_gray = np.array(img_gray)

    if cmap == 'green':
        rgb_vol[..., 1] = img_gray
    elif cmap == 'magenta':
        rgb_vol[..., 0] = img_gray
        rgb_vol[..., 2] = img_gray

    return rgb_vol


# Function to save 3D surface video
def save_3d_surface_video(volume, channel, cmap, out_path, fps=10):
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    movie = volume[:, channel]

    pv.set_plot_theme("document")
    plotter = BackgroundPlotter(off_screen=True,
                                window_size=(512, 512))  # Using BackgroundPlotter for off-screen rendering
    plotter.set_background("white")
    plotter.hide_axes()

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)

    num_frames = movie.shape[0]

    # Rotate 360 degrees for the first frame (to show the 3D volume from all angles)
    for t in range(num_frames):
        vol = movie[t]
        vol = (vol - vol.min()) / (vol.ptp())  # Normalize the volume

        grid = pv.wrap(vol)

        plotter.clear()
        plotter.add_volume(
            grid,
            cmap='gray',
            opacity="linear",
            shade=True,
        )
        plotter.remove_scalar_bar()

        if t == 0:
            for angle in np.linspace(0, 360, 30):  # Rotate the volume 360 degrees over 60 steps
                plotter.view_vector([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
                img = plotter.screenshot(transparent_background=False, return_img=True)
                img = get_rgb(img, cmap=cmap)  # Apply your colormap
                writer.append_data(img)

        else:
            img = plotter.screenshot(transparent_background=False, return_img=True)
            img = get_rgb(img, cmap=cmap)  # Apply your colormap
            writer.append_data(img)

    writer.close()


def videos_exist(video_save_dir, idx):
    return (
        osp.exists(osp.join(video_save_dir, f"mtg_{str(idx).zfill(6)}.mp4")) and
        osp.exists(osp.join(video_save_dir, f"tmrm_{str(idx).zfill(6)}.mp4"))
    )

# Function to process a single image for metadata generation
def process_single(args):
    i, embeddings, labels, label_names, image_paths, colors, colors_phenotypic, video_save_dir = args

    img = np.load(image_paths[i])

    vid_name = f"{str(i).zfill(6)}.mp4"

    save_3d_surface_video(img, channel=1, cmap="green", out_path=osp.join(video_save_dir, f"mtg_{vid_name}"))
    save_3d_surface_video(img, channel=0, cmap="magenta", out_path=osp.join(video_save_dir, f"tmrm_{vid_name}"))

    img_url_mito = f"https://mitospace4d.s3.us-east-2.amazonaws.com/mtg_{vid_name}"
    img_url_tmrm = f"https://mitospace4d.s3.us-east-2.amazonaws.com/tmrm_{vid_name}"

    lbl = labels[i]
    label_name = label_names[lbl]
    r, g, b = colors[lbl]
    r_p, g_p, b_p = colors_phenotypic[lbl]

    point = {
        "id": f"p{i}",
        "x": float(embeddings[i][0]),
        "y": float(embeddings[i][1]),
        "z": float(embeddings[i][2]),
        "phenotype": label_name,
        "color": {"r": r, "g": g, "b": b},
        "color_phenotypic": {"r": r_p, "g": g_p, "b": b_p},
        "treatment": {
            "drug": label_name,
            "dose": "10 nM",
            "time": "1h"
        },
        "images": [img_url_mito, img_url_tmrm],
        "metadata": {
            "cellLine": "Cal27",
            "experimentDate": "2025-03-15",
            "sampleId": f"MS{i}",
            "quality": 100
        }
    }

    return point


# Main function for orchestrating the processing
def main():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('--checkpoint_path', help='Checkpoint path')
    parser.add_argument('--config', default='/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml',
                        type=str, help='Config path.')
    parser.add_argument('--data_path', help='Data to predict')
    parser.add_argument('--load_epoch', help='Load weights from this epoch')
    args = parser.parse_args()

    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    torch.multiprocessing.set_sharing_strategy('file_system')

    cfg = load_config(args.config)
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"

    save_dir = osp.join(proj_dir, "runs", "lightning_logs", cfg['experiment_name'])
    os.makedirs(save_dir, exist_ok=True)

    print("Experiment name:", cfg['experiment_name'])

    image_paths = get_fpaths("/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data")

    # Function to load colors
    def load_colors(filepath):
        color_dict = {}
        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 6:
                    _, _, index, r, g, b = parts
                    color = [float(v) / 255 if float(v) > 1.0 else float(v) for v in (r, g, b)]
                    color_dict[int(index)] = color
        return color_dict

    colors = load_colors(osp.join(proj_dir, "extraction_utils/colors.txt"))
    colors_phenotypic = load_colors(osp.join(proj_dir, "extraction_utils/colors_phenotypic.txt"))

    embedding_dir = osp.join(save_dir, "embeddings")
    if not osp.exists(embedding_dir):
        raise RuntimeError("Embeddings are not saved. Please run the script again with --save_embeddings flag.")

    embeddings = np.load(osp.join(embedding_dir, "embeddings_umap.npy"))
    labels = np.load(osp.join(embedding_dir, "labels.npy"))
    label_names = np.load(osp.join(embedding_dir, "label_names.npy"))

    chosen_idxs = uniform_choose_indices(labels, 500)
    embeddings = embeddings[chosen_idxs]
    labels = labels[chosen_idxs]
    image_paths = [image_paths[i] for i in chosen_idxs]

    video_save_dir = osp.join(save_dir, "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    incomplete_idxs = [i for i in range(len(embeddings)) if not videos_exist(video_save_dir, i)]
    print(f"Found {len(incomplete_idxs)} incomplete entries. Will resume processing.")

    # Pack arguments for multiprocessing
    all_args = [(i, embeddings, labels, label_names, image_paths, colors, colors_phenotypic, video_save_dir) for i in
                range(len(embeddings))]

    # If any are missing, process just those
    # num_processes = min(cpu_count(), 16)  # Use up to 16 processes or the total number of cores
    # if incomplete_idxs:
    #     incomplete_args = [all_args[i] for i in incomplete_idxs]
    #     with Pool(processes=num_processes) as pool:
    #         for _ in tqdm(pool.imap(process_single, incomplete_args), total=len(incomplete_args)):
    #             pass
    #
    # else:
    #     print(f"Launching {num_processes} processes...")
    #     # Multiprocessing with tqdm for progress bar
    #     with Pool(processes=num_processes) as pool:
    #         points = list(tqdm(pool.imap(process_single, all_args), total=len(all_args)))

    points = []
    for i in range(len(embeddings)):
        if videos_exist(video_save_dir, i):
            args = all_args[i]
            try:
                point = process_single(args)  # This only builds the dict, doesn't re-run video saving
                points.append(point)
            except Exception as e:
                print(f"Error processing index {i}: {e}")

    # Save metadata to file
    metadata = {"points": points}
    with open(osp.join(save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {osp.join(save_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
