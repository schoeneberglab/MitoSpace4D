import os
import os.path as osp
import numpy as np
import pandas as pd
import pyvista as pv
import imageio
import atexit
from PIL import Image
from PIL.ImageOps import grayscale
from matplotlib.colors import LinearSegmentedColormap
import json
import glob
from tqdm import tqdm
import multiprocessing as mp

# Number of parallel rendering processes. Each worker holds a PyVista off-screen
# GL context, so don't set this higher than ~ (CPU cores) / 2 in practice.
NUM_PROCESSES = 16

# Max samples per drug/condition to render. Set to None to render every sample
# in the filtered parquet (after the exclude-paths filter, still excluding
# metformin / label 26).
# SAMPLES_PER_CONDITION = 500
SAMPLES_PER_CONDITION = None

# Seed for reproducibility
np.random.seed(42)

# Worker-local state populated by _init_worker (avoids pickling large arrays
# per task). Indexed via process_single(i).
_WORKER_STATE = {}


def _init_worker(embeddings, labels, label_names_per_row, image_paths,
                 colors, video_save_dir, points_dir):
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    pv.OFF_SCREEN = True

    pv.set_plot_theme("document")
    pv.global_theme.multi_samples = 4  # disable MSAA so frames compress like the old runs
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    plotter.image_scale = 1
    plotter.set_background("white")
    plotter.hide_axes()
    atexit.register(plotter.close)

    _WORKER_STATE.update(
        embeddings=embeddings,
        labels=labels,
        label_names_per_row=label_names_per_row,
        image_paths=image_paths,
        colors=colors,
        video_save_dir=video_save_dir,
        points_dir=points_dir,
        plotter=plotter,
    )


def _point_sidecar_path(points_dir, i):
    return osp.join(points_dir, f"p{i:06d}.json")


def _write_point_sidecar(points_dir, i, point):
    """Atomically persist a single point's metadata as a JSON sidecar."""
    target = _point_sidecar_path(points_dir, i)
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(point, f, indent=2)
    os.replace(tmp, target)

# Function to select uniform indices
def uniform_choose_indices(labels, num):
    """Pick row indices, capped at `num` per label (None = take all).

    Excludes label 26 (metformin) regardless of `num`.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 26]  # Removing metformin
    chosen_idxs = []
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        if num is not None and len(idxs) > num:
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
def save_3d_surface_video(volume, cmap, out_path, plotter, fps=10):
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    movie = volume

    writer = imageio.get_writer(out_path, format='FFMPEG', mode='I',
                                fps=fps, codec='libx264', quality=8)

    num_frames = movie.shape[0]

    try:
        # Rotate 360 degrees for the first frame (to show the 3D volume from all angles)
        for t in range(num_frames):
            vol = movie[t].astype(np.float32)
            vol_range = np.ptp(vol)
            vol = (vol - vol.min()) / (vol_range if vol_range > 0 else 1.0)  # Normalize the volume

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
    finally:
        writer.close()


def videos_exist(video_save_dir, idx):
    return osp.exists(osp.join(video_save_dir, f"mtg_{str(idx).zfill(6)}.mp4"))


# Function to process a single image for metadata generation
def process_single(i):
    try:
        embeddings = _WORKER_STATE['embeddings']
        labels = _WORKER_STATE['labels']
        label_names_per_row = _WORKER_STATE['label_names_per_row']
        image_paths = _WORKER_STATE['image_paths']
        colors = _WORKER_STATE['colors']
        video_save_dir = _WORKER_STATE['video_save_dir']
        points_dir = _WORKER_STATE['points_dir']

        sidecar_path = _point_sidecar_path(points_dir, i)
        vid_name = f"{str(i).zfill(6)}.mp4"
        out_path = osp.join(video_save_dir, f"mtg_{vid_name}")

        # Fast-path: video + sidecar both already on disk → nothing to do.
        if osp.exists(sidecar_path) and osp.exists(out_path):
            return i

        if not osp.exists(out_path):
            img = np.load(image_paths[i])
            save_3d_surface_video(img, cmap="green", out_path=out_path,
                                  plotter=_WORKER_STATE['plotter'])

        img_url_mito = f"https://mitospace4d.s3.us-east-2.amazonaws.com/v3/mtg_{vid_name}"

        lbl = labels[i]
        label_name = label_names_per_row[i]
        r, g, b = colors[lbl]

        point = {
            "id": f"p{i}",
            "x": float(embeddings[i][0]),
            "y": float(embeddings[i][1]),
            "z": float(embeddings[i][2]),
            "phenotype": label_name,
            "color": {"r": r, "g": g, "b": b},
            "treatment": {
                "drug": label_name,
                "dose": "10 nM",
                "time": "1h"
            },
            "images": [img_url_mito],
            "metadata": {
                "cellLine": "Cal27",
                "experimentDate": "2025-03-15",
                "sampleId": f"MS{i}",
                "quality": 100
            }
        }
        _write_point_sidecar(points_dir, i, point)
        return i
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        return None


# Main function for orchestrating the processing
def main():
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

    proj_dir = "/home/earkfeld/Projects/MitoSpace4D/"
    embeddings_root = osp.join(proj_dir, "manuscript_v2/data/")
    embeddings_dir = osp.join(embeddings_root, "ms4d_2024v3_252eps")
    datafile = "embeddings+metadata_vis_joined.parquet"
    filter_infile = osp.join(embeddings_root, "2024v3_exclude_paths.parquet")

    save_dir = osp.join(embeddings_dir, "website_data")
    video_save_dir = osp.join(save_dir, "videos")
    points_dir = osp.join(save_dir, "points")
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(points_dir, exist_ok=True)

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

    print(f"Loading data from {osp.join(embeddings_dir, datafile)}")
    df = pd.read_parquet(osp.join(embeddings_dir, datafile)).reset_index(drop=True)

    df_filter = pd.read_parquet(filter_infile)
    n_init = len(df)
    df = df[~df['image_paths'].isin(df_filter['image_paths'])].reset_index(drop=True)
    print(f"Filtered out {n_init - len(df)} samples based on {osp.basename(filter_infile)}.")

    # Persist the filtered parquet so video filenames / sidecars can be joined
    # back to it externally. Row index in this file == the integer used in
    # mtg_NNNNNN.mp4 / pNNNNNN.json filenames.
    filtered_parquet_path = osp.join(save_dir, "embeddings+metadata_vis_joined_filtered.parquet")
    df.to_parquet(filtered_parquet_path, index=False)
    print(f"Saved filtered parquet ({len(df)} rows) to {filtered_parquet_path}")

    embeddings = np.stack(df['embeddings_umap'].values)
    labels = df['labels'].to_numpy()
    label_names_per_row = df['label_names'].to_numpy()
    image_paths = df['image_paths'].tolist()

    umap_outpath = osp.join(save_dir, "embeddings_umap.npy")
    np.save(umap_outpath, embeddings)
    print(f"Saved filtered UMAP embeddings ({embeddings.shape}) to {umap_outpath}")

    # Keep the full filtered arrays — process_single is called with indices that
    # refer to rows in the filtered parquet, so video/sidecar filenames line up
    # with that file's row order.
    chosen_idxs = [int(x) for x in uniform_choose_indices(labels, SAMPLES_PER_CONDITION)]

    incomplete_idxs = [i for i in chosen_idxs if not videos_exist(video_save_dir, i)]
    print(f"Selected {len(chosen_idxs)} cells; {len(incomplete_idxs)} still need rendering.")

    init_args = (embeddings, labels, label_names_per_row, image_paths,
                 colors, video_save_dir, points_dir)

    print(f"Launching {NUM_PROCESSES} worker processes...")
    ctx = mp.get_context('spawn')  # spawn avoids fork issues with Qt/OpenGL state
    with ctx.Pool(processes=NUM_PROCESSES,
                  initializer=_init_worker,
                  initargs=init_args) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single, chosen_idxs, chunksize=1),
            total=len(chosen_idxs),
        ):
            pass

    # Aggregate per-sample sidecars (written incrementally by workers) into the
    # final metadata.json. This is the source of truth — it survives crashes.
    sidecar_paths = sorted(glob.glob(osp.join(points_dir, "p*.json")))
    points = []
    for fp in sidecar_paths:
        with open(fp) as f:
            points.append(json.load(f))

    metadata = {"points": points}
    with open(osp.join(save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Aggregated {len(points)} point sidecars into "
          f"{osp.join(save_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
