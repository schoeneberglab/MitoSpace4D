import os

import imageio
import napari
import numpy as np
import pandas as pd
import torch

from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder

DATA_DIR = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v3_data/"
PARQUET_PATH = (
    "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/"
    "ms4d_2024v3_supcon_210eps/embeddings+metadata_vis_joined.parquet"
)
AE_CKPT_PATH = (
    "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/checkpoints/2024v3_autoencoder.pt"
)
SAVE_DIR = "manuscript_v2/example_images/autoencoder_example_images_bright"
FRAME = 0

EXPECTED_RAW_SHAPE = (20, 60, 256, 256)
EXPECTED_LATENT_SHAPE = (20, 2, 16, 64, 64)
RAW_Z = EXPECTED_RAW_SHAPE[1]  # encoder pads (0, 64-60) at depth tail, so decoded[:60] is the real region


def get_bounding_box_lines(z, y, x):
    return [
        [[0, 0, 0], [z, 0, 0]], [[0, y, 0], [z, y, 0]], [[0, 0, x], [z, 0, x]], [[0, y, x], [z, y, x]],
        [[0, 0, 0], [0, y, 0]], [[z, 0, 0], [z, y, 0]], [[0, 0, x], [0, y, x]], [[z, 0, x], [z, y, x]],
        [[0, 0, 0], [0, 0, x]], [[z, 0, 0], [z, 0, x]], [[0, y, 0], [0, y, x]], [[z, y, 0], [z, y, x]],
    ]


def load_decoder(ckpt_path, device):
    """Load the 2024v3 autoencoder `.pt` and return its decoder, frozen and on `device`."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MitoSpace3DAutoencoder(input_dim=1, latent_dim=2, output_dim=1)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    decoder = model.decoder.eval().to(device)
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


def render_volume_png(
    viewer,
    volume_zyx,
    save_path,
    name="volume",
    contrast_limits=(0.0, 0.8),
    colormap="green",
    gamma=1.2,
    camera_angles=(-10, 20, 130),
    perspective=10,
    zoom=4,
    window_size=(2010, 1733),
    scale_factors=(1, -1, 1, 1),
    show_bounding_box=True,
):
    """Render a single (Z,Y,X) volume into `viewer` with the manuscript camera
    setup, save a screenshot to `save_path`, and clear the viewer's layers.

    Caller owns the viewer lifecycle (create + close). `contrast_limits` is
    per-cell because raw is min-max normalized to [0, 1] in `load_raw_frame`
    and decoded output is sigmoid-bounded to [0, 1] — so (0, 0.8) shows the
    same relative window of each cell's dynamic range."""
    z_max, y_max, x_max = volume_zyx.shape

    viewer.layers.clear()

    bounding_box = None
    if show_bounding_box:
        bounding_box = viewer.add_shapes(
            get_bounding_box_lines(z_max, y_max, x_max),
            shape_type="line",
            edge_width=0.8,
            edge_color="gray",
            blending="translucent",
            opacity=0.8,
            name="Bounding Box",
        )

    layer = viewer.add_image(
        volume_zyx,
        name=name,
        contrast_limits=list(contrast_limits),
        colormap=colormap,
        blending="additive",
        rendering="mip",
        gamma=gamma,
    )

    viewer.dims.ndisplay = 3
    viewer.window.resize(*window_size)

    spatial_scale = scale_factors[1:]
    layer.scale = spatial_scale
    if bounding_box is not None:
        bounding_box.scale = spatial_scale

    viewer.camera.angles = camera_angles
    viewer.camera.perspective = perspective
    viewer.camera.center = (
        (z_max / 2) * spatial_scale[0],
        (y_max / 2) * spatial_scale[1],
        (x_max / 2) * spatial_scale[2],
    )
    viewer.camera.zoom = zoom

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    imageio.imwrite(save_path, screenshot)
    viewer.layers.clear()
    return save_path


def encoded_path_for(raw_path):
    """processed_data/<plate>/<file>.npy -> encoded_data/<plate>/<file>.npy"""
    if "/processed_data/" not in raw_path:
        raise ValueError(f"raw path missing /processed_data/: {raw_path}")
    return raw_path.replace("/processed_data/", "/encoded_data/")


def load_raw_frame(raw_path, frame):
    """Load raw, apply the same per-cell min-max normalization the autoencoder
    was trained with (mitospace_autoencoder/encode_data.py:normalize_channel),
    and return frame `frame`."""
    raw = np.load(raw_path)
    assert raw.shape == EXPECTED_RAW_SHAPE, f"unexpected raw shape {raw.shape} at {raw_path}"
    raw = raw.astype(np.float32)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return raw[frame]  # (Z=60, Y, X)


def decode_frame(encoded_path, frame, decoder, device):
    latent = np.load(encoded_path)
    assert latent.shape == EXPECTED_LATENT_SHAPE, (
        f"unexpected latent shape {latent.shape} at {encoded_path}"
    )
    z = torch.from_numpy(latent).to(device).unsqueeze(0)  # (1, T, 2, 16, 64, 64)
    with torch.no_grad():
        y = decoder(z)  # (1, T, 1, 64, 256, 256)
    vol = y[0, frame, 0].cpu().numpy()  # (64, Y, X)
    return vol[:RAW_Z]  # (Z=60, Y, X) — drop the tail-padded slices


def output_paths(save_dir, label_name):
    safe_name = label_name.replace("/", "-").replace(" ", "_")
    return (
        os.path.join(save_dir, f"{safe_name}_original.png"),
        os.path.join(save_dir, f"{safe_name}_decoded.png"),
    )


def visualize_condition(label_name, raw_path, decoder, device, save_dir, frame, **render_kwargs):
    """Render raw + decoded for one condition. Creates a fresh napari.Viewer
    for the pair, clears it between renders, and closes it before returning —
    so each pair starts from a clean Qt/viewer state."""
    encoded = encoded_path_for(raw_path)
    raw_vol = load_raw_frame(raw_path, frame)
    decoded_vol = decode_frame(encoded, frame, decoder, device)

    out_orig, out_dec = output_paths(save_dir, label_name)

    viewer = napari.Viewer()
    try:
        render_volume_png(viewer, raw_vol, out_orig, name=f"{label_name}_original", **render_kwargs)
        print(f"Saved {out_orig}")
        render_volume_png(viewer, decoded_vol, out_dec, name=f"{label_name}_decoded", **render_kwargs)
        print(f"Saved {out_dec}")
    finally:
        viewer.close()
    return out_orig, out_dec


def representative_samples(parquet_path):
    """Pick one random sample per condition. Re-running yields a fresh draw."""
    df = pd.read_parquet(parquet_path)
    return (
        df[["label_names", "image_paths"]]
        .groupby("label_names", as_index=False)
        .sample(n=1)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_decoder(AE_CKPT_PATH, device)

    samples = representative_samples(PARQUET_PATH)
    print(f"Considering {len(samples)} conditions for {SAVE_DIR}")

    for _, row in samples.iterrows():
        label_name = row["label_names"]
        out_orig, out_dec = output_paths(SAVE_DIR, label_name)
        if os.path.exists(out_orig) and os.path.exists(out_dec):
            print(f"Skipping {label_name} (both PNGs already exist)")
            continue
        visualize_condition(
            label_name,
            row["image_paths"],
            decoder,
            device,
            SAVE_DIR,
            FRAME,
        )
