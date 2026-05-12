import os
import numpy as np
import imageio
import napari
import torch

from simclr.augs_dtai import (
    RandomHorizontalFlip3D,
    RandomVerticalFlip3D,
    RandomDepthicalFlip3D,
    RandomRotation3D,
    RandomAffine3D,
    RandomExchangeFlip,
    RandomResizedCrop,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomErasing,
    RandomBrightness,
    RandomTimeFlip,
    RandomTimeMask,
)

save_dir = "manuscript_v2/example_images"
data_dir = "/mnt/aquila/others/MitoSpace4D/2024v2_data/processed_data/20240729-1"
# cell_ids = np.random.choice(1520, size=100, replace=False).tolist()
# cell_ids = [486]
cell_ids = [410]
window_id = 0
frames = [0]


def cell_path(data_dir, cell_id, window_id=0):
    return os.path.join(data_dir, f"{str(cell_id).zfill(6)}-{window_id}.npy")


def get_bounding_box_lines(z, y, x):
    return [
        [[0, 0, 0], [z, 0, 0]], [[0, y, 0], [z, y, 0]], [[0, 0, x], [z, 0, x]], [[0, y, x], [z, y, x]],
        [[0, 0, 0], [0, y, 0]], [[z, 0, 0], [z, y, 0]], [[0, 0, x], [0, y, x]], [[z, 0, x], [z, y, x]],
        [[0, 0, 0], [0, 0, x]], [[z, 0, 0], [z, 0, x]], [[0, y, 0], [0, y, x]], [[z, y, 0], [z, y, x]],
    ]


def build_augmentations():
    """Build the *active* augmentations from
    `simclr/config_2024v3_4d_dtai.yaml` (p > 0 there) with `p=1.0` so they
    always fire, and parameter ranges pinned to their maximum-strength
    endpoint so each visualization shows the worst-case distortion training
    would see.

    Each entry is `(name, module, kind)`:
    - `"3d"`: transform expects (B,C,Z,H,W) — used by `transforms_3d` after the
      `reshape(b, t*c, z, h, w)` at simclr/augs_dtai.py:368.
    - `"2d"`: per-frame transform expects (B,C,H,W) — used by `transforms_2d`
      after the `reshape(b, -1, h, w)` at simclr/augs_dtai.py:363, with t/c/z
      collapsed into the channel dim so a single sampled crop / erase / blur
      / noise instance is shared across all (t,z) slices.
    - `"4d"`: 6-D-input transform expects (B,T,C,Z,H,W) — used by
      `temporal_transform_*` and the new `brightness_3d` at
      simclr/augs_dtai.py:369-370.

    `RandomTimeFlip` is omitted because the config sets its `p=0.0`.
    """
    # RandomTimeMask draws clip length from an internal probability table
    # (simclr/augs_dtai.py:88–92). Pin to the most aggressive setting:
    # shortest clip = 11 of 20 frames kept. Use `n_views=1` so the module
    # returns a single masked view (no time-delay logic).
    time_mask = RandomTimeMask(p=1.0, n_views=1)
    time_mask.clip_len = torch.tensor([11.0])
    time_mask.clip_len_probs = torch.tensor([1.0])

    return [
        # 3D spatial (transforms_3d)
        ("RandomHorizontalFlip3D", RandomHorizontalFlip3D(p=1.0), "3d"),
        ("RandomVerticalFlip3D", RandomVerticalFlip3D(p=1.0), "3d"),
        ("RandomDepthicalFlip3D", RandomDepthicalFlip3D(p=1.0), "3d"),
        ("RandomRotation3D", RandomRotation3D(p=1.0, degrees=((35, 35), (35, 35), (35, 35))), "3d"),
        ("RandomAffine3D", RandomAffine3D(p=1.0, degrees=0, translate=(0.2, 0.2, 0.1)), "3d"),
        ("RandomExchangeFlip", RandomExchangeFlip(p=1.0), "3d"),
        # 2D shared-across-(t,z) (transforms_2d). Tighten ranges so the
        # strongest endpoint is always sampled.
        ("RandomResizedCrop", RandomResizedCrop(p=1.0, size=(256, 256), scale=(0.35, 0.35)), "2d"),
        ("RandomGaussianBlur", RandomGaussianBlur(p=1.0, kernel_size=(13, 13), sigma=(1.5, 1.5)), "2d"),
        ("RandomErasing", RandomErasing(p=1.0, scale=(0.165, 0.165), ratio=(0.3, 3.3)), "2d"),
        ("RandomGaussianNoise", RandomGaussianNoise(p=1.0, mean=0.0, std=0.05), "2d"),
        # Gain+Offset intensity (brightness_3d). Force max gain (1.2) and
        # max additive offset (+0.05). `per_channel=False` matches the config
        # and is the single-channel default.
        (
            "RandomBrightness",
            RandomBrightness(
                p=1.0,
                lower=1.2,
                upper=1.2,
                add_lower=0.05,
                add_upper=0.05,
                per_channel=False,
            ),
            "4d",
        ),
        # temporal (temporal_transform_1)
        ("RandomTimeMask", time_mask, "4d"),
    ]


def apply_aug(arr, aug_module, kind):
    """Apply an augmentation. Input/output dimensionality matches `kind`:
    - `"3d"`: arr is (Z,Y,X) → wraps to (1,1,Z,Y,X), returns (Z,Y,X). Matches
      the `reshape(b, t*c, z, h, w)` at simclr/augs_dtai.py:368 (with b=t=c=1
      here, so a single 3D transform is sampled per call).
    - `"2d"`: arr is (Z,Y,X) → wraps to (1,Z,Y,X), so Z is collapsed into the
      channel dim and a single sampled transform (one crop, one erase
      rectangle, etc.) is shared across all z-slices. Matches the
      `reshape(b, -1, h, w)` at simclr/augs_dtai.py:363.
    - `"4d"`: arr is (T,Z,Y,X) → wraps to (1,T,1,Z,Y,X), returns (T,Z,Y,X).
      Matches the (b,t,c,z,h,w) shape used by `temporal_transform_*` and the
      new `brightness_3d` (Gain+Offset) at simclr/augs_dtai.py:357-370.

    For modules that return a list of views (e.g. `RandomTimeMask`), the first
    view is returned.
    """
    x = torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32)
    if kind == "3d":
        x = x.unsqueeze(0).unsqueeze(0)
    elif kind == "2d":
        x = x.unsqueeze(0)
    elif kind == "4d":
        x = x.unsqueeze(0).unsqueeze(2)
    else:
        raise ValueError(f"unknown augmentation kind: {kind!r}")
    with torch.no_grad():
        y = aug_module(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    if kind == "3d":
        return y.squeeze(0).squeeze(0).cpu().numpy()
    if kind == "2d":
        return y.squeeze(0).cpu().numpy()
    return y.squeeze(0).squeeze(1).cpu().numpy()


def render_volume_png(
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
    """Open napari, render a single (Z,Y,X) volume with the manuscript camera
    setup, save a screenshot to `save_path`, and close the viewer."""
    z_max, y_max, x_max = volume_zyx.shape

    viewer = napari.Viewer()

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
    viewer.close()
    return save_path


def visualize_augmentations(
    img_path,
    save_dir,
    frames,
    augmentations=None,
    name=None,
    channel=1,
    **render_kwargs,
):
    """For each frame in `frames`, save the original 3D volume plus one PNG per
    augmentation in `augmentations` (a list of (name, module) pairs). Output
    files: `<save_dir>/<base>_<frame>_<original|aug>.png`."""
    os.makedirs(save_dir, exist_ok=True)

    if augmentations is None:
        augmentations = build_augmentations()

    movie = np.load(img_path)[channel]
    n_t = movie.shape[0]
    if name is None:
        name = os.path.splitext(os.path.basename(img_path))[0]

    saved = []
    for t in frames:
        if t < 0 or t >= n_t:
            raise ValueError(f"frame index {t} out of range [0, {n_t})")
        frame = movie[t]

        out = os.path.join(save_dir, "original.png")
        render_volume_png(frame, out, name=f"{name}_t{t}_original", **render_kwargs)
        saved.append(out)
        print(f"Saved {out}")

        for aug_name, aug_module, kind in augmentations:
            if kind == "4d":
                augmented = apply_aug(movie, aug_module, kind)[t]
            else:
                augmented = apply_aug(frame, aug_module, kind)
            out = os.path.join(save_dir, f"{aug_name}.png")
            render_volume_png(augmented, out, name=f"{name}_t{t}_{aug_name}", **render_kwargs)
            saved.append(out)
            print(f"Saved {out}")

    return saved


def visualize_and_save_sample(
    img_path,
    save_dir,
    frames,
    name=None,
    channel=1,
    **render_kwargs,
):
    """Render the chosen `frames` of a 4D cell movie at `img_path` (shape
    (C,T,Z,Y,X)) and save one PNG per frame."""
    os.makedirs(save_dir, exist_ok=True)
    movie = np.load(img_path)[channel]
    n_t = movie.shape[0]
    if name is None:
        name = os.path.splitext(os.path.basename(img_path))[0]

    saved = []
    for t in frames:
        if t < 0 or t >= n_t:
            raise ValueError(f"frame index {t} out of range [0, {n_t})")
        out = os.path.join(save_dir, f"{name}_{str(t).zfill(3)}.png")
        render_volume_png(movie[t], out, name=f"{name}_t{t}", **render_kwargs)
        saved.append(out)
        print(f"Saved {out}")
    return saved


if __name__ == "__main__":
    augmentations = build_augmentations()
    for cell_id in cell_ids:
        path = cell_path(data_dir, cell_id, window_id)
        visualize_augmentations(path, save_dir, frames, augmentations=augmentations)
