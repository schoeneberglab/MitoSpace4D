#!/usr/bin/env python3
"""
Napari visualizer for 3D/4D numpy volumes with optional per-channel overlays and 3D boxes.

Features
--------
- Accepts .npy or .tif/.tiff volumes.
- Shapes: (Z,Y,X) or (C,Z,Y,X); with --has_t also (T,Z,Y,X) or (T,C,Z,Y,X).
- Select one or more channels to display per file (defaults to channel 0).
- Automatic grid tiling (n columns) with configurable padding.
- Robust normalization: percent-clip and/or vmin/vmax; auto-handle [-1,1] data.
- Per-layer colormap, gamma, blending, MIP/translucent rendering.
- Optional world-scale (voxel spacing) for proper aspect.
- 3D bounding boxes via shapes layer (pass precomputed 8-corner boxes).

Usage
-----
python napari_visualizer.py \
  --img_paths a.npy b.npy \
  --channels 0 1 \
  --ncols 2 --pad 16 \
  --clip_percent 0.5 --rendering mip \
  --cmaps cyan magenta \
  --scale 1.0 0.2 0.2

Notes
-----
- Colormaps should match the number of channels you request; if fewer provided, they repeat.
- For time-series, use --has_t and optionally --t to pick a time index.
"""

from __future__ import annotations
import argparse
import math
import os
import os.path as osp
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import tifffile
except Exception:
    tifffile = None

import napari


# ------------------------- I/O -------------------------

def load_volume(path: str) -> np.ndarray:
    ext = osp.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".tif", ".tiff"):
        if tifffile is None:
            raise RuntimeError("tifffile is not installed; pip install tifffile")
        arr = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Loaded object is not a NumPy array: {path}")
    return arr


# ------------------------- Normalization -------------------------

def to_float01(
    vol: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip_percent: float = 0.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Convert volume to float32 in [0,1] with optional endpoint control.

    Heuristics:
      - If data looks like [-1, 1] (min < 0 and max <= ~1.5), map (x+1)/2.
      - Else apply percentile clipping if clip_percent > 0.
      - Else if vmin/vmax provided, use those.
      - Else min-max on the data.

    Returns normalized volume and (lo, hi) used for contrast_limits.
    """
    v = vol.astype(np.float32, copy=False)

    # Try the common normalized case [-1, 1]
    if v.min() < 0 and v.max() <= 1.5 and v.min() >= -1.5:
        v = 0.5 * (v + 1.0)
        v = np.clip(v, 0.0, 1.0)
        return v, (0.0, 1.0)

    # If explicit limits supplied
    if vmin is not None or vmax is not None:
        lo = float(vmin) if vmin is not None else float(np.nanmin(v))
        hi = float(vmax) if vmax is not None else float(np.nanmax(v))
        if hi <= lo:
            hi = lo + 1.0
        v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        return v, (0.0, 1.0)

    # Percentile clip (robust to outliers)
    if clip_percent and clip_percent > 0:
        p = float(clip_percent)
        lo = float(np.percentile(v, p))
        hi = float(np.percentile(v, 100.0 - p))
        if hi <= lo:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            if hi <= lo:  # constant array
                hi = lo + 1.0
        v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        return v, (0.0, 1.0)

    # Default: min-max
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi <= lo:
        hi = lo + 1.0
    v = (v - lo) / (hi - lo)
    v = np.clip(v, 0.0, 1.0)
    return v, (0.0, 1.0)


# ------------------------- Shapes / Boxes -------------------------

def boxes_to_paths(boxes: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Convert list of 3D boxes (8 corners, shape (8,3)) into path-ordered polylines.
    Each path will draw the 12 edges. Assumes conventional cube corner ordering
    or any ordering; we re-index edges by a fixed pair list.

    Corner indexing expected: the function works as long as `boxes` corners are consistent
    for each box (we do not rely on absolute orientation).
    """
    paths = []
    # 12 edges of a cube given corners 0..7 in a typical voxelgrid convention
    # We will attempt to infer edges using a simple spanning approach:
    # For robustness across corner orderings, we compute a minimal edge set by a kNN rule.
    # But to keep it deterministic and fast, we just connect a known pattern after
    # computing a local index permutation by sorting corners along axes.
    for B in boxes:
        if B.shape != (8, 3):
            raise ValueError("Each box must be of shape (8,3)")
        # Sort corners to get a consistent indexing (z,y,x lexicographic)
        order = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
        P = B[order]

        # Edges: bottom square (0-1-2-3-0), top square (4-5-6-7-4),
        # vertical edges (0-4,1-5,2-6,3-7)
        path_seq = [
            P[[0, 1, 2, 3, 0]],
            P[[4, 5, 6, 7, 4]],
            P[[0, 4]],
            P[[1, 5]],
            P[[2, 6]],
            P[[3, 7]],
        ]
        # Concatenate small breaks between segments by duplicating last point
        for seg in path_seq:
            paths.append(seg.astype(np.float32, copy=False))
    return paths


# ------------------------- Viewer helpers -------------------------

def ensure_cmaps(cmaps: List[str], need: int) -> List[str]:
    if not cmaps:
        cmaps = ["cyan"]
    if len(cmaps) < need:
        reps = math.ceil(need / len(cmaps))
        cmaps = (cmaps * reps)[:need]
    return cmaps


def add_volume_layer(
    viewer: napari.Viewer,
    vol_zyx: np.ndarray,
    *,
    name: str,
    translate_zyx: Tuple[float, float, float],
    colormap: str,
    rendering: str,
    gamma: float,
    scale_zyx: Tuple[float, float, float],
    blending: str = "additive",
    contrast_limits: Tuple[float, float] = (0.0, 1.0),
):
    """Add a single (Z,Y,X) volume as an image layer with proper 3D settings."""
    viewer.add_image(
        vol_zyx,
        name=name,
        translate=translate_zyx,   # (tz, ty, tx) in data units
        colormap=colormap,
        rendering=rendering,       # 'mip', 'translucent', 'additive', 'attenuated_mip'
        gamma=gamma,
        blending=blending,
        scale=scale_zyx,           # voxel spacing
        contrast_limits=contrast_limits,
    )


# ------------------------- Main -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Napari Visualizer")
    p.add_argument("--img_paths", nargs="+", required=True, help="Paths to images (.npy or .tif/.tiff)")
    p.add_argument("--channels", nargs="+", type=int, default=[0],
                   help="Channel indices to visualize (for (C,Z,Y,X) or (T,C,Z,Y,X)).")
    p.add_argument("--has_t", action="store_true",
                   help="Treat axis 0 as time when ndim==4, or axis 0 as time and axis 1 as channels when ndim==5.")
    p.add_argument("--t", type=int, default=0, help="Time index to visualize when --has_t is set.")
    p.add_argument("--ncols", type=int, default=2, help="Number of columns in the grid.")
    p.add_argument("--pad", type=int, default=12, help="Padding (in pixels) between tiles (x/y).")
    p.add_argument("--rendering", choices=["mip", "translucent", "additive", "attenuated_mip"],
                   default="mip", help="3D rendering method.")
    p.add_argument("--cmaps", nargs="+", default=["cyan", "magenta", "yellow", "green"],
                   help="Colormaps to cycle over for channels.")
    p.add_argument("--gamma", type=float, default=1.0, help="Display gamma.")
    p.add_argument("--scale", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   help="Voxel spacing as Z Y X (world units).")
    p.add_argument("--clip_percent", type=float, default=0.0,
                   help="Percent for symmetric clipping (e.g., 0.5 => [0.5, 99.5] percentiles).")
    p.add_argument("--vmin", type=float, default=None, help="Explicit lower bound before normalization.")
    p.add_argument("--vmax", type=float, default=None, help="Explicit upper bound before normalization.")
    p.add_argument("--name_prefix", type=str, default="Image", help="Prefix for layer names.")
    p.add_argument("--boxes_npy", type=str, default=None,
                   help="Optional .npy file containing a list/array of (8,3) boxes to overlay per image. "
                        "If provided and has shape (N,8,3) it will be used for every image; "
                        "if shape (M,N,8,3) and M==len(img_paths) it's per-image.")
    return p.parse_args()


def main():
    args = parse_args()
    for pth in args.img_paths:
        if not osp.exists(pth):
            raise FileNotFoundError(f"Missing file: {pth}")

    # Load optional boxes
    per_image_boxes: Optional[List[List[np.ndarray]]] = None
    if args.boxes_npy is not None:
        raw = np.load(args.boxes_npy, allow_pickle=True)
        # Normalize to List[List[(8,3)]]
        if raw.ndim == 3 and raw.shape[1:] == (8, 3):
            # Same set of boxes for all images
            per_image_boxes = [list(raw)] * len(args.img_paths)
        elif raw.ndim == 4 and raw.shape[2:] == (8, 3) and raw.shape[0] == len(args.img_paths):
            per_image_boxes = [list(raw[i]) for i in range(len(args.img_paths))]
        else:
            raise ValueError("boxes_npy has unsupported shape; expected (N,8,3) or (M,N,8,3) with M=len(img_paths).")

    # Preload and prepare volumes per path per channel
    prepared: List[List[Tuple[np.ndarray, Tuple[float, float]]]] = []  # [[(vol_zyx, (lo,hi)), ...], ...]
    yz_sizes: List[Tuple[int, int]] = []  # (Y, X) per image (max over channels)
    z_sizes: List[int] = []

    for path in args.img_paths:
        arr = load_volume(path)

        # Interpret shape
        if arr.ndim == 3:
            # (Z,Y,X)
            vols = [arr]
        elif arr.ndim == 4:
            if args.has_t:
                # (T,Z,Y,X)
                if args.t < 0 or args.t >= arr.shape[0]:
                    raise IndexError(f"--t {args.t} out of range for T={arr.shape[0]}")
                vols = [arr[args.t]]
            else:
                # (C,Z,Y,X)
                C = arr.shape[0]
                chs = [c for c in args.channels if 0 <= c < C]
                if not chs:
                    raise IndexError(f"No valid channels from {args.channels} for C={C} in {path}")
                vols = [arr[c] for c in chs]
        elif arr.ndim == 5:
            if not args.has_t:
                raise ValueError("5D input requires --has_t (interprets as (T,C,Z,Y,X) or (T,Z,Y,X)).")
            # Heuristic: if axis1 <= 4, treat (T,C,Z,Y,X); else (T,Z,Y,X,?) unsupported
            if arr.shape[1] <= 8:  # channel count likely small
                if args.t < 0 or args.t >= arr.shape[0]:
                    raise IndexError(f"--t {args.t} out of range for T={arr.shape[0]}")
                C = arr.shape[1]
                chs = [c for c in args.channels if 0 <= c < C]
                if not chs:
                    raise IndexError(f"No valid channels from {args.channels} for C={C} in {path}")
                vols = [arr[args.t, c] for c in chs]
            else:
                raise ValueError("Unsupported 5D shape; expected (T,C,Z,Y,X).")
        else:
            raise ValueError(f"Unsupported ndim={arr.ndim} for {path}")

        # Normalize each selected volume
        normed = [to_float01(v, vmin=args.vmin, vmax=args.vmax, clip_percent=args.clip_percent) for v in vols]
        prepared.append(normed)

        # Record sizes for tiling
        Ys = [v.shape[1] for v, _ in normed]
        Xs = [v.shape[2] for v, _ in normed]
        Zs = [v.shape[0] for v, _ in normed]
        yz_sizes.append((max(Ys), max(Xs)))
        z_sizes.append(max(Zs))

    # Grid step = max size across all images (uniform tile)
    maxY = max(y for y, _ in yz_sizes) if yz_sizes else 0
    maxX = max(x for _, x in yz_sizes) if yz_sizes else 0
    y_step = maxY + args.pad
    x_step = maxX + args.pad

    # Prepare colormaps
    max_ch_count = max(len(group) for group in prepared) if prepared else 1
    cmaps = ensure_cmaps(args.cmaps, max_ch_count)

    # Create viewer and add layers
    viewer = napari.Viewer()
    scale_zyx = tuple(float(s) for s in args.scale)  # (Z,Y,X)

    for i, group in enumerate(prepared):
        row = i // args.ncols
        col = i % args.ncols

        # Top-left of this tile in (ty, tx). Translate order must be (tz, ty, tx).
        ty = row * y_step
        tx = col * x_step

        # Add each selected channel as its own layer
        for ch_idx, (vol, clims) in enumerate(group):
            name = f"{args.name_prefix} [{osp.basename(args.img_paths[i])}] ch{ch_idx}"
            add_volume_layer(
                viewer,
                vol_zyx=vol,
                name=name,
                translate_zyx=(0.0, float(ty), float(tx)),
                colormap=cmaps[ch_idx],
                rendering=args.rendering,
                gamma=args.gamma,
                scale_zyx=scale_zyx,
                blending="additive",
                contrast_limits=clims,
            )

        # Add boxes if provided for this image
        if per_image_boxes is not None:
            boxes = per_image_boxes[i]
            if boxes:
                paths = boxes_to_paths(boxes)
                # Translate each vertex by (0, ty, tx) in Z,Y,X
                translated = [np.stack([p[:, 0], p[:, 1] + ty, p[:, 2] + tx], axis=1) for p in paths]
                viewer.add_shapes(
                    translated,
                    shape_type="path",
                    edge_color="red",
                    edge_width=2,
                    name=f"3D Boxes [{osp.basename(args.img_paths[i])}]",
                    scale=scale_zyx,
                    blending="translucent",
                )

    napari.run()


if __name__ == "__main__":
    main()
