# #!/usr/bin/env python3
# """
# Rotate a point cloud (PCD: point cloud data) about the z-axis and save an MP4.

# Requirements:
#   pip install open3d imageio[ffmpeg] pillow

# Usage examples
# --------------
# # Best color fidelity (no subsampling):
# python spin_movie.py --pcd mitospace4d.pcd --out mitospace4d.mp4 --hd --ssaa 2 --pix-fmt yuv444p --crf 18 --preset slow

# # Lossless-like RGB path (very large files):
# python spin_movie.py --pcd mitospace4d.pcd --out mitospace4d_rgb.mp4 --pix-fmt rgb24 --crf 0 --preset slow
# """

# import argparse
# import math
# from pathlib import Path

# import imageio
# import numpy as np
# import open3d as o3d
# from PIL import Image  # high-quality downscale (LANCZOS)


# def parse_args():
#     p = argparse.ArgumentParser(description="Render a z-axis rotation video of a point cloud (PCD).")
#     p.add_argument("--pcd", required=True, type=Path, help="Input point cloud (.pcd/.ply/.xyz/...)")
#     p.add_argument("--out", required=True, type=Path, help="Output video (e.g., out.mp4)")
#     p.add_argument("--fps", type=int, default=30, help="Frames per second (FPS)")
#     p.add_argument("--duration", type=float, default=15.0, help="Duration (seconds)")
#     p.add_argument("--width", type=int, default=1920, help="Target render width (px) [final video size]")
#     p.add_argument("--height", type=int, default=1080, help="Target render height (px) [final video size]")
#     p.add_argument("--hd", action="store_true", help="Force 1920x1080 output (overrides width/height)")
#     p.add_argument("--ssaa", type=int, default=1,
#                    help="SSAA factor: render at (width*ssaa, height*ssaa) then downsample")
#     p.add_argument("--fov-deg", type=float, default=50.0, help="Vertical FOV (degrees)")
#     p.add_argument("--elev-deg", type=float, default=0.0, help="Camera elevation above xy-plane (degrees)")
#     p.add_argument("--zoom", type=float, default=1.0,
#                    help="Zoom factor via camera distance. 1.0=default, >1.0 closer, <1.0 farther")
#     p.add_argument("--bg", choices=["black", "white"], default="white", help="Background color")
#     p.add_argument("--point-size", type=float, default=3, help="Point size (px at final resolution)")
#     # Encoding controls
#     p.add_argument("--codec", type=str,
#                    default=None, 
#                 #    default="libx264rgb",
#                    help="FFmpeg codec (default auto: libx264 or libx264rgb for rgb24)")
#     p.add_argument("--crf", type=int, default=18,
#                    help="x264 CRF (18~20 ≈ visually lossless). For libx264rgb, 0=lossless.")
#     # Default to yuv444p to avoid color subsampling artifacts:
#     p.add_argument("--pix-fmt", type=str, 
#                    default="yuv444p",
#                     # default="rgb24",
#                    help="Pixel format: yuv444p (best color), rgb24 (RGB path), or yuv420p (max compatibility)")
#     p.add_argument("--preset", type=str, default="medium",
#                    help="x264 preset: ultrafast..placebo (slower = better compression)")
#     return p.parse_args()


# def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
#     if not path.exists():
#         raise FileNotFoundError(f"Point cloud not found: {path}")
#     pcd = o3d.io.read_point_cloud(str(path))
#     if pcd.is_empty():
#         raise ValueError("Loaded point cloud has zero points.")
#     if not pcd.has_colors():
#         n = np.asarray(pcd.points).shape[0]
#         pcd.colors = o3d.utility.Vector3dVector(np.tile([0.85, 0.85, 0.85], (n, 1)))
#     return pcd


# def create_renderer(width: int, height: int, bg: str) -> o3d.visualization.rendering.OffscreenRenderer:
#     renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     renderer.scene.set_background([1, 1, 1, 1] if bg == "white" else [0, 0, 0, 1])
#     try:
#         aa_enum = o3d.visualization.rendering.AntiAliasing
#         renderer.scene.set_antialiasing(aa_enum.FXAA)
#     except Exception:
#         pass
#     return renderer


# def add_geometry(renderer, name: str, geom: o3d.geometry.Geometry, point_size_px_final: float, ssaa: int):
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = "defaultUnlit"
#     mat.point_size = float(point_size_px_final) * max(1, ssaa)
#     try:
#         mat.point_size_units = o3d.visualization.rendering.PointSizeUnits.PIXELS
#     except Exception:
#         pass
#     renderer.scene.add_geometry(name, geom, mat)


# def setup_camera_look_at(renderer, center, radius, elev_deg, theta_deg, fov_deg, width, height, zoom):
#     elev = math.radians(elev_deg)
#     theta = math.radians(theta_deg)
#     base_dist = 1.8 * max(radius, 1e-6)
#     dist = base_dist / max(1e-3, float(zoom))

#     eye = center + np.array([
#         dist * math.cos(theta) * math.cos(elev),
#         dist * math.sin(theta) * math.cos(elev),
#         dist * math.sin(elev),
#     ], dtype=float)
#     up = np.array([0.0, 0.0, 1.0], dtype=float)

#     cam = renderer.scene.camera
#     aspect = float(width) / float(height) if height > 0 else (16.0 / 9.0)
#     near = max(1e-3, 0.02 * radius)
#     far = max(10.0 * radius, dist + 5.0 * radius) if radius > 0 else 1000.0

#     cam.set_projection(float(fov_deg), float(aspect), float(near), float(far),
#                        o3d.visualization.rendering.Camera.FovType.Vertical)
#     cam.look_at(center, eye, up)


# def downsample_to_target(frame_ssaa: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
#     mode = "RGB" if (frame_ssaa.ndim == 3 and frame_ssaa.shape[2] == 4) else "RGB"
#     img = Image.fromarray(frame_ssaa, mode=mode)
#     img = img.resize((out_w, out_h), resample=Image.LANCZOS)
#     arr = np.asarray(img)
#     if arr.ndim == 3 and arr.shape[2] == 4:
#         arr = arr[..., :3]
#     # Ensure uint8 for encoder
#     if arr.dtype != np.uint8:
#         arr = np.clip(arr, 0, 255).astype(np.uint8)
#     return arr


# def main():
#     args = parse_args()
#     if args.hd:
#         args.width, args.height = 1920, 1080

#     pcd = load_point_cloud(args.pcd)

#     aabb = pcd.get_axis_aligned_bounding_box()
#     center = np.asarray(aabb.get_center(), dtype=float)
#     radius = float(np.linalg.norm(aabb.get_extent()))
#     radius = max(radius, 1e-6)

#     ssaa = max(1, int(args.ssaa))
#     render_w, render_h = args.width * ssaa, args.height * ssaa

#     renderer = create_renderer(render_w, render_h, args.bg)
#     add_geometry(renderer, "pcd", pcd, args.point_size, ssaa)

#     n_frames = max(1, int(round(args.fps * args.duration)))
#     args.out.parent.mkdir(parents=True, exist_ok=True)

#     # Decide codec automatically based on pixel format unless explicitly set.
#     codec = args.codec
#     if codec is None:
#         codec = "libx264rgb" if args.pix_fmt.lower() == "rgb24" else "libx264"

#     # FFmpeg parameters that help keep colors consistent.
#     ffmpeg_params = [
#         "-crf", str(args.crf),
#         "-preset", str(args.preset),
#         "-pix_fmt", str(args.pix_fmt),
#         # Color metadata: BT.709 full range for HD computer graphics
#         "-colorspace", "bt709",
#         "-color_trc", "bt709",
#         "-color_primaries", "bt709",
#         "-movflags", "+faststart",
#     ]
#     # For YUV paths, prefer full-range if your player honors it (most do).
#     if args.pix_fmt.lower().startswith("yuv"):
#         ffmpeg_params += ["-color_range", "pc"]

#     writer = imageio.get_writer(
#         str(args.out),
#         fps=args.fps,
#         codec=codec,
#         quality=None,
#         macro_block_size=None,
#         ffmpeg_params=ffmpeg_params,
#     )

#     try:
#         for i in range(n_frames):
#             theta_deg = 360.0 * (i / n_frames)
#             setup_camera_look_at(renderer, center, radius,
#                                  args.elev_deg, theta_deg, args.fov_deg,
#                                  render_w, render_h, args.zoom)

#             img = renderer.render_to_image()
#             frame_ssaa = np.asarray(img)
#             frame = downsample_to_target(frame_ssaa, args.width, args.height)
#             writer.append_data(frame)
#     finally:
#         writer.close()
#         try:
#             renderer.scene.remove_geometry("pcd")
#         except Exception:
#             pass
#         del renderer

#     print(f"Saved video to: {args.out.resolve()}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Rotate a point cloud (PCD: point cloud data) about the z-axis and save an MP4, with
color-preserving encode options for highly saturated colors.

Requirements:
  pip install open3d imageio[ffmpeg] pillow
"""

import argparse
import math
from pathlib import Path

import imageio
import numpy as np
import open3d as o3d
from PIL import Image  # high-quality downscale (LANCZOS)


def parse_args():
    p = argparse.ArgumentParser(description="Render a z-axis rotation video of a point cloud (PCD).")
    p.add_argument("--pcd", required=True, type=Path, help="Input point cloud (.pcd/.ply/.xyz/...)")
    p.add_argument("--out", required=True, type=Path, help="Output video (e.g., out.mp4)")
    p.add_argument("--fps", type=int, default=30, help="Frames per second (FPS)")
    p.add_argument("--duration", type=float, default=15.0, help="Duration (seconds)")

    # Output resolution and AA
    p.add_argument("--width", type=int, default=1920, help="Target render width (px) [final video size]")
    p.add_argument("--height", type=int, default=1080, help="Target render height (px) [final video size]")
    p.add_argument("--hd", action="store_true", help="Force 1920x1080 output (overrides width/height)")
    p.add_argument("--ssaa", type=int, default=1,
                   help="SSAA factor: render at (width*ssaa, height*ssaa) then downsample")

    # Camera
    p.add_argument("--fov-deg", type=float, default=50.0, help="Vertical field of view (degrees)")
    p.add_argument("--elev-deg", type=float, default=0.0, help="Camera elevation above xy-plane (degrees)")
    p.add_argument("--zoom", type=float, default=1.0,
                   help="Zoom factor via camera distance. 1.0=default, >1.0 closer, <1.0 farther")

    # Appearance
    p.add_argument("--bg", choices=["black", "white"], default="white", help="Background color")
    p.add_argument("--point-size", type=float, default=2.5, help="Point size (px at final resolution)")

    # Encoding – defaults chosen to preserve saturated colors
    p.add_argument("--pix-fmt", type=str, default="yuv444p",
                   help="Pixel format: yuv444p (best color), rgb24 (RGB path), yuv420p (max compat)")
    p.add_argument("--bit-depth", type=int, default=8, choices=[8, 10],
                   help="Bit depth for YUV encodes (10-bit reduces banding/clipping).")
    p.add_argument("--codec", type=str, default="auto",
                   help="Codec: auto|libx264|libx264rgb|libx265|prores_ks")
    p.add_argument("--crf", type=int, default=18,
                   help="CRF quality target (x264/x265). For libx264rgb, 0=lossless.")
    p.add_argument("--preset", type=str, default="medium",
                   help="x264/x265 preset: ultrafast..placebo (slower = better compression)")
    p.add_argument("--dither", choices=["none", "ordered"], default="none",
                   help="Apply ordered dithering before 8-bit encode to reduce banding.")

    return p.parse_args()


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError("Loaded point cloud has zero points.")
    if not pcd.has_colors():
        n = np.asarray(pcd.points).shape[0]
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.85, 0.85, 0.85], (n, 1)))
    return pcd


def create_renderer(width: int, height: int, bg: str) -> o3d.visualization.rendering.OffscreenRenderer:
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1] if bg == "white" else [0, 0, 0, 1])
    try:
        aa_enum = o3d.visualization.rendering.AntiAliasing
        renderer.scene.set_antialiasing(aa_enum.FXAA)
    except Exception:
        pass
    return renderer


def add_geometry(renderer, name: str, geom: o3d.geometry.Geometry, point_size_px_final: float, ssaa: int):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size_px_final) * max(1, ssaa)
    try:
        mat.point_size_units = o3d.visualization.rendering.PointSizeUnits.PIXELS
    except Exception:
        pass
    renderer.scene.add_geometry(name, geom, mat)


def setup_camera_look_at(renderer, center, radius, elev_deg, theta_deg, fov_deg, width, height, zoom):
    elev = math.radians(elev_deg)
    theta = math.radians(theta_deg)
    base_dist = 1.8 * max(radius, 1e-6)
    dist = base_dist / max(1e-3, float(zoom))

    eye = center + np.array([
        dist * math.cos(theta) * math.cos(elev),
        dist * math.sin(theta) * math.cos(elev),
        dist * math.sin(elev),
    ], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    cam = renderer.scene.camera
    aspect = float(width) / float(height) if height > 0 else (16.0 / 9.0)
    near = max(1e-3, 0.02 * radius)
    far = max(10.0 * radius, dist + 5.0 * radius) if radius > 0 else 1000.0

    cam.set_projection(float(fov_deg), float(aspect), float(near), float(far),
                       o3d.visualization.rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)


def downsample_to_target(frame_ssaa: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Downsample supersampled RGB(A) frame to target size using high-quality LANCZOS."""
    mode = "RGBA" if (frame_ssaa.ndim == 3 and frame_ssaa.shape[2] == 4) else "RGB"
    img = Image.fromarray(frame_ssaa, mode=mode)
    img = img.resize((out_w, out_h), resample=Image.LANCZOS)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


# 8x8 Bayer matrix for ordered dithering (values scaled 0..63)
_BAYER8 = (1/64.0) * np.array([
    [ 0, 48, 12, 60,  3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [ 8, 56,  4, 52, 11, 59,  7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [ 2, 50, 14, 62,  1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58,  6, 54,  9, 57,  5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
], dtype=np.float32)


def ordered_dither_8bit(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Apply simple ordered dithering to reduce banding when encoding vivid gradients in 8-bit.
    Operates per-channel in sRGB space (fast; good enough for rendering output).
    """
    h, w = rgb_uint8.shape[:2]
    tiled = np.tile(_BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    # Add small thresholded offset (±0.5 LSB) before quantization boundaries
    # Map to [-0.5, +0.5] LSB in 8-bit
    jitter = (tiled - 0.5).astype(np.float32)
    out = rgb_uint8.astype(np.float32) + jitter[..., None] * 255.0 / 256.0
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def choose_encode_params(pix_fmt: str, bit_depth: int, codec_choice: str):
    """Decide codec and ffmpeg params that preserve saturated colors well."""
    pix = pix_fmt.lower()
    codec = codec_choice.lower()

    # Auto codec choice
    if codec == "auto":
        if pix == "rgb24":
            codec = "libx264rgb"
        elif bit_depth == 10:
            # Prefer intra 10-bit 4:4:4 for fidelity
            codec = "prores_ks"
        else:
            codec = "libx264"

    ff = [
        "-pix_fmt", pix,
        "-colorspace", "bt709",
        "-color_trc", "bt709",
        "-color_primaries", "bt709",
        "-movflags", "+faststart",
    ]

    # Adjust for 10-bit choices
    if bit_depth == 10:
        if codec == "prores_ks":
            # Force 4:4:4 10-bit; great for saturated colors, intra-frame
            ff = [
                "-pix_fmt", "yuv444p10le",
                "-profile:v", "4444",
                "-qscale:v", "7",   # quality; lower is better (1..31), tweak as needed
                "-vendor", "apl0",
                "-colorspace", "bt709",
                "-color_trc", "bt709",
                "-color_primaries", "bt709",
                "-movflags", "+faststart",
            ]
        elif codec == "libx265":
            # H.265 4:4:4 10-bit path
            ff = [
                "-pix_fmt", "yuv444p10le",
                "-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
                "-movflags", "+faststart",
            ]
        # libx264 10-bit builds exist, but not always available; prefer x265/prores for 10-bit.

    return codec, ff


def main():
    args = parse_args()
    if args.hd:
        args.width, args.height = 1920, 1080

    pcd = load_point_cloud(args.pcd)

    # Scene extents
    aabb = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center(), dtype=float)
    radius = float(np.linalg.norm(aabb.get_extent()))
    radius = max(radius, 1e-6)

    # Supersampled render size
    ssaa = max(1, int(args.ssaa))
    render_w, render_h = args.width * ssaa, args.height * ssaa

    renderer = create_renderer(render_w, render_h, args.bg)
    add_geometry(renderer, "pcd", pcd, args.point_size, ssaa)

    n_frames = max(1, int(round(args.fps * args.duration)))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Decide codec + ffmpeg parameters
    codec, base_ff = choose_encode_params(args.pix_fmt, args.bit_depth, args.codec)

    # Build final ffmpeg params (CRF applies to x264/x265 paths)
    ffmpeg_params = list(base_ff)
    if codec in ("libx264", "libx265", "libx264rgb"):
        ffmpeg_params += ["-crf", str(args.crf), "-preset", str(args.preset)]
        # Prefer full-range for YUV paths when possible
        if args.pix_fmt.lower().startswith("yuv"):
            ffmpeg_params += ["-color_range", "pc"]

    writer = imageio.get_writer(
        str(args.out),
        fps=args.fps,
        codec=codec,
        quality=None,
        macro_block_size=None,
        ffmpeg_params=ffmpeg_params,
    )

    try:
        for i in range(n_frames):
            theta_deg = 360.0 * (i / n_frames)
            setup_camera_look_at(renderer, center, radius,
                                 args.elev_deg, theta_deg, args.fov_deg,
                                 render_w, render_h, args.zoom)

            img = renderer.render_to_image()
            frame_ssaa = np.asarray(img)  # uint8, RGB(A)
            frame = downsample_to_target(frame_ssaa, args.width, args.height)

            # Optional dithering before 8-bit encode (helps saturated gradients when not using 10-bit)
            if args.bit_depth == 8 and args.dither == "ordered":
                frame = ordered_dither_8bit(frame)

            writer.append_data(frame)
    finally:
        writer.close()
        try:
            renderer.scene.remove_geometry("pcd")
        except Exception:
            pass
        del renderer

    print(f"Saved video to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
