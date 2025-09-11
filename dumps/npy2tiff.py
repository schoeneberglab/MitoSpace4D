#!/usr/bin/env python3
"""
Convert .npy files to .tiff images.

Usage examples
--------------
# Single file
python npy_to_tiff.py --infile /path/in.npy --outfile /path/out.tiff

# Flat directory (no recursion)
python npy_to_tiff.py --src /data/src --dst /data/dst

# Recursive, mirroring the src directory tree under dst
python npy_to_tiff.py --src /data/src --dst /data/dst --recurse

# Recursive with overwrite
python npy_to_tiff.py --src /data/src --dst /data/dst --recurse --overwrite
"""

import argparse
import glob
import os
import os.path as osp
from typing import Iterator

import numpy as np
import tifffile


def npy_to_tiff(infile: str, outfile: str, overwrite: bool = False) -> None:
    """Load a .npy file and save it as a .tiff file."""
    if (not overwrite) and osp.exists(outfile):
        print(f"[skip] Exists: {outfile}")
        return

    # Ensure parent directory exists
    out_dir = osp.dirname(outfile)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load array and write TIFF.
    # tifffile handles nD arrays (e.g., (Z, Y, X) or (C, Z, Y, X)).
    data = np.load(infile)
    # Use BigTIFF automatically if needed; keep dtype as-is.
    tifffile.imwrite(outfile, data)
    print(f"[ok] {infile} -> {outfile}")


def iter_npy_files(src: str, recurse: bool) -> Iterator[str]:
    """Yield .npy file paths from src, optionally recursively."""
    if recurse:
        for root, _dirs, files in os.walk(src):
            for name in files:
                if name.lower().endswith(".npy"):
                    yield osp.join(root, name)
    else:
        # Only the top level of src
        yield from glob.iglob(osp.join(src, "*.npy"))


def main():
    parser = argparse.ArgumentParser(description="Convert .npy files to .tiff format.")
    parser.add_argument("--infile", help="Path to the input .npy file")
    parser.add_argument("--outfile", help="Path to the output .tiff file")
    parser.add_argument("--src", help="Source directory")
    parser.add_argument("--dst", help="Destination directory")
    parser.add_argument("--recurse", action="store_true", help="Recurse into subdirectories and mirror tree")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    # Mode 1: directory mode
    if args.src and args.dst:
        if not osp.isdir(args.src):
            raise SystemExit(f"Source is not a directory: {args.src}")

        for infile in iter_npy_files(args.src, args.recurse):
            # Determine output path
            if args.recurse:
                # Mirror src tree under dst
                rel_dir = osp.relpath(osp.dirname(infile), start=args.src)
                out_dir = args.dst if rel_dir == "." else osp.join(args.dst, rel_dir)
            else:
                # Flat output into dst
                out_dir = args.dst

            base = osp.splitext(osp.basename(infile))[0]
            outfile = osp.join(out_dir, base + ".tiff")
            npy_to_tiff(infile, outfile, overwrite=args.overwrite)

    # Mode 2: single-file mode
    elif args.infile and args.outfile:
        npy_to_tiff(args.infile, args.outfile, overwrite=args.overwrite)

    else:
        raise SystemExit(
            "Specify either (--infile and --outfile) for a single file, or (--src and --dst) for a directory."
        )


if __name__ == "__main__":
    main()
