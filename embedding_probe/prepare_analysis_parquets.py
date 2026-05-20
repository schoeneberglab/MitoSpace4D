#!/usr/bin/env python3
"""Build probe-ready parquets (labels + MitoTNT) for MS2D / MS3D."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]  # MitoSpace4D repo root
sys.path.insert(0, str(REPO_ROOT))


def join_labels_and_save(merged_path: Path, out_path: Path, ms4d_meta: Path) -> int:
    m = pd.read_parquet(merged_path)
    meta = pd.read_parquet(ms4d_meta, columns=["cell_id", "labels"])
    meta["cell_id"] = meta["cell_id"].astype(str)
    out = m.merge(meta, on="cell_id", how="inner")
    out.to_parquet(out_path, index=False)
    return len(out)


def prepare_ms2d() -> Path:
    run_dir = REPO_ROOT / "runs" / "ms2d_2024v3"
    merged = run_dir / "ms2d_merged_mitotnt_regression.parquet"
    if not merged.is_file():
        pooled = REPO_ROOT / "runs" / "mitospace_pooled_features.csv"
        if not (run_dir / "embeddings.npy").is_file():
            raise FileNotFoundError(f"Missing {run_dir / 'embeddings.npy'}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "paper.build_ms3d_mitotnt_regression_parquet",
                "--run_dir",
                str(run_dir),
                "--pooled_csv",
                str(pooled),
                "--output",
                str(merged),
            ]
        )
    out = run_dir / "ms2d_probe_analysis.parquet"
    n = join_labels_and_save(merged, out, REPO_ROOT / "runs" / "ms4d_2024v3_252eps" / "metadata.parquet")
    print(f"MS2D probe parquet: {out} ({n} rows)")
    return out


def prepare_ms3d() -> Path:
    run_dir = REPO_ROOT / "runs" / "ms3d_2024v3_225eps"
    merged = run_dir / "ms3d_merged_mitotnt_regression.parquet"
    external = Path("/Volumes/HP P500/4DMitoSpace Paper/ms3d_2024v3_225eps/ms3d_merged_mitotnt_regression.parquet")

    if not merged.is_file() and external.is_file():
        import shutil

        shutil.copy2(external, merged)
        print(f"Copied MS3D merged parquet from external drive -> {merged}")

    if not merged.is_file():
        if not (run_dir / "embeddings.npy").is_file():
            raise FileNotFoundError(
                f"MS3D embeddings missing. Mount external drive or place embeddings.npy + image_paths.csv in {run_dir}"
            )
        pooled = REPO_ROOT / "runs" / "mitospace_pooled_features.csv"
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "paper.build_ms3d_mitotnt_regression_parquet",
                "--run_dir",
                str(run_dir),
                "--pooled_csv",
                str(pooled),
                "--output",
                str(merged),
            ]
        )

    out = run_dir / "ms3d_probe_analysis.parquet"
    n = join_labels_and_save(merged, out, REPO_ROOT / "runs" / "ms4d_2024v3_252eps" / "metadata.parquet")
    print(f"MS3D probe parquet: {out} ({n} rows)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ms2d", "ms3d", "all"], default="all")
    args = ap.parse_args()
    models = ["ms2d", "ms3d"] if args.model == "all" else [args.model]
    for m in models:
        if m == "ms2d":
            prepare_ms2d()
        else:
            prepare_ms3d()


if __name__ == "__main__":
    main()
