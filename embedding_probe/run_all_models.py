#!/usr/bin/env python3
"""Run embedding probe for 4D, 3D, and 2D; then write comparative findings.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE = REPO_ROOT / "embedding_probe" / "prepare_analysis_parquets.py"
RUN = REPO_ROOT / "embedding_probe" / "run.py"


def main() -> None:
    py = sys.executable

    # Ensure 2D/3D analysis parquets exist where possible
    for model in ("ms2d", "ms3d"):
        try:
            subprocess.run([py, str(PREPARE), "--model", model], check=True, cwd=REPO_ROOT)
        except subprocess.CalledProcessError:
            print(f"WARNING: could not prepare {model} (missing data?)", file=sys.stderr)

    for model in ("ms4d", "ms3d", "ms2d"):
        cfg_path = REPO_ROOT / "embedding_probe" / "probe" / "model_configs.py"
        try:
            from embedding_probe.probe.model_configs import get_model_config

            get_model_config(model)
        except FileNotFoundError as e:
            print(f"Skipping {model}: {e}", file=sys.stderr)
            continue

        print(f"\n========== {model.upper()} ==========\n")
        subprocess.run(
            [py, str(RUN), "--model", model, "--experiments", "all"],
            check=False,
            cwd=REPO_ROOT,
        )

    from embedding_probe.probe.report import render_comparative

    out = render_comparative()
    print(f"\nComparative report: {out}")


if __name__ == "__main__":
    main()
