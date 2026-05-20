#!/usr/bin/env python3
"""Run embedding probe for MS4D / MS3D / MS2D."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedding_probe.probe import data_utils as du
from embedding_probe.probe import experiments as ex
from embedding_probe.probe import plotting as P
from embedding_probe.probe.model_configs import get_model_config
from embedding_probe.probe.report import render_comparative, render_model

ALL_EXPERIMENTS = ["e1", "e2", "e4", "e6", "e7", "e8", "e9", "e10", "e11"]
NEEDS_RANDOM = {"e2", "e7"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MitoSpace embedding probe")
    p.add_argument("--model", default="ms4d", choices=["ms4d", "ms3d", "ms2d", "4d", "3d", "2d"])
    p.add_argument("--experiments", default="all", help="Comma list or 'all'")
    p.add_argument("--trained-parquet", default=None, help="Override trained embeddings parquet")
    p.add_argument("--random-parquet", default=None, help="Override random-init parquet (4D only)")
    p.add_argument("--colors", default=str(REPO_ROOT / "metadata" / "colors.txt"))
    p.add_argument("--control-label", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-random", action="store_true")
    p.add_argument("--subsample-geometry", type=int, default=20000)
    p.add_argument("--replicates-per-class", type=int, default=20)
    p.add_argument("--min-reps-e10", type=int, default=50)
    return p.parse_args()


def run_probe_for_model(args: argparse.Namespace) -> None:
    cfg = get_model_config(args.model)
    trained_path = Path(args.trained_parquet) if args.trained_parquet else cfg.trained_parquet
    random_path = Path(args.random_parquet) if args.random_parquet else cfg.random_parquet

    requested = (
        ALL_EXPERIMENTS
        if args.experiments == "all"
        else [e.strip().lower() for e in args.experiments.split(",") if e.strip()]
    )

    P.set_plots_dir(REPO_ROOT / "embedding_probe" / "results" / cfg.model_id / "plots")

    log(f"[{cfg.display_name}] Loading {trained_path}")
    trained_df = du.load_trained_bundle(trained_path)
    x_trained = du.stack_embeddings(trained_df["embeddings"])
    labels = trained_df["labels"].to_numpy(dtype=np.int32)
    label_names = du.load_label_names(args.colors)
    log(f"  {len(trained_df):,} cells, {len(np.unique(labels))} drugs, dim={x_trained.shape[1]}")

    x_random = None
    if not args.no_random and random_path and os.path.isfile(random_path):
        log(f"  Loading random-init from {random_path}")
        x_trained, x_random, labels, trained_df = du.load_aligned_trained_random_df(
            trained_path, random_path, label_names_path=args.colors
        )
    elif cfg.random_parquet and not args.no_random:
        log(f"  WARNING: random parquet not found for {cfg.model_id}")

    ctx = ex.ProbeContext(
        trained_df=trained_df,
        label_names=label_names,
        x_trained=x_trained,
        model_id=cfg.model_id,
        display_name=cfg.display_name,
        x_random=x_random,
        labels=labels,
        control_label=args.control_label,
        seed=args.seed,
    )

    runners = {
        "e1": lambda: ex.experiment_1_geometry(ctx, subsample=args.subsample_geometry),
        "e2": lambda: ex.experiment_2_trained_vs_random(ctx),
        "e4": lambda: ex.experiment_4_discriminability(
            ctx, n_classes=len(np.unique(labels)), replicates_per_class=args.replicates_per_class
        ),
        "e6": lambda: ex.experiment_6_pca_viz(ctx),
        "e7": lambda: ex.experiment_7_baseline_matrix(ctx),
        "e8": lambda: ex.experiment_8_treatment_viz(ctx),
        "e9": lambda: ex.experiment_9_constancy(ctx),
        "e10": lambda: ex.experiment_10_cossim_by_condition(ctx, min_reps=args.min_reps_e10),
        "e11": lambda: ex.experiment_11_phenotype_vs_gap(ctx, min_reps=args.min_reps_e10),
    }

    for eid in requested:
        if eid not in runners:
            raise ValueError(f"Unknown experiment {eid}")
        if eid in NEEDS_RANDOM and x_random is None:
            log(f"  Skipping {eid.upper()} (no random-init baseline)")
            continue
        log(f"  Running {eid.upper()} ...")
        t0 = time.time()
        findings = runners[eid]()
        log(f"    done in {time.time() - t0:.1f}s")

    render_model(cfg.model_id)
    log(f"  Wrote {ctx.results_dir / 'findings.md'}")


def main() -> None:
    args = parse_args()
    run_probe_for_model(args)


if __name__ == "__main__":
    main()
