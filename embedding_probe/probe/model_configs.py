"""Per-model paths for the embedding probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    trained_parquet: Path
    random_parquet: Path | None
    embedding_dim: int | None = None


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def get_model_config(model_id: str) -> ModelConfig:
    mid = model_id.lower()
    if mid in ("ms4d", "4d"):
        return ModelConfig(
            model_id="ms4d",
            display_name="MitoSpace4D (4D)",
            trained_parquet=REPO_ROOT / "runs" / "ms4d_2024v3_252eps" / "metadata.parquet",
            random_parquet=REPO_ROOT
            / "runs"
            / "ms4d_2024v3_random_init"
            / "ms4d_random_init_mitotnt_regression.parquet",
            embedding_dim=2048,
        )
    if mid in ("ms2d", "2d"):
        p = _first_existing(
            REPO_ROOT / "runs" / "ms2d_2024v3" / "ms2d_probe_analysis.parquet",
            REPO_ROOT / "runs" / "ms2d_2024v3" / "ms2d_merged_mitotnt_regression.parquet",
            REPO_ROOT / "runs" / "ms2d_2024v3" / "embeddings+metadata.parquet",
        )
        if p is None:
            raise FileNotFoundError("No MS2D parquet found under runs/ms2d_2024v3/")
        return ModelConfig(
            model_id="ms2d",
            display_name="MitoSpace2D (2D)",
            trained_parquet=p,
            random_parquet=None,
            embedding_dim=512,
        )
    if mid in ("ms3d", "3d"):
        p = _first_existing(
            REPO_ROOT / "runs" / "ms3d_2024v3_225eps" / "ms3d_probe_analysis.parquet",
            REPO_ROOT / "runs" / "ms3d_2024v3_225eps" / "ms3d_merged_mitotnt_regression.parquet",
            Path("/Volumes/HP P500/4DMitoSpace Paper/ms3d_2024v3_225eps/ms3d_merged_mitotnt_regression.parquet"),
        )
        if p is None:
            raise FileNotFoundError(
                "MS3D parquet not found. Build with:\n"
                "  python -m paper.build_ms3d_mitotnt_regression_parquet --run_dir <ms3d_dir>\n"
                "  python embedding_probe/prepare_analysis_parquets.py --model ms3d"
            )
        return ModelConfig(
            model_id="ms3d",
            display_name="MitoSpace3D (3D)",
            trained_parquet=p,
            random_parquet=None,
            embedding_dim=2048,
        )
    raise ValueError(f"Unknown model {model_id!r}; use ms4d, ms3d, or ms2d")


ALL_MODELS = ("ms4d", "ms3d", "ms2d")
