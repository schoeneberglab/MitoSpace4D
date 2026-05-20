# MitoSpace embedding probe

Embedding-space diagnostics for **MitoSpace4D (4D)**, **MitoSpace3D (3D)**, and **MitoSpace2D (2D)**, using the same experimental framing as [what-does-openphenom-learn](https://github.com/drv-agwl/what-does-openphenom-learn) and Svatko et al. (2026), [*Deep Learning for BioImaging: What Are We Learning?*](https://arxiv.org/abs/2603.13377).

## Quick start

```bash
cd MitoSpace4D
conda activate deeplearning

# Prepare 2D/3D analysis parquets (joins labels + MitoTNT from MS4D metadata)
python embedding_probe/prepare_analysis_parquets.py --model all

# Run all models + comparative findings.md
python embedding_probe/run_all_models.py

# Or single model:
python embedding_probe/run.py --model ms4d --experiments all
python embedding_probe/run.py --model ms2d --experiments all --no-random
```

## Outputs

| Path | Description |
|------|-------------|
| `results/findings.md` | **Comparative report** (4D vs 3D vs 2D vs OpenPhenom) |
| `results/ms4d/findings.md` | Per-model report + `numbers.json` + `plots/` |
| `results/ms2d/…` | 2D results |
| `results/ms3d/…` | 3D results (after data available) |

## Data requirements

| Model | Trained embeddings | Random baseline (E2, E7) |
|-------|-------------------|---------------------------|
| **MS4D** | `runs/ms4d_2024v3_252eps/metadata.parquet` | `runs/ms4d_2024v3_random_init/ms4d_random_init_mitotnt_regression.parquet` |
| **MS2D** | `runs/ms2d_2024v3/ms2d_probe_analysis.parquet` (built by `prepare_analysis_parquets.py`) | — |
| **MS3D** | `runs/ms3d_2024v3_225eps/ms3d_probe_analysis.parquet` | — |

**MS3D:** Requires `embeddings.npy` + `image_paths.csv` under `runs/ms3d_2024v3_225eps/`, or copy `ms3d_merged_mitotnt_regression.parquet` from external storage, then run `prepare_analysis_parquets.py --model ms3d`.

## Experiments

| ID | Name | Images? |
|----|------|---------|
| E1 | Geometry (PCA, participation ratio, pairwise cos-sim) | No |
| E2 | Trained vs random-init (Spearman ρ, mAP) | No (4D only today) |
| E4 | Discriminability + PCA-CenterScale | No |
| E6 | PCA scatter by drug | No |
| E7 | Baseline mAP comparison | No (4D only) |
| E8 | Drug cosine heatmap | No |
| E9 | Per-dimension variability | No |
| E10 | Within vs between cos-sim (Cohen's d, AUROC) | No |
| E11 | MitoTNT unusualness vs per-drug gap | No |

Image-based probes (invariances, pathological inputs) are deferred per project plan.

## OpenPhenom comparison

Reference scalars live in `openphenom_reference.json` (from the public [OpenPhenom probe repo](https://github.com/drv-agwl/what-does-openphenom-learn)). The comparative table is written to `results/findings.md`.
