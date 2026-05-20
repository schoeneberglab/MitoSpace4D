# Embedding probe — comparative findings

Cross-study of **MitoSpace 4D / 3D / 2D** self-supervised embeddings vs **OpenPhenom** (Cell Painting, RxRx3-core), using the experimental framing from [what-does-openphenom-learn](https://github.com/drv-agwl/what-does-openphenom-learn) and Svatko et al. (2026) [*Deep Learning for BioImaging: What Are We Learning?*](https://arxiv.org/abs/2603.13377).

> OpenPhenom numbers are from the reference run documented in the external probe repository (~2.4k wells, 300 treatments). MitoSpace numbers use the 2024v3 drug screen (~36.6k cells, 25 drugs, shared `cell_id` cohort where noted).

## Summary table

| Metric | OpenPhenom | MS4D (4D) | MS3D (3D) | MS2D (2D) |
|--------|------------|-----------|-----------|-----------|
| Participation ratio (effective rank) | 1.8 | 16 | 14.69 | 19.52 |
| Top-1 PC variance share | 0.74 | 0.1291 | 0.153 | 0.1387 |
| Mean pairwise cosine similarity | 0.94 | 0.2344 | 0.2204 | 0.2161 |
| Within−between gap (trained) | 0.022 | 0.2653 | 0.2672 | 0.2114 |
| Spearman ρ vs random baseline | None | 0.005944 | — | — |
| Replicate retrieval mAP | None | 0.3008 | — | — |
| Replicate mAP (balanced sample, raw) | 0.021 | 0.2957 | 0.2926 | 0.2656 |
| Replicate mAP after PCA-CenterScale | 0.03 | 0.3263 | 0.3096 | 0.2319 |
| AUROC (within vs between pairs) | 0.68 | 0.8096 | 0.8045 | 0.7876 |

## How to read these metrics (plain language)

### Participation ratio — not the same as “% of dimensions used”

The table lists **participation ratio** next to embedding size (e.g. 16 / 2048). That ratio is an **effective number of PCA directions** that carry variance: ~1 means almost all cells vary along one dominant axis (collapsed); ~16 means variance is spread across roughly sixteen directions.

If you divide by embedding dimension (16/2048 ≈ 0.8% vs OpenPhenom 1.8/384 ≈ 0.5%), the gap looks tiny. That is misleading. Compare instead:

| | OpenPhenom | MS4D | MS3D | MS2D |
|--|------------|------|------|------|
| **Absolute participation ratio** | **~2** | ~16 | ~15 | **~20** |
| **Top-1 PC variance share** | **~74%** | ~13% | ~15% | ~14% |
| **Mean pairwise cosine similarity** | **~0.94** | ~0.23 | ~0.22 | ~0.22 |

OpenPhenom is collapsed in **absolute** effective rank and in **how similar all pairs are** (0.94). MitoSpace models use ~9–11× more effective directions and much lower average similarity. **2D has the highest participation ratio among MitoSpace models** (19.5), not the lowest.

### Two questions: “separate drugs” vs “match extreme MitoTNT”

The probe asks separate things:

1. **Drug structure in raw cosine space** — Are cells treated with the same drug more similar to each other than to other drugs? (within−between gap, AUROC, raw mAP.)
2. **Link to MitoTNT severity (E11)** — For each drug, is its embedding separation large when its MitoTNT profile is extreme compared with all cells?

**2D captures most of (1):** gaps and AUROC are close to 3D/4D (e.g. raw mAP ~0.27 vs ~0.30). A single 2D mitochondrial image already supports strong drug clustering.

**4D is clearer on (2):** Spearman ρ between MitoTNT “unusualness” and per-drug embedding gap is **0.5392** (p≈0.005409) for 4D vs **0.3023** (not significant) for 2D. So 4D separation tracks severe mitochondrial remodeling more reliably; 2D can separate drugs without that separation scaling as cleanly with MitoTNT outliers.

### What is “PCA-CenterScale” and “replicate mAP”?

**Replicate retrieval** is a practical test: pretend you “lost” the drug label of one cell, find its nearest neighbors in embedding space by cosine similarity, and ask whether those neighbors are mostly the **same drug** (technical replicates / same treatment).

- **mAP (mean average precision)** summarizes how highly same-drug cells rank among neighbors.   - **0** ≈ useless (random guessing).
  - **~0.05** ≈ random floor on this benchmark.
  - **~0.30** ≈ strong retrieval for 25 drugs.
  - **1.0** ≈ perfect.

**Raw mAP** uses embeddings exactly as the model outputs them.

**PCA-CenterScale** is a post-processing step (same idea as Recursion’s OpenPhenom recipe, adapted here):

1. Take **control (untreated) cells** only.
2. Fit **PCA** on their embeddings — captures the main shared “baseline mitochondrial look”.
3. **Center and scale** in that PCA space (StandardScaler fit on controls).
4. Apply that transform to **all** cells (including drugs).

Intuition: many embeddings share a big common direction (imaging conditions, cell health baseline). PCA-CenterScale **subtracts that shared bulk** so neighbors are chosen more from **drug-specific residual** structure than from “everyone looks a bit like control”.

On this cohort (25 drugs × 20 cells per drug in E4):

| | MS4D (4D) | MS2D (2D) |
|--|-----------|-----------|
| Raw replicate mAP | 0.2957 | 0.2656 |
| mAP after PCA-CenterScale | **0.3263** | 0.2319 |
| Random ranking floor | ~0.04893 | ~0.04893 |

4D **gains** more from correction (0.30 → 0.33) than 2D (0.27 → 0.23). So when you explicitly remove the control-dominated subspace, **4D’s neighbors are more drug-faithful** — better for “find my replicate / same treatment” workflows. 2D already separates drugs reasonably in raw space but benefits less (here, mAP slightly drops after correction, suggesting more drug signal was already in the raw 512-d directions or the control subspace differs).

**When to care:** use **raw mAP / gap** for “does the model cluster drugs at all?”; use **PCA-CenterScale mAP** for “after removing baseline morphology, can I retrieve same-treatment cells?” — especially relevant when comparing to OpenPhenom (raw mAP ~0.02, corrected ~0.03).

## Short answer

**All three MitoSpace dimensionalities learn strong, wide drug-discriminative embeddings on this mitochondrial screen; OpenPhenom on RxRx3-core is comparatively collapsed and weakly discriminative in the reference probe.** 3D matches or slightly exceeds 4D on raw within−between gap; 4D leads on PCA-CenterScale mAP and MitoTNT–gap correlation. 2D is competitive at 512-d but weakest on replicate retrieval after post-processing.

## MitoSpace 4D vs 3D vs 2D (same cells, same drugs)

| Metric | MS4D (4D) | MS3D (3D) | MS2D (2D) |
|--------|-----------|-----------|-----------|
| Embedding dim | 2048 | 2048 | 512 |
| Participation ratio | 16 | 14.69 | 19.52 |
| Raw within−between gap (E4/E10) | 0.2653 | 0.2672 | 0.2114 |
| Raw replicate mAP (E4) | 0.2957 | 0.2926 | 0.2656 |
| mAP after PCA-CenterScale | 0.3263 | 0.3096 | 0.2319 |
| AUROC within vs between (E10) | 0.8096 | 0.8045 | 0.7876 |
| MitoTNT unusualness vs gap ρ (E11) | 0.5392 | 0.2523 | 0.3023 |

**Ranking (this panel):** discriminability **3D ≈ 4D > 2D**; post-processed replicate retrieval **4D > 3D > 2D**; link to extreme MitoTNT phenotypes **4D only (significant)**.

## Interpretation

### Geometry (E1)

- **MitoSpace4D (4D):** participation ratio **16 / 2048**, mean pairwise cos-sim **0.2344** (OpenPhenom ≈ 0.94 on 384-d — highly collapsed).
- **MitoSpace3D (3D):** participation ratio **14.69 / 2048**, mean pairwise cos-sim **0.2204** (OpenPhenom ≈ 0.94 on 384-d — highly collapsed).
- **MitoSpace2D (2D):** participation ratio **19.52 / 512**, mean pairwise cos-sim **0.2161** (OpenPhenom ≈ 0.94 on 384-d — highly collapsed).

### Training signal vs architectural prior (E2, E7)

- **OpenPhenom:** Pairwise rankings largely shared with a **vanilla random ViT** (Spearman ρ ≈ 0.36); same-architecture random twin is *not* comparable (ρ ≈ −0.03).
- **MitoSpace4D (4D):** Spearman ρ vs random-init = **0.005944**; mAP trained **0.3008** vs random **0.04557** (floor **0.0437**).

### Drug discriminability (E4, E10)

- **OpenPhenom:** Small absolute gaps; subtle pharmacology is hard without PCA-CenterScale.
- **MitoSpace4D (4D):** raw gap **0.2755**, mAP **0.2957**; after PCA-CenterScale mAP **0.3263**.
- **MitoSpace3D (3D):** raw gap **0.2869**, mAP **0.2926**; after PCA-CenterScale mAP **0.3096**.
- **MitoSpace2D (2D):** raw gap **0.2386**, mAP **0.2656**; after PCA-CenterScale mAP **0.2319**.

### Morphology vs embedding gap (E11)

- **OpenPhenom:** pixel unusualness vs embedding gap Spearman ρ ≈ **0.04** (weak).
- **MitoSpace4D (4D):** MitoTNT unusualness vs gap ρ = **0.5392** (p=0.005409).
- **MitoSpace3D (3D):** MitoTNT unusualness vs gap ρ = **0.2523** (p=0.2237).
- **MitoSpace2D (2D):** MitoTNT unusualness vs gap ρ = **0.3023** (p=0.1419).

## Per-model reports

| Model | Report |
|-------|--------|
| MitoSpace4D (4D) | [results/ms4d/findings.md](results/ms4d/findings.md) |
| MitoSpace3D (3D) | [results/ms3d/findings.md](results/ms3d/findings.md) |
| MitoSpace2D (2D) | [results/ms2d/findings.md](results/ms2d/findings.md) |

## Practical takeaways

1. **MitoSpace (4D/3D/2D)** on this mitochondrial drug panel shows **much wider** embedding geometry and **stronger drug separation** than OpenPhenom on RxRx3-core in the reference probe.
2. **Random-init MitoSpace4D** does *not* preserve ranking structure (unlike OpenPhenom’s ViT prior) — training is doing substantial work.
3. **PCA-CenterScale** (control-calibrated) remains useful for replicate retrieval on MitoSpace, analogous to OpenPhenom’s post-processing recipe.
4. **Temporal (4D) vs volumetric (3D) vs projection (2D):** adding time helps post-processing and MitoTNT alignment; 3D alone is nearly as discriminative in raw cosine space; 2D is the most compressed baseline.
