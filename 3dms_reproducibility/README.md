# 3dms_reproducibility

MitoSpace-**3D** embeddings and cross-day k-NN evaluation for the reproducibility dataset.

## How this was produced

| | |
|---|---|
| Checkpoint | `manuscript_v2/checkpoints/ms3d_2024v3_epoch=225-step=24182_best.ckpt` |
| Model definition | `archive/models_simple_3d_2024v3.py::Lightweight3DResNet` |
| Data | `/mnt/aquila/ssd_processing/Others/MitoSpace4D/reproducibility/processed_data` |
| Samples | 12,481 movies, `(T=20, Z=60, Y=256, X=256)` uint16, single (morphology) channel |
| Scripts | `generate_3dms_reproducibility_embeddings.py`, `evaluate_3dms_reproducibility_crossday.py` |

```bash
python generate_3dms_reproducibility_embeddings.py --embeddings_dir 3dms_reproducibility
python evaluate_3dms_reproducibility_crossday.py   --embeddings_dir 3dms_reproducibility
```

**Last frame only.** The 3D model embeds one volume per movie. In training it received
autoencoder latents, decoded the *last* latent frame, and embedded that; here the data is
raw (un-encoded), so the frozen decoder is bypassed and the raw last frame `movie[-1]` is
fed straight into the ResNet trunk. This is the same treatment `evaluate.py` applies to
un-encoded data for the 4D model (`decoder_checkpoint_path=None`). The checkpoint's 136
trunk tensors are loaded; its 128 decoder tensors are unused.

**Normalization.** Each movie is min-max scaled over *all 20 frames* before the last frame
is taken, matching the whole-movie normalization the autoencoder used when the training
latents were written (`autoencoder/encode_data.py`), then rescaled to `[-1, 1]` inside the
model.

## Files

| File | Contents |
|---|---|
| `embeddings.npy` | `(12481, 2048)` float32, L2-normalized `fc` embeddings |
| `embeddings_resnet.npy` | `(12481, 512)` float32, L2-normalized ResNet-trunk features |
| `labels.npy` | `(12481,)` int32 drug label (0 control, 9 oligomycin, 17 nocodazole) |
| `image_paths.csv` | source `.npy` path, one per row, same order as the arrays |
| `metadata.csv` | path, label, drug name, acquisition date, replicate — the `date` column defines the folds |
| `label_names.npy`, `args.yaml`, `config.yaml` | run provenance |
| `crossval_perday/` | k-NN results and confusion matrices |

## Cross-day evaluation

The dataset is **two dye batches of three days each**, each batch containing the same
three conditions, so 3-fold leave-one-day-out is run once per batch:

- `deepred` — 20260330, 20260331, 20260401 (MitoTracker DeepRed)
- `mitobright` — 20260407, 20260408, 20260409 (MitoBright)

Protocol matches `utils/evaluate_reproducibility_crossday.py`: classes balanced to the
smallest class within a batch (seed 1123), one day held out per fold, each held-out cell
classified by a **similarity-weighted vote over its k=100 nearest cosine neighbours** among
the other two days. Re-running this evaluator on the existing 4D embeddings reproduces the
published fold accuracies exactly, so the two models' numbers are directly comparable.

### Top-1 accuracy (k=100)

| Batch | Held-out day | MitoSpace-3D | MitoSpace-4D (published) |
|---|---|---|---|
| deepred | 20260330 | 78.30% | 84.61% |
| deepred | 20260331 | 77.39% | 83.42% |
| deepred | 20260401 | 86.12% | 88.66% |
| **deepred** | **mean ± sd** | **80.60 ± 4.80%** | **85.56 ± 2.74%** |
| mitobright | 20260407 | 90.44% | 92.69% |
| mitobright | 20260408 | 86.93% | 82.17% |
| mitobright | 20260409 | 74.09% | 81.68% |
| **mitobright** | **mean ± sd** | **83.82 ± 8.61%** | **85.51 ± 6.22%** |

Top-3 accuracy is ≥99.1% in every fold; with only three classes top-3 is just a sanity
check.

### Per-drug top-1 (micro-averaged over the three folds)

| Batch | control | oligomycin | nocodazole |
|---|---|---|---|
| deepred | 90.03% | 59.62% | 93.97% |
| mitobright | 70.24% | 81.64% | 97.42% |

Oligomycin is the hard class for the 3D model on the DeepRed batch — 24% of oligomycin
cells are called control. That is consistent with its mechanism: oligomycin inhibits ATP
synthase and its earliest effects are functional rather than morphological, so a
single-frame model has less to work with than the 4D model (which reaches 69.9% on the
same class).

### Result files in `crossval_perday/`

| File | Contents |
|---|---|
| `knn_accuracy_per_fold.csv` | accuracy per batch × held-out day × top-N |
| `knn_accuracy_mean_across_folds.csv` | mean ± sd across the three folds |
| `knn_accuracy_per_drug_per_fold.csv` | per-drug accuracy for every fold |
| `knn_accuracy_per_drug_mean_across_folds.csv` | per-drug macro mean and micro (pooled) accuracy |
| `<batch>/cm_top1-test_date=<day>.png` | row-normalized top-1 confusion matrix for one fold |
| `<batch>/cm_top1_counts-test_date=<day>.png` | same fold, raw counts |
| `<batch>/cm_top1-mean_across_folds_normalized.png` | all three folds pooled, row-normalized |
| `<batch>/cm_top1-sum_across_folds_counts.png` | all three folds pooled, raw counts |
| `<batch>/cm_top1-sum_across_folds.npy` | the pooled count matrix |
| `<batch>/cm_top1-class_order.txt` | row/column order of the saved matrices |
