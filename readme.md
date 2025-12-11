# How To Run This Code

This project implements incremental training and evaluation for video-based slice-wise (z-dimension) classification using a VideoMAE backbone.

## File Descriptions

1. **videomae_zslices.py**
    - Contains configuration logic (`Config` class) and the definition of the ZSlice `Dataset`.
    - Responsible for setting up the dataloader and any transforms or data organization specific to z-slices.

2. **train_incremental function (`train_validate_z_slices.py`)**
    - Handles training using incremental batches, saving checkpoints after each batch of files.
    - Designed for large datasets that can't fit in memory all at once.

3. **Train and Validate (`train_validate_z_slices.py` main section)**
    - After training is complete, runs validation on the held-out data.
    - Provides command-line arguments to specify which folders to use, paths, epochs, batch sizes, etc.
    - Example command:
      ```
      python train_validate_z_slices.py --pick_folders 20240729-1 20240823-1 --save_path your_checkpoint_dir --num_epochs 30 --batch_size 20
      ```
    - See `python train_validate_z_slices.py --help` for all available arguments.

4. **validate_zslices.py**
    - Contains evaluation code including various cluster and k-NN metrics in embedding space.
    - Loads the trained model and data in CPU-memory-efficient batches.
    - Produces per-volume accuracy and saves intermediate/aggregate embeddings (supports different aggregation schemes, e.g. max, mean, etc.), as well as visualizations in UMAP space if enabled in config.

5. **How to Generate Experimental Space (zv2)**
    - The code allows for flexible aggregation and visualization of embeddings along the z-dimension, supporting experiments for hypothesis-driven space generation methods (space zv2).
    - For custom experiments, set aggregation and visualization options in the relevant sections of `validate_zslices.py` and/or `train_validate_z_slices.py`.

---

## Minimal Training and Evaluation Example

```bash
# Step 1: Train incrementally over specific folders
python train_validate_z_slices.py \
  --pick_folders 20240729-1 20240823-1 \
  --save_path checkpoint_new_loss_mdivi_control \
  --num_epochs 30 \
  --batch_size 20

# Step 2: Validate and extract embeddings (from validate_zslices.py)
python validate_zslices.py --save_path checkpoint_new_loss_mdivi_control
```

Outputs will include incremental checkpoints, per-volume accuracy metrics, and saved confusion matrix/UMAP visualizations.

---

## Customizing/Extending

- See `videomae_zslices.py` to change dataset configuration or augmentations.
- Check arguments in `train_validate_z_slices.py` to control data splits and model hyperparameters.
- Use aggregation methods in `validate_zslices.py` to experiment with how per-slice embeddings are combined.

---

