"""
train_z_custom_loss.py
----------------------------------------
This script extends the existing Z-predictor training to use a custom
loss function that evaluates how many Z slices are correctly predicted
per volume (% correct per image stack).

It imports:
  - model & dataset loading from your main script
  - adds volume IDs to the dataset
  - defines and uses ZVolumeAccuracyLoss
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForVideoClassification
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
# Import your utilities from the main Z-predictor training script
from videomae_zslicer import (
    # Config,
    ZSliceDataset,
    normalize_and_mask,
    ZVolumeAccuracyLoss
)
from train_zslices_v1 import Config

# This function loads a subset of the data and stacks it into a tensor
def load_data_subset(cfg, start_idx, end_idx):
    """
    Loads and stacks .npy video files into a tensor subset.
    Returns:
        all_data_combined_tensor (torch.Tensor)
        all_original_z_indices (list[int])
        global_volume_counter (list[int])
    """
    batch_files = cfg.image_filepaths[start_idx:end_idx]
    z_video_clips, all_original_z_indices, global_volume_counter = [], [], []
    global_slice_counter = 0
    global_volume_ID = 0

    for filepath in tqdm(batch_files, desc="Loading subset"):
        full_image_data_np = np.load(filepath)
        data_tensor = torch.from_numpy(full_image_data_np)  # (T, C, Z, H, W)
        if data_tensor.shape[1] != 2:
            C, time_points, Z, H, W = data_tensor.shape
            # print("data shape", data_tensor.shape)
            data_tensor = data_tensor.permute(1, 0, 2, 3, 4)
        else:
            time_points, C, Z, H, W = data_tensor.shape

        for z_idx in range(Z):
            z_video_clips.append(data_tensor[3:19, :, z_idx, :, :])
            all_original_z_indices.append(z_idx)
            global_volume_counter.append(global_volume_ID)
            global_slice_counter += 1
        global_volume_ID += 1

    data_tensor = torch.stack(z_video_clips, dim=0).to(torch.float32)
    return data_tensor, all_original_z_indices, global_volume_counter


def incremental_train_z_predictor(cfg, batch_group_size=2):
    """
    Train the Z-predictor in streaming batches of files.
    Each `batch_group_size` corresponds to number of batches of 100 files each.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    # Initialize model & processor once
    image_processor = AutoImageProcessor.from_pretrained(cfg.image_processor_name, do_rescale=False)
    model = AutoModelForVideoClassification.from_pretrained(cfg.model_name, num_labels=3)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ce_loss_fn = nn.CrossEntropyLoss()
    vol_loss_fn = ZVolumeAccuracyLoss()

    start_file_idx = 0
    total_files = len(cfg.image_filepaths)
    batch_size_files = 100  # same as before
    total_batches = (total_files + batch_size_files - 1) // batch_size_files

    print(f"Total files: {total_files} -> {total_batches} batches of {batch_size_files}")

    # Resume if checkpoint exists
    checkpoint_path = Path(cfg.save_path) / "z_predictor_incremental.pth"

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path, exist_ok=True)

    if checkpoint_path.exists():
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_file_idx = checkpoint["next_file_idx"]
        # start_file_idx = 0 #only for valinomycin and nigericin c
    

    for batch_start in range(start_file_idx, total_files, batch_group_size * batch_size_files):
        batch_end = min(batch_start + batch_group_size * batch_size_files, total_files)
        print(f"\n📦 Loading files {batch_start}–{batch_end}")

        # === Load and preprocess this subset ===
        data_tensor, orig_z_indices, vol_ids = load_data_subset(cfg, batch_start, batch_end)

        # === Split & create Datasets ===
        train_idx, test_idx = train_test_split(range(len(orig_z_indices)), test_size=cfg.test_size_z)
        train_set = ZSliceDataset(data_tensor[train_idx], [orig_z_indices[i] for i in train_idx],
                                  [vol_ids[i] for i in train_idx], image_processor)
        val_set = ZSliceDataset(data_tensor[test_idx], [orig_z_indices[i] for i in test_idx],
                                [vol_ids[i] for i in test_idx], image_processor)
        
        
        num_labels = 3
        max_z = 60
        divider = max_z//num_labels
        all_unique_original_z_indices = sorted(list(set(orig_z_indices)))
        z_to_label_mapping = {z: i//divider for i, z in enumerate(all_unique_original_z_indices)}
        label_to_z_mapping = {i: i*divider + divider*0.5 for i in set(z_to_label_mapping.values())}
        train_set.set_label_mapping(z_to_label_mapping)
        val_set.set_label_mapping(z_to_label_mapping)

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        

        # === Train for a few epochs on this chunk ===
        print(f"🚀 Training on files {batch_start}–{batch_end}")
        for epoch in range(cfg.num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                volume_ids = batch["volume_id"].to(device)

                optimizer.zero_grad()
                logits = model(pixel_values=pixel_values).logits
                loss_ce = ce_loss_fn(logits, labels)
                loss_vol = vol_loss_fn(logits, labels, volume_ids)
                # Assign higher weight to volume accuracy loss.
                # Additionally, increase loss weight for class 2 or 3 (z), assuming 0-based classes
                # Note: make sure labels are 0,1,2 or adjust as necessary.
                volume_weight = 0.7
                ce_weight = 0.3

                # Default: no label-specific boost
                label_weight = torch.ones_like(labels, dtype=torch.float32).to(labels.device)
                # Boost weight for class 2 or 3 (indices 2 and 3); adjust if only 0,1,2 exist
                for_high_z = (labels == 2) | (labels == 1)
                label_weight[for_high_z] = 2.0

                weighted_loss_ce = (loss_ce * label_weight).mean()
                weighted_loss_vol = (loss_vol * label_weight).mean()
                
                loss = ce_weight * weighted_loss_ce + volume_weight * weighted_loss_vol
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"✅ Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f}")

        # === Save incremental checkpoint ===
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "next_file_idx": batch_end
        }, checkpoint_path)

        print(f"💾 Saved incremental checkpoint at file index {batch_end}")

        # === Free memory ===
        del data_tensor, train_set, val_set, train_loader, val_loader
        torch.cuda.empty_cache()

    print("🎉 Training complete across all file batches!")


if __name__ == "__main__":
    cfg = Config()
    cfg.master_base_path = "/run/user/1004/gvfs/afp-volume:host=JSLab-Server1.local,user=JSLab_FileShare,volume=SSD_Processing/Others/MitoSpace4D/2024_summer_new/"
    cfg.base_paths = [f"{cfg.master_base_path}{i}" for i in os.listdir(cfg.master_base_path)]
    cfg.image_filepaths = []
    cfg.val_filepaths = []
    
    cfg.save_path = "checkpoint_lowlr_epoch_mdvivi1_control"
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path, exist_ok=True)
    print(cfg.base_paths[1])

    # === Quick pick labels/folders: manually specify folder dates or partials ===
    pick_folders = [
        "20240729-1",#control
        # "20240805-1",#H2O2 
        # "20240802-1",#tbhp
        # "20240814-1",#valinomycin
        # "20240911-1",#nigericin
        # "20240826-1",#nocodazole
        # "20240830-1",#colchicine
        # "20240816-1",#mitomycinC
        # "20240905-1",#cisplatin
        "20240823-1",#mdivi1
        # Add more folder names or date-based identifiers as needed
    ]
    # If you want to match multiple folders per pick_label, just add them above

    # Filter base_paths for the ones that match pick_folders
    filtered_base_paths = [bp for bp in cfg.base_paths if any(bp.endswith(pick) for pick in pick_folders)]

    print("Picked folders:", filtered_base_paths)
    for base_path in filtered_base_paths:
        print("Checking:", base_path)
        print("Loading data from:", base_path)
        files = sorted(os.listdir(base_path))
        for i in files[0:500]:
            cfg.image_filepaths.append(f"{base_path}/{i}")
        for i in files[500:]:
            cfg.val_filepaths.append(f"{base_path}/{i}")

    print(len(cfg.image_filepaths))

    incremental_train_z_predictor(cfg)
