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
from validation_zslices import pick_folders, validate_saved_model
from auto_encoder_kinetics.ae_util import AEUtil

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
        # full_image_data_np = np.load(filepath)
        decode = True # for encoded data reading

        if decode:
            ae_util = AEUtil(ckpt_path="auto_encoder_kinetics/mitospace_resnet_autoencoder_20251018.ckpt")
            full_image_data_np = ae_util.load(filepath)  # decoded_image shape: (C,T,Z,Y,X)
        else:
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


def incremental_train_z_predictor(cfg):
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
    cfg.batch_size_files = 100  # same as before
    total_batches = (total_files + cfg.batch_size_files - 1) // cfg.batch_size_files

    print(f"Total files: {total_files} -> {total_batches} batches of {cfg.batch_size_files}")

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

    

    for batch_start in range(start_file_idx, total_files,  cfg.batch_size_files):
        batch_end = min(batch_start + cfg.batch_size_files, total_files)
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
                loss = 0.5 * loss_ce + 0.5 * loss_vol 
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


    # === Quick pick labels/folders: manually specify folder dates or partials ===
    import argparse

    parser = argparse.ArgumentParser(description="Train Z-predictor with selected folders.")
    parser.add_argument(
        "--pick_folders",
        nargs="+",
        default=["20240729-1", "20240823-1"],
        # default=
        help="List of folder names (space separated) to include for training. Example: --pick_folders 20240729-1 20240823-1"
    )
    parser.add_argument("--save_path", default="checkpoint_new_loss_mdivi_control", help="Path to save the checkpoint")
    parser.add_argument("--master_base_path", default="/work/nvme/begq/MitoSpace4D/2024_encoded_data/", help="Path to the master base path")
    parser.add_argument("--batch_size", default=20, help="Batch size for training")
    parser.add_argument("--num_epochs", default=30, help="Number of epochs for training")
    parser.add_argument("--learning_rate", default=1e-4, help="Learning rate for training")
    parser.add_argument("--test_size_z", default=0.3, help="Test size for Z-slices")
    parser.add_argument("--batch_size_files", default=100, help="Batch size for files")
    parser.add_argument("--train_split", default=300, help="Split point for training and validation")
    parser.add_argument("--device", default="cuda:0", help="Device to use for training")
    parser.add_argument("--image_processor_name", default="MCG-NJU/videomae-base", help="Image processor name")
    parser.add_argument("--all_drugs", default=False, help="Use all drugs")
    args = parser.parse_args()
    
    train_split = args.train_split
    cfg = Config()
    cfg.master_base_path = args.master_base_path
    cfg.base_paths = [f"{cfg.master_base_path}{i}" for i in os.listdir(cfg.master_base_path)]
    cfg.image_filepaths = []
    cfg.val_filepaths = []
    all_drugs = [i for i in os.listdir(cfg.master_base_path) if os.path.isdir(os.path.join(cfg.master_base_path, i))]
    
    pick_folders = all_drugs[10:]
    print(pick_folders)
    # if not args.all_drugs:
    #     pick_folders = args.pick_folders
    # else:
    #     pick_folders = all_drugs
    
    cfg.save_path = args.save_path
    cfg.batch_size = int(args.batch_size)
    cfg.num_epochs = int(args.num_epochs)
    cfg.learning_rate =float(args.learning_rate)
    cfg.test_size_z = float(args.test_size_z)
    cfg.batch_size_files = int(args.batch_size_files)
    
    cfg.device = args.device
    cfg.image_processor_name = args.image_processor_name
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path, exist_ok=True)
    print(cfg.base_paths[1])
    # If you want to match multiple folders per pick_label, just add them above

    # Filter base_paths for the ones that match pick_folders
    filtered_base_paths = [bp for bp in cfg.base_paths if any(bp.endswith(pick) for pick in pick_folders)]

    print("Picked folders:", filtered_base_paths)
    for base_path in filtered_base_paths:
        print("Checking:", base_path)
        print("Loading data from:", base_path)
        files = sorted(os.listdir(base_path))
        for i in files[0:train_split]:
            cfg.image_filepaths.append(f"{base_path}/{i}")
        for i in files[train_split:]:
            cfg.val_filepaths.append(f"{base_path}/{i}")

    print(len(cfg.image_filepaths))

    incremental_train_z_predictor(cfg)

    # model_path = f"{cfg.save_path}/z_predictor_incremental.pth"
    # aggregate_method = "max"
    # device = "cuda:0"
    # visualize_umap = False
    # save_embeddings = True
    # concatenate_embeddings = True
    # exp_name = cfg.save_path.split("_")[1]
    # save_umap_path = f"umap_validation_{exp_name}"
    # # cfg.val_filepaths_2 = cfg.val_filepaths[i:i+100]
    # print(len(cfg.val_filepaths), cfg.val_filepaths[0], cfg.val_filepaths[-1], cfg.val_filepaths[100], cfg.val_filepaths[200])
    # #  = cfg.batch_size_files
    # for folder in pick_folders:
    #     idxs = [i for i, fpath in enumerate(cfg.val_filepaths) if folder in fpath]
    #     if not idxs:
    #         print(f"⚠️ No filepaths found for folder: {folder}")
    #         continue
    #     for start in range(0, len(idxs), cfg.batch_size_files):
    #         selected_idxs = idxs[start:start+cfg.batch_size_files]
    #         if not selected_idxs:
    #             continue
    #         cfg.val_filepaths_2 = [cfg.val_filepaths[i] for i in selected_idxs]
    #         print(f"Folder: {folder} | Batch size: {len(cfg.val_filepaths_2)}")
    #         validate_saved_model(model_path, 
    #                             device =device, 
    #                             visualize_umap = visualize_umap,
    #                             save_embeddings = save_embeddings,
    #                             save_umap_path = save_umap_path,
    #                             concatenate_embeddings = concatenate_embeddings,
    #                             aggregate_method = aggregate_method,
    #                             cfg = cfg)
