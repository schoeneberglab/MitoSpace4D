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
    Config,
    ZSliceDataset,
    normalize_and_mask,
    ZVolumeAccuracyLoss
)

from validation_zslices import validate_saved_model
# ------------------------------------------------------------
# 2️⃣ Training Loop using the Custom Loss
# ------------------------------------------------------------

def train_z_predictor_with_custom_loss():
    """
    Loads model, data, and trains with both CrossEntropy + VolumeAccuracy loss
    """

    all_z_video_clips = [] # Will store (T, C_orig, H, W) for each z_slice from all files
    all_original_z_indices = [] # Will store the original z_index for each video clip
    global_slice_counter = [] # Stores the actual global slice number 
    # Store global Z index across all files to handle unique Z values
    global_z_idx_counter = 0 
    global_volume_counter = []

    for volume_id , filepath in enumerate(Config.image_filepaths):
        try:
            full_image_data_np = np.load(filepath)
            # Ensure it's a PyTorch tensor
            current_data_tensor = torch.from_numpy(full_image_data_np) 
            print("data shape", current_data_tensor.shape)
            
            # Expected (T, C, Z, H, W)
            time_points, num_channels_orig, z_slices_file, H, W = current_data_tensor.shape
            print(f"Loaded {filepath}: (T={time_points}, C={num_channels_orig}, Z={z_slices_file}, H={H}, W={W})")

            # Extract each z-slice as a (T, C, H, W) video clip
            for z_idx_in_file in range(z_slices_file):
                video_clip_for_z = current_data_tensor[3:19, :, z_idx_in_file, :, :] # (T, C, H, W)
                all_z_video_clips.append(video_clip_for_z)
                # We need a unique Z identifier across all files if Z values can overlap
                # Or, if Z-indices are truly unique per biological sample (e.g., cell)
                # For simplicity here, we create a global unique Z_index for each physical Z-slice across all files
                all_original_z_indices.append(z_idx_in_file) 
                global_slice_counter.append(global_z_idx_counter)
                global_volume_counter.append(volume_id)
                global_z_idx_counter += 1

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}. Skipping.")
        except Exception as e:
            print(f"Error loading {filepath}: {e}. Skipping.")

    if not all_z_video_clips:
        print("No image data loaded. Exiting.")
        return

    # Combine all video clips into a single tensor (N_total_samples, T, C_orig, H, W)
    all_data_combined_tensor = torch.stack(all_z_video_clips, dim=0) # Shape: (Total_Z_slices_across_files, T, C_orig, H, W)
    total_z_samples = all_data_combined_tensor.shape[0]
    
    print(f"Total {total_z_samples} Z-slice video samples loaded from {len(Config.image_filepaths)} files.")

    # Split the *indices* of these samples into train and test sets
    sample_indices = global_slice_counter
    train_sample_indices, test_sample_indices = train_test_split(
        sample_indices, test_size=Config.test_size_z, random_state=Config.random_seed
    )
    
    # Get the actual original Z-indices for the train/test splits
    train_original_z_indices = [all_original_z_indices[i] for i in train_sample_indices]
    test_original_z_indices = [all_original_z_indices[i] for i in test_sample_indices]
    # Get the actual volume ids split by training and testing
    test_volume_ids = [global_volume_counter[i] for i in test_sample_indices]
    train_volume_ids = [global_volume_counter[i] for i in train_sample_indices]
    max_z = z_slices_file
    print(max_z)
    # Create a consistent mapping from *all unique original Z-indices* to model class labels (0 to N-1)
    # This is critical for consistent labeling across train and test datasets.
    # num_labels = len(all_unique_original_z_indices)
    num_labels = 3
    divider = max_z//num_labels
    all_unique_original_z_indices = sorted(list(set(train_original_z_indices + test_original_z_indices)))
    z_to_label_mapping = {z: i//divider for i, z in enumerate(all_unique_original_z_indices)}
    label_to_z_mapping = {i: i*divider + divider*0.5 for i in set(z_to_label_mapping.values())}
    # label_to_z_mapping = {i: z for i, z in enumerate(all_unique_original_z_indices)} # For visualization
    # print("Z to label-mapping: ",z_to_label_mapping)
    # print("Label to Z-mapping: ",label_to_z_mapping)

    print(f"Number of unique Z-slice classes (total labels): {num_labels}")
    print(f"Number of Z-slice samples for training: {len(train_sample_indices)}")
    print(f"Number of Z-slice samples for testing: {len(test_sample_indices)}")

    # Initialize image processor and model
    image_processor = AutoImageProcessor.from_pretrained(Config.image_processor_name, do_rescale=False)
    
    # Determine the number of input channels for the model
    # If create_third_channel is True, it will be 3, otherwise 2 (assuming original input has 2)
    model_in_channels = 2 + (1 if Config.create_third_channel else 0)
    
    model = AutoModelForVideoClassification.from_pretrained(Config.model_name, num_labels=num_labels)
    # print("Shape of all data: ", all_data_combined_tensor.shape, train_sample_indices)
    all_data_combined_tensor = all_data_combined_tensor.to(torch.int32)

    # Create datasets
    train_dataset = ZSliceDataset(
        all_data_combined_tensor[train_sample_indices], # Only pass the actual data for this split
        train_original_z_indices,
        train_volume_ids, 
        image_processor,
        create_third_channel=Config.create_third_channel,
    )
    test_dataset = ZSliceDataset(
        all_data_combined_tensor[test_sample_indices],
        test_original_z_indices,
        test_volume_ids,
        image_processor,
        create_third_channel=Config.create_third_channel
    )
    complete_set = ZSliceDataset(
        all_data_combined_tensor,
        all_original_z_indices,
        global_volume_counter,
        image_processor,
        create_third_channel=Config.create_third_channel
    )
    # Set the global label mapping for both datasets
    train_dataset.set_label_mapping(z_to_label_mapping)
    test_dataset.set_label_mapping(z_to_label_mapping)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    complete_loader = DataLoader(complete_set, batch_size = Config.batch_size , shuffle = False, num_workers=4)
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")


    # ✅ Define losses
    ce_loss_fn = nn.CrossEntropyLoss()
    vol_loss_fn = ZVolumeAccuracyLoss()


    # ✅ Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    history = {"total": [], "ce": [], "vol": [], "val_slice": [], "val_vol": []}

    if not os.path.exists(Config.save_path):
            os.makedirs(Config.save_path)

    for epoch in range(Config.num_epochs):
        model.train()
        epoch_total, epoch_ce, epoch_vol = 0.0, 0.0, 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            volume_ids = batch["volume_id"].to(device)
            # print("No of samples in the batch", len(labels))
            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits

            loss_ce = ce_loss_fn(logits, labels)
            loss_vol = vol_loss_fn(logits, labels, volume_ids)
            loss = 0.7 * loss_ce + 0.3 * loss_vol

            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_ce += loss_ce.item()
            epoch_vol += loss_vol.item()

        avg_total = epoch_total / len(train_loader)
        avg_ce = epoch_ce / len(train_loader)
        avg_vol = epoch_vol / len(train_loader)
        val_slice, val_vol = evaluate_z_predictor(model, val_loader, device)

        print(
            f"Epoch {epoch+1}: Loss {avg_total:.4f} "
            f"(CE {avg_ce:.4f}, Vol {avg_vol:.4f}) "
            f"| Val Slice {val_slice:.3f} Val Vol {val_vol:.3f}"
        )

        history["total"].append(avg_total)
        history["ce"].append(avg_ce)
        history["vol"].append(avg_vol)
        history["val_slice"].append(val_slice)
        history["val_vol"].append(val_vol)
        
        

        if min(history["vol"]) == avg_vol and ((epoch+1)%5)==0 :
            torch.save(model.state_dict(), f"{Config.save_path}/z_pred_{avg_vol:0.2f}.pth")
            print("✅ Model saved with custom Z-loss!")


    # Optional validation step
    evaluate_z_predictor(model, val_loader, device)

    torch.save(model.state_dict(), f"{Config.save_path}/z_predictor_custom_loss.pth")
    print("✅ Model saved with custom Z-loss!")

     # --------------------------------------------------------
    # 4️⃣ Plot Loss History
    # --------------------------------------------------------
    out_dir = Path(Config.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history["total"], label="Total Loss")
    plt.plot(history["ce"], label="CE Loss")
    plt.plot(history["vol"], label="Vol Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png")
    plt.close()

    # --------------------------------------------------------
    # 5️⃣ Evaluate on Test Set
    # --------------------------------------------------------
    test_slice_acc, test_vol_acc = evaluate_z_predictor(model, test_loader, device)
    print(f"\n✅ Final Test Results – Slice Acc: {test_slice_acc:.3f} | Volume Acc: {test_vol_acc:.3f}")
# ------------------------------------------------------------
# 3️⃣ Validation with Volume Accuracy
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_z_predictor(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    vol_accs = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        volume_ids = batch["volume_id"].to(device)

        outputs = model(pixel_values=pixel_values)
        preds = outputs.logits.argmax(dim=1)
        correct_mask = (preds == labels).float()

        for vid in torch.unique(volume_ids):
            mask = (volume_ids == vid)
            vol_accs.append(correct_mask[mask].mean())

        correct += correct_mask.sum().item()
        total += labels.numel()

    slice_acc = correct / total
    vol_acc = torch.stack(vol_accs).mean().item() if vol_accs else 0.0
    return slice_acc, vol_acc


# ------------------------------------------------------------
# 4️⃣ Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    train_z_predictor_with_custom_loss()

    cfg = Config()
    # model_path = "checkpoint_20240826/z_pred_0.03.pth"
    model_path = f"{cfg.save_path}/z_predictor_custom_loss.pth"
    device = "cuda:0"
    visualize_umap = True
    save_embeddings = True
    concatenate_embeddings = True
    exp_name = cfg.save_path.split("_")[1]
    save_umap_path = f"umap_validation_{exp_name}"
    
    # cfg.val_filepaths_2 = cfg.val_filepaths[i:i+100]
    for i in range(100, len(cfg.val_filepaths), 100):
        cfg.val_filepaths_2 = cfg.val_filepaths[i:i+100]
        validate_saved_model(model_path, 
                            device =device, 
                            visualize_umap = visualize_umap,
                            save_embeddings = save_embeddings,
                            save_umap_path = save_umap_path,
                            concatenate_embeddings = concatenate_embeddings,
                            cfg = cfg)
