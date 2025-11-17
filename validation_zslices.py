from typing import Any
import numpy as np
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
import umap
import seaborn as sns

'''
1. We want to load the existing model
2. load the config from videomae_zslicer and import the entire dataset
3. After importing run the validation again on the enitre dataset to calculate the volume accuracy for each bigger 4D image (not just over a batch. Load them in CPU to allow this)
3. Extract the embeddings and concatenate embeddings along the z dimensions
4. We also want to visualise the embeddings in UMAP embedding space 

'''

@torch.no_grad()
def validate_saved_model(model_path, device=None, 
                        visualize_umap=True, 
                        save_umap_path=None, 
                        concatenate_embeddings=True,
                        save_embeddings=True,
                        cfg=Config):

    """
    Validate a saved model on the entire dataset volume-by-volume.
    """

    # ------------------------------------------------------------
    # 1️⃣ Load model
    # ------------------------------------------------------------
    print(f"🔹 Loading model from {model_path}")
    if model_path.endswith(".pt") or model_path.endswith(".pth") and "incremental" not in model_path:
        checkpoint = torch.load(model_path, map_location="cpu")
        
        model = AutoModelForVideoClassification.from_pretrained(checkpoint)
        model.load_state_dict(checkpoint)

    elif model_path.endswith(".pt") or model_path.endswith(".pth") and "incremental" in model_path:
        # checkpoint = torch.load(model_path, map_location="cpu")
        # model = AutoModelForVideoClassification.from_pretrained(Config.model_name, num_labels=3)
        # model.load_state_dict(checkpoint)
        checkpoint = torch.load(model_path, map_location=device)
        model = AutoModelForVideoClassification.from_pretrained(cfg.model_name, num_labels=3)
        model.load_state_dict(checkpoint["model_state"])

    else:
        model = AutoModelForVideoClassification.from_pretrained(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model has been loaded")
    model.eval()

    # Initialize image processor and model
    image_processor = AutoImageProcessor.from_pretrained(cfg.image_processor_name, do_rescale=False)

    #-------------------------------------------------------------
    # 2 Load Data volume by volume

    all_z_video_clips = [] # Will store (T, C_orig, H, W) for each z_slice from all files
    all_original_z_indices = [] # Will store the original z_index for each video clip
    global_slice_counter = [] # Stores the actual global slice number 
    # Store global Z index across all files to handle unique Z values
    global_z_idx_counter = 0 
    global_volume_counter = []


    for volume_id , filepath in enumerate(cfg.val_filepaths_2):
        try:
            full_image_data_np = np.load(filepath)
            # Ensure it's a PyTorch tensor
            current_data_tensor = torch.from_numpy(full_image_data_np) 
            
            
            # Expected (T, C, Z, H, W)
            if current_data_tensor.shape[1] != 2:
                C, time_points, Z, H, W = current_data_tensor.shape
                current_data_tensor = current_data_tensor.permute(1, 0, 2, 3, 4)
            else:
                time_points, C, Z, H, W = current_data_tensor.shape
                
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


    all_data_combined_tensor = torch.stack(all_z_video_clips, dim=0).to(torch.int32) # Shape: (Total_Z_slices_across_files, T, C_orig, H, W)
    total_z_samples = all_data_combined_tensor.shape[0]
    

    dataset = ZSliceDataset(
        all_data_combined_tensor,
        all_original_z_indices,
        global_volume_counter,
        image_processor,
        create_third_channel=Config.create_third_channel
    )

    # set label mapping
    num_labels = 3
    max_z = z_slices_file # this assumes all files are of similar z length
    divider = max_z//num_labels
    all_unique_original_z_indices = sorted(list(set(all_original_z_indices)))
    z_to_label_mapping = {z: i//divider for i, z in enumerate(all_unique_original_z_indices)}
    label_to_z_mapping = {i: i*divider + divider*0.5 for i in set(z_to_label_mapping.values())}
    dataset.set_label_mapping(z_to_label_mapping)

    print(f"✅ Loaded dataset with {len(dataset)} samples")

    # ------------------------------------------------------------
    # 3️⃣ Run validation per volume
    # ------------------------------------------------------------
    print("🔹 Running validation over full volumes...")
    model.eval()

    unique_vols = sorted(set(global_volume_counter))

    assert len(unique_vols) == len(cfg.val_filepaths_2), "Number of unique volumes should be equal to the number of validation filepaths"
    assert len(unique_vols)*60 == len(global_volume_counter), "Number of unique volumes should be equal to the number of unique volume ids multiplied by 60"
    
    per_volume_acc = {}
    all_preds, all_labels, all_embeds, all_vol_ids = [], [], [], []

    for vol in tqdm(unique_vols, desc="Evaluating volumes"):
        # get indices belonging to this volume
        vol_indices = [i for i, vid in enumerate(global_volume_counter) if vid == vol]

        preds_v, labels_v, embeds_v = [], [], []

        for idx in vol_indices:
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            label = sample["labels"].unsqueeze(0).to(device)

            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            pred = torch.argmax(outputs.logits, dim=1).cpu().item()
            true_label = label.cpu().item()

            preds_v.append(pred)
            labels_v.append(true_label)

            hidden = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()  # (1, hidden_dim)
            embeds_v.append(hidden)

            all_preds.append(pred)
            all_labels.append(true_label)
            all_vol_ids.append(vol)

        embeds_v = np.concatenate(embeds_v, axis=0)
        all_embeds.append(embeds_v)

        # per-volume accuracy
        vol_acc = np.mean(np.array(preds_v) == np.array(labels_v))
        per_volume_acc[vol] = vol_acc
    print("Per volume accuracies are: ", per_volume_acc)
    # compute mean accuracy
    mean_acc = np.mean(list(per_volume_acc.values()))
    print(f"\n✅ Mean per-volume accuracy: {mean_acc:.3f}")

    # ------------------------------------------------------------
    # 4️⃣ Concatenate embeddings along z-dimension
    # ------------------------------------------------------------
    print("🔹 Concatenating embeddings per volume...")
    if concatenate_embeddings : 
        concatenated_embeds = []
        for emb_v in all_embeds:
            concat_emb = np.mean(emb_v, axis=0)  # average across z for now
            concatenated_embeds.append(concat_emb)
    else:
        concatenated_embeds = all_embeds
        

    # Save embeddings
    if save_embeddings:
        # Extract filename from the last validation filepath
        filepaths = map(lambda x: os.path.basename(x).replace('.npy', ''), cfg.val_filepaths_2)
        # filepaths_2 = map(lambda x: os.path.basename(x).replace('.npy', ''), cfg.val_filepaths_2    
        # get drug label from the filepath 
        # drug_label_list = [
        drug_label_list = list(map(lambda x: os.path.split(x)[-2].split("/")[-1], cfg.val_filepaths_2))
        assert len(drug_label_list) == len(cfg.val_filepaths_2), "Number of drug labels should be equal to the number of filepaths"
        # Create save directory if it doesn't exist
        save_dir = os.path.join(cfg.save_path, "embeddings")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save concatenated embeddings
        for i, filename in enumerate(filepaths):
            embed_save_path = os.path.join(save_dir, f"embeddings_{drug_label_list[i]}_{filename}.npy")
            np.save(embed_save_path, concatenated_embeds[i])
        print(f"✅ Saved embeddings to {embed_save_path}")
        

    concatenated_embeds = np.stack(concatenated_embeds)
    print(f"✅ Concatenated embeddings shape: {concatenated_embeds.shape}")

    
    if visualize_umap:
        print("🔹 Running UMAP projection...")
        
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine", random_state=42)
        umap_embeds = reducer.fit_transform(concatenated_embeds)

        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=umap_embeds[:, 0], y=umap_embeds[:, 1],
                        hue=list(per_volume_acc.values()),
                        palette="viridis", s=80)
        plt.title("UMAP projection of Volume Embeddings (colored by volume accuracy)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(title="Vol Accuracy", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filepaths = list(map(lambda x: os.path.basename(x).replace('.npy', ''), cfg.val_filepaths_2))

        if save_umap_path:
            plt.savefig(f"{cfg.save_path}/{save_umap_path}_{filepaths[0]}_{filepaths[-1]}.png", dpi=300)
            print(f"✅ Saved UMAP plot to {save_umap_path}")
        else:
            plt.show()

    # ------------------------------------------------------------
    # 6️⃣ Return results
    # ------------------------------------------------------------
    return {
        "volume_accuracy": per_volume_acc,
        "mean_accuracy": mean_acc,
        "embeddings_per_volume": concatenated_embeds,
        "all_preds": np.array(all_preds),
        "all_labels": np.array(all_labels),
        "all_volume_ids": np.array(all_vol_ids),
    }

@torch.no_grad()
def extract_embeddings(model, dataloader, device, layer_name="pooler"):
    """
    Extract intermediate embeddings from the model.

    Args:
        model: Trained model (e.g., AutoModelForVideoClassification).
        dataloader: DataLoader.
        device: torch.device.
        layer_name: Which part to extract from ('pooler', 'classifier', etc.)
    Returns:
        embeddings: np.ndarray of shape (N_samples, D)
        labels: np.ndarray of shape (N_samples,)
    """
    model.eval()
    all_embeds = []
    all_labels = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, output_hidden_states=True)

        # Get embeddings from a chosen layer
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]  # last transformer block
            # Mean pool over time dimension (T)
            embeddings = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        elif hasattr(outputs, layer_name):
            embeddings = getattr(outputs, layer_name)
        else:
            raise ValueError("Could not extract embeddings: model output missing hidden_states.")

        all_embeds.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    print(f"✅ Extracted embeddings shape: {all_embeds.shape}")
    return all_embeds, all_labels


if __name__ == "__main__":
    # model_path = "checkpoint_20240826/z_pred_0.03.pth"
    # device = "cuda:0"
    # visualize_umap = True
    # save_embeddings = True
    # concatenate_embeddings = True
    # save_umap_path = "umap_validation_20240826"
    # cfg = Config()
    cfg = Config()
    # model_path = "checkpoint_20240826/z_pred_0.03.pth"
    
    cfg.save_path = "checkpoint_new_data_all_drugs"
    model_path = f"{cfg.save_path}/z_predictor_incremental.pth"
    # cfg.val_filepaths_2 = cfg.val_filepaths[0:]
    pick_folders = [
        # "20240729-1",
        "20240805-1",
        "20240814-1",
        "20240911-1",
        "20240826-1",
        "20240830-1"
        # Add more folder names or date-based identifiers as needed
    ]
    device = "cuda:0"
    visualize_umap = False
    save_embeddings = True
    concatenate_embeddings = True
    exp_name = cfg.save_path.split("_")[1]
    save_umap_path = f"umap_validation_{exp_name}"
    # cfg.val_filepaths_2 = cfg.val_filepaths[i:i+100]
    print(len(cfg.val_filepaths), cfg.val_filepaths[0], cfg.val_filepaths[-1], cfg.val_filepaths[100], cfg.val_filepaths[200])
    batch_size = 150
    for folder in pick_folders:
        idxs = [i for i, fpath in enumerate(cfg.val_filepaths) if folder in fpath]
        if not idxs:
            print(f"⚠️ No filepaths found for folder: {folder}")
            continue
        for start in range(0, len(idxs), 3*batch_size):
            selected_idxs = idxs[start:start+batch_size]
            if not selected_idxs:
                continue
            cfg.val_filepaths_2 = [cfg.val_filepaths[i] for i in selected_idxs]
            print(f"Folder: {folder} | Batch size: {len(cfg.val_filepaths_2)}")
            validate_saved_model(model_path, 
                                device =device, 
                                visualize_umap = visualize_umap,
                                save_embeddings = save_embeddings,
                                save_umap_path = save_umap_path,
                                concatenate_embeddings = concatenate_embeddings,
                                cfg = cfg)
    # validate_saved_model(model_path, 
    #                     device =device, 
    #                     visualize_umap = visualize_umap,
    #                     save_embeddings = save_embeddings,
    #                     save_umap_path = save_umap_path,
    #                     concatenate_embeddings = concatenate_embeddings,
    #                     cfg = cfg)