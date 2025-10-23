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
import torch
from torch.utils.data import Dataset
import numpy as np

# --- Helper function from previous response ---
def normalize_and_mask(img_slice_2d, eps=1e-6, mask_threshold=0.1):
    """
    Remove morphology-related brightness (TMRM) from MitoTracker
    to emphasize functional membrane potential signal.

    Args:
        img_slice_2d: A 2D slice from the image, expected shape [C, H, W]
             channel 0 = TMRM (morphology)
             channel 1 = MitoTracker (function + morphology)
        eps: small value to prevent divide-by-zero
        mask_threshold: relative threshold for mitochondrial mask
    Returns:
        functional_masked: The processed 2D image showing functional signal.
        mask: The binary mitochondrial mask.
        (If C < 2, returns zeros for functional_masked and mask)
    """
    if img_slice_2d.ndim != 3 or img_slice_2d.shape[0] < 2:
        # print("Warning: normalize_and_mask received input not matching [C>=2, H, W]. Returning zeros.")
        H, W = img_slice_2d.shape[-2:]
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    tmrm = img_slice_2d[0, :, :]      # morphology only
    mitotk = img_slice_2d[1, :, :]    # morphology + function

    # Convert to float32 for calculations
    tmrm = tmrm.astype(np.float32)
    mitotk = mitotk.astype(np.float32)

    # --- Step 1: background subtraction ---
    # Only subtract if there are negative values or if a shift to non-negative is desired
    if tmrm.min() < 0: tmrm = tmrm - tmrm.min()
    if mitotk.min() < 0: mitotk = mitotk - mitotk.min()

    # --- Step 2: normalize both to [0, 1] ---
    tmrm_max = tmrm.max()
    mitotk_max = mitotk.max()
    if tmrm_max > 0: tmrm = tmrm / (tmrm_max + eps)
    else: tmrm = np.zeros_like(tmrm) # Avoid NaN/Inf if all zeros

    if mitotk_max > 0: mitotk = mitotk / (mitotk_max + eps)
    else: mitotk = np.zeros_like(mitotk)

    # --- Step 3: remove morphology contribution ---
    functional = mitotk / (tmrm + eps)
    functional_max = functional.max()
    if functional_max > 0:
        functional = functional / (functional_max + eps)
    else:
        functional = np.zeros_like(functional)
    
    # --- Step 4: create mitochondrial mask ---
    mask = (tmrm > mask_threshold).astype(np.float32)

    # --- Step 5: apply mask ---
    functional_masked = functional * mask

    return functional_masked, mask


# Conditions to train on:
# 20240729 control 0
# 20240730 p110 1
# 20240826 nocodazole 17
# 20240830 colchicine 18
# --- 1. Configuration ---
class Config:
    # --- Dataset specific ---
    # List of .npy file paths
    base_path = "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240830/"
    image_filepaths = [

        # f"{base_path}{i}" for i in os.listdir(base_path)
        # "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy",
        # "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000069.npy",
        # "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000079.npy",
        # "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000089.npy",
        # "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000099.npy",
        # # Add more file paths here as needed for multiple datasets
        # # "/path/to/your/another_image.npy",
    ]

    files = os.listdir(base_path)
    for i in files[0:150]:
        image_filepaths.append(f"{base_path}{i}")

    val_filepaths = []

    for i in files:
        val_filepaths.append(f"{base_path}{i}")
    # --- Channel Handling ---
    # True to create a 3rd channel from 'functional_masked' output of normalize_and_mask
    create_third_channel = True 
    use_only_mitotracker = True
    # --- Model specific ---
    model_name = "MCG-NJU/videomae-base" # A good choice for video classification
    image_processor_name = "MCG-NJU/videomae-base" # The associated image processor
    
    # --- Training specific ---
    test_size_z = 0.3 # 30% of Z-slices for testing, 70% for training
    batch_size = 16 # Reduced batch size, especially if creating 3 channels and multiple files
    num_epochs = 20
    learning_rate = 1e-5
    random_seed = 42
    do_rescale = False
    device = "cuda:0"
    save_path = "checkpoint_20240830"

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.random_seed)

class ZVolumeAccuracyLoss(nn.Module):
    """
    Computes:  loss = 1 - mean(% correct per volume)
    Each volume’s % correct = (# correct slices / # total slices)
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, volume_ids):
        """
        Args:
            logits: (N, num_classes)
            labels: (N,)
            volume_ids: (N,) tensor where same volume has same ID
        Returns:
            scalar tensor: loss value
        """
        preds = torch.argmax(logits, dim=1)
        correct_mask = (preds == labels).float()

        unique_vols = torch.unique(volume_ids)
        vol_accs = []
        for vid in unique_vols:
            mask = (volume_ids == vid)
            if mask.sum() > 0:
                acc = correct_mask[mask].mean()
                vol_accs.append(acc)

        if len(vol_accs) == 0:
            return torch.tensor(0.0, device=logits.device)

        mean_vol_acc = torch.stack(vol_accs).mean()
        loss = 1.0 - mean_vol_acc
        return loss



class ZSliceDataset(Dataset):
    """
    PyTorch Dataset for individual Z-slice video clips.
    
    Each sample corresponds to a single z-slice video (T, C, H, W).
    Labels are derived from the Z index via a provided z-to-label mapping.
    """

    def __init__(
        self,
        all_data_combined: torch.Tensor,
        z_indices: list,
        volume_ids: list,
        image_processor,
        create_third_channel: bool = False,
        use_only_mitotracker: bool = True,
    ):
        """
        Args:
            all_data_combined (torch.Tensor): Combined tensor of shape (N, T, C, H, W),
                                              where N = total Z-slice samples.
            z_indices (list[int]): Original Z indices corresponding to each sample.
            volume_ids (list[int]): Volume ID for each sample.
            image_processor: Hugging Face image processor to prepare frames.
            create_third_channel (bool): Whether to compute and add a 3rd channel.
            use_only_mitotracker (bool): Whether to use only the MitoTracker channel.
        """
        self.all_data_combined = all_data_combined
        self.original_z_indices = z_indices
        self.volume_ids = volume_ids
        self.image_processor = image_processor
        self.create_third_channel = create_third_channel
        self.use_only_mitotracker = use_only_mitotracker
        self.z_to_label_mapping = None  # will be set externally via set_label_mapping()

    def set_label_mapping(self, z_to_label_map: dict):
        """Assigns a z-to-label mapping (e.g., {z_idx: class_label})."""
        self.z_to_label_mapping = z_to_label_map

    def __len__(self):
        """Number of z-slice samples."""
        return self.all_data_combined.shape[0]

    def __getitem__(self, idx: int):
        """Returns one video clip and its corresponding label + volume ID."""
        video_clip_raw = self.all_data_combined[idx]  # (T, C, H, W)
        original_z_index = self.original_z_indices[idx]
        volume_id = self.volume_ids[idx]

        # -------------------------------
        # 1️⃣ Channel preparation
        # -------------------------------
        if self.use_only_mitotracker:
            # Use only the MitoTracker channel (replicate to RGB)
            mito_channel = video_clip_raw[:, 1, :, :]  # assuming channel 1 = MitoTracker
            current_channels = [mito_channel] * 3
        else:
            # Use original channels (TMRM + MitoTracker)
            num_channels = video_clip_raw.shape[1]
            if num_channels >= 2:
                ch0 = video_clip_raw[:, 0, :, :]  # TMRM
                ch1 = video_clip_raw[:, 1, :, :]  # MitoTracker
            else:
                # Duplicate if only one channel
                ch0 = ch1 = video_clip_raw[:, 0, :, :]
            current_channels = [ch0, ch1]

            # Optionally add a functional (third) channel
            if self.create_third_channel:
                func_frames = []
                for t_idx in range(video_clip_raw.shape[0]):
                    masked_frame, _ = normalize_and_mask(video_clip_raw[t_idx].cpu().numpy())
                    func_frames.append(masked_frame)
                func_channel = torch.from_numpy(np.stack(func_frames)).float()
                current_channels.append(func_channel)
            else:
                # If not creating, just duplicate the last channel
                current_channels.append(current_channels[-1])

        # Stack → (T, C_new, H, W)
        processed_video_clip = torch.stack(current_channels, dim=1)

        # -------------------------------
        # 2️⃣ Normalization
        # -------------------------------
        clip_min, clip_max = processed_video_clip.min(), processed_video_clip.max()
        if clip_max > clip_min:
            processed_video_clip = (processed_video_clip - clip_min) / (clip_max - clip_min)
        else:
            processed_video_clip = torch.zeros_like(processed_video_clip)

        # -------------------------------
        # 3️⃣ Convert to image processor format
        # -------------------------------
        frames_for_processor = [
            processed_video_clip[t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            for t in range(processed_video_clip.shape[0])
        ]

        processed_inputs = self.image_processor(
            images=frames_for_processor, return_tensors="pt"
        )
        pixel_values = processed_inputs["pixel_values"].squeeze(0)  # (T, C_proc, H, W)

        # -------------------------------
        # 4️⃣ Label mapping
        # -------------------------------
        if self.z_to_label_mapping is None:
            raise ValueError("z_to_label_mapping not set. Call set_label_mapping() first.")

        label = self.z_to_label_mapping[original_z_index]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
            "volume_id": torch.tensor(volume_id, dtype=torch.long),
        }

# --- 3. Main Training Function ---
def train_z_predictor_old():
    # Load and combine data from all specified file paths
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
    
    # VideoMAE models usually expect 3 channels by default.
    # We might need to explicitly set `num_channels` in a custom config
    # or handle the channel repetition/padding if the HF model doesn't automatically adapt.
    # For now, let's assume `AutoModelForVideoClassification` will adapt or we need to preprocess
    # to 3 channels regardless (e.g., duplicate a channel if only 2 are used as input to the model)
    # The image_processor often handles padding/repeating if input channels != model's expected.
    
    # Let's adjust the model's first convolution layer if it expects a different number of channels.
    # However, AutoModelForVideoClassification is usually designed to handle this via `image_processor`
    # or expects `num_channels` to be passed during init.
    # VideoMAE models from MCG-NJU typically use `num_channels=3` for RGB.
    # If we pass 2 channels to a model expecting 3, we might need a dummy channel or repetition.
    # Let's trust the image_processor to handle initial channel transformation, or just pass `num_channels`
    # explicitly if the model config supports it.
    
    # The default VideoMAE base config has `num_channels=3`.
    # When you pass `pixel_values` from the processor, it will usually be (N, T, 3, H, W)
    # The processor itself will handle converting (T, C_new, H, W) to (T, 3, H_proc, W_proc) by repeating/padding channels if C_new != 3.
    
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
    
    z_to_label = {z : int(z // float(max_z/num_labels)) for z in range(max_z)}
    
    # Set the global label mapping for both datasets
    train_dataset.set_label_mapping(z_to_label_mapping)
    test_dataset.set_label_mapping(z_to_label_mapping)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    print("Length", test_loader)
    # Training loop
    for epoch in range(Config.num_epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        min_loss = 100
        losses = []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            volume_id = batch['volume_id'].to(device)
               # print(pixel_values.shape, labels.shape)
            print(volume_id)
            optimizer.zero_grad()
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        
        train_accuracy = train_correct / train_total
        losses.append(avg_train_loss)

        if avg_train_loss < min_loss and ((epoch+1)%5)==0: 
            min_loss = avg_train_loss
            torch.save(model, f"checkpoints/Epoch_{epoch+1}_loss_{avg_train_loss:.2f}_acc_{train_accuracy:.2f}.pth")

        print(f"Epoch {epoch+1}/{Config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            p1, predicted = torch.max(logits, 1)
            print("P1, labels",p1, predicted,labels)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = test_correct / test_total
    print(f"Epoch {epoch+1}/{Config.num_epochs}, Test Accuracy: {test_accuracy:.4f}\n")

    print("Training finished!")
    
    # --- Visualize some test predictions ---
    visualize_predictions(model, test_dataset, image_processor, device, label_to_z_mapping, num_samples=5)


def visualize_predictions(model, dataset, image_processor, device, label_to_z_map, num_samples=5):
    """
    Visualizes actual vs. predicted Z-slices for a few samples from the test set.
    """
    model.eval()
    if len(dataset) == 0:
        print("No samples in test dataset to visualize.")
        return

    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    print("\n--- Visualizing Test Predictions ---")
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1: # Ensure axes is always iterable
        axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device) # Add batch dimension
            true_label_local = sample["labels"].item()
            
            # Map local label back to original Z-index
            true_original_z = label_to_z_map[true_label_local]

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predicted_label_local = torch.argmax(logits, dim=-1).item()
            
            # Map local predicted label back to original Z-index
            predicted_original_z = label_to_z_map[predicted_label_local]

            # To get a 'displayable' image, we need the original video_clip
            # before it went through the processor (which might resize/normalize)
            original_video_clip_display_raw = dataset.all_data_combined[idx] # (T, C_orig, H, W)
            
            # Reconstruct the processed video clip for visualization (including 3rd channel if created)
            display_channels = []
            display_channels.append(original_video_clip_display_raw[:, 0, :, :])
            if original_video_clip_display_raw.shape[1] >= 2:
                display_channels.append(original_video_clip_display_raw[:, 1, :, :])
            else: # If only one channel in original, repeat it for Ch1 display
                display_channels.append(original_video_clip_display_raw[:, 0, :, :])

            if dataset.create_third_channel:
                functional_channel_frames_display = []
                for t_idx in range(original_video_clip_display_raw.shape[0]):
                    functional_masked_frame, _ = normalize_and_mask(original_video_clip_display_raw[t_idx, :, :, :].cpu().numpy())
                    functional_channel_frames_display.append(functional_masked_frame)
                display_channels.append(torch.from_numpy(np.stack(functional_channel_frames_display)).float())
            
            reconstructed_display_clip = torch.stack(display_channels, dim=1) # (T, C_display, H, W)

            # Take a middle frame for display and the first channel (TMRM)
            display_frame_idx = reconstructed_display_clip.shape[0] // 2
            display_image = reconstructed_display_clip[display_frame_idx, 0, :, :].cpu().numpy() # Show Ch0 (TMRM)
            
            # Normalize for display if not already [0,1] or [0,255]
            display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-6)

            axes[i][0].imshow(display_image, cmap='gray')
            axes[i][0].set_title(f"True Z: {true_original_z}")
            axes[i][0].axis('off')

            axes[i][1].text(0.5, 0.5, f"Predicted Z: {predicted_original_z}",
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=12, color='green' if predicted_original_z == true_original_z else 'red',
                            transform=axes[i][1].transAxes) # Use transform for relative positioning
            axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_z_predictor()