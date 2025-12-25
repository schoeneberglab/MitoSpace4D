import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from simclr.models_simple import Lightweight3DResNet
from utils.utils import load_config
from train_simclr import SimCLRRunner

device = "cuda" if torch.cuda.is_available() else "cpu"

def perturb_shuffle(video_tensor):
    """
    video_tensor: (T,C,D,H,W)
    returns shuffled video and permutation used
    """
    T = video_tensor.shape[0]
    perm = np.random.permutation(T)
    shuffled = video_tensor[perm]
    return shuffled, perm

# ------------------------ Parallel Data Loading ------------------------
def load_and_preprocess_file(args):
    """
    Worker function for parallel data loading.
    Loads and preprocesses a single .npy file.
    
    Args:
        args: tuple of (file_path, folder_path, use_perturbed)
        use_perturbed: If True, applies temporal shuffling; if False, uses original order
    
    Returns:
        tuple: (success: bool, video_tensor or None, error_message or None)
    """
    file_path, folder_path, use_perturbed = args
    try:
        full_path = osp.join(folder_path, file_path)
        data = np.load(full_path).astype(np.float32)   # (T, C, Z, Y, X)
        
        # normalize channels like your earlier code
        data[:, 0] = np.clip(data[:, 0], 0, 25000) / 25000.
        data[:, 1] = np.clip(data[:, 1], 0, 10000) / 10000.
        
        video = torch.tensor(data).float()
        if use_perturbed:
            video, perm = perturb_shuffle(video)  # (T, C, D, H, W)
        # else: use original video as-is
        return (True, video, None)
    except Exception as e:
        return (False, None, str(e))

# ------------------------ Utilities ------------------------
def get_lstm_hidden(model, video_tensor, batch_size=1):
    """
    Returns LSTM hidden states per timestep.
    video_tensor: (B, T, C, D, H, W) or (1, T, C, D, H, W)
    Output: hidden_states: (B, T, D) or (T, D) if batch_size=1
    """
    model.eval()
    with torch.no_grad():
        # x: per-timestep embeddings, out: final embedding, lstm_hidden: (B, T, D) or (T, D)
        _, _, lstm_hidden = model(video_tensor.to(device))
    if batch_size == 1 and lstm_hidden.dim() == 2:
        return lstm_hidden.cpu()  # (T, D)
    return lstm_hidden.cpu()  # (B, T, D)

# ------------------------ Main Function ------------------------
def plot_drug_average_hidden_states(model, data_root, k=10, output_dir=None, batch_size=8, chunk_size=50, num_workers=None, use_perturbed=False):
    """
    For each drug folder:
        - loads k samples
        - extracts per-timestep LSTM hidden states (T, D)
        - averages across samples → (T, D)
        - plots as heatmap and saves to file
    
    Args:
        model: The trained model
        data_root: Root directory containing drug folders
        k: Number of samples to process per drug
        output_dir: Directory to save plots (if None, creates 'plots' in data_root)
        batch_size: Batch size for processing videos (to optimize GPU usage)
        chunk_size: Process videos in chunks to avoid OOM errors
        num_workers: Number of parallel workers for data loading (default: cpu_count())
        use_perturbed: If True, applies temporal shuffling; if False, uses original temporal order (default: False)
    """
    if output_dir is None:
        output_dir = osp.join(data_root, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Cap at 16 to avoid too many processes
    
    drug_folders = sorted([x for x in os.listdir(data_root) if osp.isdir(osp.join(data_root, x))])
    drug_hidden_map = {}

    # Process each drug folder
    for drug_folder in tqdm(drug_folders, desc="Processing drug folders"):
        folder_path = osp.join(data_root, drug_folder)
        filenames = sorted([
            f for f in os.listdir(folder_path)
            if osp.isfile(osp.join(folder_path, f)) and f.endswith('.npy')
        ])

        if len(filenames) == 0:
            print(f"Warning: No .npy files found in {drug_folder}, skipping...")
            continue

        num_samples = len(filenames)
        selected_idx = np.random.choice(num_samples, size=min(k, num_samples), replace=False)
        selected_filenames = [filenames[idx] for idx in selected_idx]

        hidden_list = []

        # Process in chunks to avoid OOM
        for chunk_start in tqdm(range(0, len(selected_filenames), chunk_size), 
                                desc=f"  Processing {drug_folder}", leave=False):
            chunk_end = min(chunk_start + chunk_size, len(selected_filenames))
            chunk_filenames = selected_filenames[chunk_start:chunk_end]
            
            # Parallel loading and preprocessing
            load_args = [(filename, folder_path, use_perturbed) for filename in chunk_filenames]
            
            videos_batch = []
            with Pool(processes=num_workers) as pool:
                results = pool.map(load_and_preprocess_file, load_args)
            
            # Collect successfully loaded videos
            for success, video, error_msg in results:
                if success:
                    videos_batch.append(video)
                elif error_msg:
                    # Only print errors for debugging (can be verbose)
                    pass  # Uncomment if you want to see errors: print(f"Error loading file: {error_msg}")
            
            if len(videos_batch) == 0:
                continue
            
            # Process in batches
            for batch_start in range(0, len(videos_batch), batch_size):
                batch_end = min(batch_start + batch_size, len(videos_batch))
                batch_videos = videos_batch[batch_start:batch_end]
                
                # Stack into batch: (B, T, C, D, H, W)
                batch_tensor = torch.stack(batch_videos, dim=0).to(device)
                
                # Extract hidden states for batch
                # Note: This assumes model can handle batched input
                # If not, fall back to individual processing
                try:
                    batch_hidden = get_lstm_hidden(model.model, batch_tensor, batch_size=len(batch_videos))
                    if batch_hidden.dim() == 3:  # (B, T, D)
                        for i in range(batch_hidden.shape[0]):
                            hidden_list.append(batch_hidden[i].numpy())
                    else:  # (T, D) - single sample
                        hidden_list.append(batch_hidden.numpy())
                except Exception as e:
                    # Fallback to individual processing if batch fails
                    print(f"Batch processing failed for {drug_folder}, falling back to individual: {e}")
                    for video in batch_videos:
                        try:
                            video_single = video.unsqueeze(0).to(device)  # (1, T, C, D, H, W)
                            hidden = get_lstm_hidden(model.model, video_single, batch_size=1)
                            hidden_list.append(hidden.numpy())
                        except Exception as e2:
                            print(f"Error processing individual video: {e2}")
                            continue
                        finally:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if len(hidden_list) == 0:
            print(f"Warning: No valid samples processed for {drug_folder}, skipping...")
            continue

        # (num_samples, T, D) → average over samples
        hidden_array = np.stack(hidden_list, axis=0)
        mean_hidden = hidden_array.mean(axis=0)  # (T, D)

        drug_hidden_map[drug_folder] = mean_hidden

        # ---------------------- Plotting and Saving ----------------------
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            mean_hidden.T,  # (D, T)
            cmap="coolwarm",
            cbar=True,
        )

        # Titles and labels with bigger font for readability
        data_type = "Perturbed" if use_perturbed else "Original"
        plt.title(f"Average LSTM Hidden States for {drug_folder} ({data_type})", fontsize=18)
        plt.xlabel("Time (frame index)", fontsize=14)
        plt.ylabel("Embedding dimension", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)

        # Save plot
        data_type_suffix = "perturbed" if use_perturbed else "original"
        plot_filename = osp.join(output_dir, f"{drug_folder}_lstm_hidden_states_{data_type_suffix}.png")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Saved plot for {drug_folder} to {plot_filename}")

    return drug_hidden_map

if __name__ == "__main__":
    cfg = load_config('/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml')
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    # data_root = '/mnt/aquila/others/MitoSpace4D/2025_summer_new'
    data_root = '/mnt/aquila/others/MitoSpace4D/data/aligned/'
    device = 'cuda'

    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=False).to(device)

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=287-step=83534-val_loss=0.00.ckpt"
    # checkpoint_path = f"{proj_dir}/runs/lightning_logs/resnetbilistm_encoder_consistent_temporal/checkpoints/epoch=90-step=19565-val_loss=0.00.ckpt"
    dataset_name = cfg["evaluate"]["dataset"]

    # print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )
    model.eval()

    output_dir = osp.join(proj_dir, "temporal_experiments/lstm_visualisations/plots")
    os.makedirs(output_dir, exist_ok=True)
    
    drug_hidden_map = plot_drug_average_hidden_states(
        model,
        data_root="/mnt/aquila/others/MitoSpace4D/data/aligned/",
        k=300,
        output_dir=output_dir,
        batch_size=1,  # Adjust based on GPU memory
        chunk_size=25,  # Process 50 videos at a time to avoid OOM
        num_workers=None,  # None = auto (uses min(cpu_count(), 16))
        use_perturbed=False  # False = original temporal order, True = shuffled
    )
    
    print(f"\nAll plots saved to: {output_dir}")