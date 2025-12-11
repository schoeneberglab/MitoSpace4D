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
from utils.vis import plot_cm, plot_cm_save_fig
from videomae_zslicer import (
    Config,
    ZSliceDataset,
    normalize_and_mask,
    ZVolumeAccuracyLoss
)
import umap
import seaborn as sns
# --- Compute confusion matrix for ground-truth vs predicted labels ---
from sklearn.metrics import confusion_matrix, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.neighbors import KDTree
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import concurrent.futures
# INSERT_YOUR_REWRITE_HERE
import os
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

# --- Entropy based on k-NN in embedding space for each volume ---
from sklearn.neighbors import NearestNeighbors
import numpy as np

'''
1. We want to load the existing model
2. load the config from videomae_zslicer and import the entire dataset
3. After importing run the validation again on the enitre dataset to calculate the volume accuracy for each bigger 4D image (not just over a batch. Load them in CPU to allow this)
3. Extract the embeddings and concatenate embeddings along the z dimensions
4. We also want to visualise the embeddings in UMAP embedding space 

'''

def topKfrequent(nums, weights, k, weighted=False):
    """Find top k most frequent items, optionally weighted."""
    d = dict()
    for i, n in enumerate(nums):
        if weighted:
            d[n] = d.setdefault(n, 0) + weights[i]
        else:
            d[n] = d.setdefault(n, 0) + 1
    sortedNumsKeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)
    return sortedNumsKeys[:k]


def cosine_distance(eval_embeddings, train_embeddings, weighted=False, temperature=1.):
    """Compute cosine distance matrix and sorted indices."""
    dist_matrix = eval_embeddings @ train_embeddings.T
    if weighted:
        dist_matrix = dist_matrix / temperature
        dist_matrix = np.exp(dist_matrix)
    dist_matrix_idxs = (-1 * dist_matrix).argsort(1)  # descending order
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    return dist_matrix_sorted, dist_matrix_idxs


def l2_distance(eval_embeddings, train_embeddings):
    """Compute L2 distance matrix and sorted indices."""
    dist_matrix = np.linalg.norm(eval_embeddings[:, None] - train_embeddings[None, :], axis=-1)
    dist_matrix_idxs = dist_matrix.argsort(1)
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    return dist_matrix_sorted, dist_matrix_idxs


def nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs,
                                num_neighbors=[100], verbose=True):
    """Evaluate nearest neighbor classification accuracy."""
    preds = None
    # normalize the distance matrix
    dist_matrix = (dist_matrix + 1) / 2
    
    results = {}
    for k in num_neighbors:
        if verbose:
            print(f"################ Evaluation for {k} Neighbors #################")
        
        correct_preds = {top_n: 0 for top_n in top_ns}
        correct_preds_per_class = {top_n: {lbl: 0 for lbl in np.unique(train_labels)} for top_n in top_ns}
        preds = {top_n: [] for top_n in top_ns}
        
        correct_preds_idxs = {top_n: [] for top_n in top_ns}
        incorrect_preds_idxs = {top_n: [] for top_n in top_ns}
        
        pbar = tqdm(total=len(dist_matrix)) if verbose else None
        for i in range(len(dist_matrix)):
            eval_lbl = eval_labels[i]
            k_nearest_nbs = train_labels[dist_matrix_idxs[i][:k]]
            k_nearest_dist = dist_matrix[i][:k]
            
            for top_n in top_ns:
                top_most_freq_lbls = topKfrequent(k_nearest_nbs, k_nearest_dist, top_n, weighted=True)
                if eval_lbl in top_most_freq_lbls:
                    correct_preds[top_n] += 1
                    correct_preds_per_class[top_n][eval_lbl] += 1
                    preds[top_n].append(eval_lbl)
                    correct_preds_idxs[top_n].append(i)
                else:
                    preds[top_n].append(top_most_freq_lbls[0])
                    incorrect_preds_idxs[top_n].append(i)
            
            if pbar is not None:
                pbar.update(1)
        
        for top_n in top_ns:
            correct = correct_preds[top_n]
            if verbose:
                print(f"------------------Top-{top_n}------------------------")
                print(f"Correct: {correct}; Total: {len(dist_matrix)}")
                acc = correct * 100. / len(eval_labels)
                print("Accuracy(%): ", acc)
                print()
            
            # print per class accuracy
            for lbl in np.unique(train_labels):
                total = np.sum(eval_labels == lbl)
                correct = correct_preds_per_class[top_n][lbl]
                if verbose:
                    print(f"Class {lbl} has {correct} correct predictions out of {total} samples: Accuracy: {correct * 100. / total}")
        
        results[k] = {
            'predictions': preds,
            'correct_preds_idxs': correct_preds_idxs,
            'incorrect_preds_idxs': incorrect_preds_idxs,
            'accuracies': {top_n: correct_preds[top_n] * 100. / len(eval_labels) for top_n in top_ns}
        }
    
    return results


def compute_cluster_metrics(embeddings, labels):
    """Compute cluster quality metrics (Davies-Bouldin, Calinski-Harabasz)."""
    metrics = {}
    try:
        db_score = davies_bouldin_score(embeddings, labels)
        metrics['davies_bouldin_score'] = float(db_score)
        print(f"Davies-Bouldin Score: {db_score:.4f} (lower is better)")
    except Exception as e:
        print(f"Error computing Davies-Bouldin score: {e}")
        metrics['davies_bouldin_score'] = None
    
    try:
        ch_score = calinski_harabasz_score(embeddings, labels)
        metrics['calinski_harabasz_score'] = float(ch_score)
        print(f"Calinski-Harabasz Score: {ch_score:.4f} (higher is better)")
    except Exception as e:
        print(f"Error computing Calinski-Harabasz score: {e}")
        metrics['calinski_harabasz_score'] = None
    
    return metrics

def load_file_and_extract_clips(filepath, volume_id, start_time=3, end_time=19):
    """
    Load a single file and extract video clips for all z-slices.
    Returns: (video_clips, z_indices, global_indices, volume_ids)
    """
    try:
        full_image_data_np = np.load(filepath)
        current_data_tensor = torch.from_numpy(full_image_data_np)
        
        # Expected (T, C, Z, H, W)
        if current_data_tensor.shape[1] != 2:
            C, time_points, Z, H, W = current_data_tensor.shape
            current_data_tensor = current_data_tensor.permute(1, 0, 2, 3, 4)
        else:
            time_points, C, Z, H, W = current_data_tensor.shape
            
        time_points, num_channels_orig, z_slices_file, H, W = current_data_tensor.shape
        
        video_clips = []
        z_indices = []
        global_indices = []
        volume_ids = []
        
        # Extract each z-slice as a (T, C, H, W) video clip
        for z_idx_in_file in range(z_slices_file):
            video_clip_for_z = current_data_tensor[start_time:end_time, :, z_idx_in_file, :, :]
            video_clips.append(video_clip_for_z)
            z_indices.append(z_idx_in_file)
            # Global indices will be set by caller based on accumulated count
            global_indices.append(None)  # Placeholder
            volume_ids.append(volume_id)
        
        return {
            'success': True,
            'filepath': filepath,
            'video_clips': video_clips,
            'z_indices': z_indices,
            'volume_ids': volume_ids,
            'z_slices_file': z_slices_file
        }
    except FileNotFoundError:
        return {'success': False, 'filepath': filepath, 'error': 'File not found'}
    except Exception as e:
        return {'success': False, 'filepath': filepath, 'error': str(e)}

@torch.no_grad()
def validate_saved_model(model_path, device=None, 
                        visualize_umap=True, 
                        save_umap_path=None, 
                        concatenate_embeddings=True,
                        save_embeddings=True,
                        aggregate_method="mean",
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
    # 2 Load Data volume by volume (PARALLELIZED)

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
    # 3️⃣ Run validation per volume (PARALLELIZED with batching)
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

    if concatenate_embeddings:
        concatenated_embeds_mean = []
        concatenated_embeds_max = []
        concatenated_embeds_min = []
        for emb_v in all_embeds:
            concat_emb_mean = np.mean(emb_v, axis=0)
            concat_emb_max = np.max(emb_v, axis=0)
            concat_emb_min = np.min(emb_v, axis=0)
            concatenated_embeds_mean.append(concat_emb_mean)
            concatenated_embeds_max.append(concat_emb_max)
            concatenated_embeds_min.append(concat_emb_min)
            # concatenated_embeds.append(emb_v)
       
    else:
        concatenated_embeds_mean = all_embeds
        concatenated_embeds_max = all_embeds
        concatenated_embeds_min = all_embeds
    
    concatenated_embeds = all_embeds
    # Save embeddings
    if save_embeddings:
        filepaths = list(map(lambda x: os.path.basename(x).replace('.npy', ''), cfg.val_filepaths_2))
        drug_label_list = list(map(lambda x: os.path.split(x)[-2].split("/")[-1], cfg.val_filepaths_2))
        assert len(drug_label_list) == len(cfg.val_filepaths_2), "Number of drug labels should be equal to the number of filepaths"

        save_dirs = {
            "mean": os.path.join(cfg.save_path, "embeddings"),
            "max": os.path.join(cfg.save_path, "embeddings_max"),
            "min": os.path.join(cfg.save_path, "embeddings_min"),
            "all": os.path.join(cfg.save_path, "embeddings_all"),
        }
        for d in save_dirs.values():
            os.makedirs(d, exist_ok=True)

        # Save mean embeddings
        for i, filename in enumerate(filepaths):
            embed_save_path_mean = os.path.join(save_dirs["mean"], f"embeddings_{drug_label_list[i]}_{filename}.npy")
            np.save(embed_save_path_mean, concatenated_embeds_mean[i])
        print(f"✅ Saved mean embeddings to {save_dirs['mean']}")

        # Save max embeddings
        for i, filename in enumerate(filepaths):
            embed_save_path_max = os.path.join(save_dirs["max"], f"embeddings_{drug_label_list[i]}_{filename}.npy")
            np.save(embed_save_path_max, concatenated_embeds_max[i])
        print(f"✅ Saved max embeddings to {save_dirs['max']}")

        # Save min embeddings (optional, already in code)
        for i, filename in enumerate(filepaths):
            embed_save_path_min = os.path.join(save_dirs["min"], f"embeddings_{drug_label_list[i]}_{filename}.npy")
            np.save(embed_save_path_min, concatenated_embeds_min[i])
        print(f"✅ Saved min embeddings to {save_dirs['min']}")
        
        # Save concatenated embeddings
        for i, filename in enumerate(filepaths):
            embed_save_path = os.path.join(save_dirs["all"], f"embeddings_{drug_label_list[i]}_{filename}.npy")
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
        
    # INSERT_YOUR_CODE
    # Save validation metrics to a file in the save path
    metrics = {
        "model_path": model_path,
        "drug_labels": list(set(drug_label_list)),
        "volume_accuracy": {str(k): float(v) for k, v in per_volume_acc.items()},
        "mean_accuracy": float(mean_acc),
    }
    metrics_save_path = os.path.join(cfg.save_path, "validation_metrics.json")
    import json
    # Append new metrics to the file if it exists, else create and write
    if os.path.exists(metrics_save_path):
        with open(metrics_save_path, 'r') as f:
            try:
                existing_metrics = json.load(f)
            except Exception:
                existing_metrics = {}
        # Combine, appending as a new entry
        if not isinstance(existing_metrics, list):
            existing_metrics = [existing_metrics]
        existing_metrics.append(metrics)
        with open(metrics_save_path, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
    else:
        with open(metrics_save_path, 'w') as f:
            json.dump([metrics], f, indent=2)
    print(f"✅ Appended validation metrics to {metrics_save_path}")

    # ------------------------------------------------------------;
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


def compute_confusion_matrix_and_entropy_from_embeddings_folder( embeddings_dir, folder_to_label=None, folder_to_drug=None, label_drug_dict=None):
    """
    Given an embeddings folder, load all embeddings and their file names,
    construct the drug_label_list (or ground truth per embedding) from file names,
    and compute per-embedding kNN entropy in embedding space.
    """
    embedding_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".npy") and f.startswith("embeddings_20")])
    all_embeds = []
    drug_label_list = []
    drug_name_list = []
    embedding_filenames = []

    # INSERT_YOUR_REWRITE_HERE
    # Parallelized embedding loading and metadata extraction

    def process_embedding_file(fname):
        embed_path = os.path.join(embeddings_dir, fname)
        emb = np.load(embed_path, allow_pickle=True)
        # Example file name: embeddings_<folder>_<imgname>.npy
        parts = fname.split("_")
        if len(parts) >= 3:
            folder = parts[1]
        else:
            folder = None

        label = None
        drug_name = None
        if folder_to_label is not None and folder in folder_to_label:
            label = folder_to_label[folder]
        if folder_to_drug is not None and folder in folder_to_drug:
            drug_name = folder_to_drug[folder]

        # If no mapping, fallback to folder string label
        used_label = label if label is not None else folder
        used_drug = drug_name if drug_name is not None else folder
        return emb, used_label, used_drug, fname

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_embedding_file, embedding_files))

    # Unpack parallel results
    for emb, used_label, used_drug, fname in results:
        all_embeds.append(emb)
        drug_label_list.append(used_label)
        drug_name_list.append(used_drug)
        embedding_filenames.append(fname)

    X = np.stack(all_embeds)
    n_neighbors = min(50, len(X))

    neigh = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
    neigh.fit(X)

    all_neighbor_entropies = []
    all_neighbor_label_distributions = []

    for i, this_emb in enumerate(X):
        dists, indices = neigh.kneighbors([this_emb], n_neighbors=n_neighbors+1)
        neighbor_idxs = indices[0][1:]  # Exclude self
        neighbor_labels = [drug_label_list[j] for j in neighbor_idxs]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        prob_dist = counts / counts.sum()
        ent = entropy(prob_dist, base=2)
        all_neighbor_entropies.append(ent)
        label_dist = dict(zip(values, prob_dist))
        all_neighbor_label_distributions.append(label_dist)

    mean_knn_entropy = float(np.mean(all_neighbor_entropies))
    print(f"Embedding-space kNN (k={n_neighbors}) mean entropy: {mean_knn_entropy:.4f} bits")
    print(f"Per-embedding kNN entropy: {all_neighbor_entropies[:5]} ...")
    
    # Compute additional evaluation metrics similar to evaluate.py
    evaluation_metrics = {}
    
    # Normalize embeddings
    X_normalized = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # Convert labels to numeric if they're strings
    unique_labels = sorted(list(set(drug_label_list)))
    label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_numeric[label] for label in drug_label_list])
    
    # Compute cluster quality metrics
    if len(X) > 1 and len(unique_labels) > 1:
        print("\n Computing cluster quality metrics...")
        cluster_metrics = compute_cluster_metrics(X_normalized, numeric_labels)
        evaluation_metrics['cluster_quality'] = cluster_metrics
        
        # Split into train/eval sets (90/10split)
        split_idx = int(len(X) * 0.9)
        if split_idx > 0 and split_idx < len(X):
            train_embeddings = X_normalized[:split_idx]
            eval_embeddings = X_normalized[split_idx:]
            train_labels = numeric_labels[:split_idx]
            eval_labels = numeric_labels[split_idx:]
            
            # Nearest neighbor evaluation with cosine distance
            print("\n Computing nearest neighbor evaluation (cosine distance)...")
            top_ns = [1, 3, 5]
            num_neighbors = [10, 50, 100]
            
            dist_matrix, dist_matrix_idxs = cosine_distance(eval_embeddings, train_embeddings, weighted=False)
            nn_results_cosine = nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, 
                                                             dist_matrix, dist_matrix_idxs, 
                                                             num_neighbors=num_neighbors, verbose=True)
            evaluation_metrics['nearest_neighbor_cosine'] = {
                k: {
                    'accuracies': v['accuracies'],
                    'num_eval_samples': len(eval_labels),
                    'num_train_samples': len(train_labels)
                } for k, v in nn_results_cosine.items()
            }
            # Nearest neighbor evaluation with L2 distance
            print("\n Computing nearest neighbor evaluation (L2 distance)...")
            tree = KDTree(train_embeddings)
            nn_results_l2 = {}
            preds = {k: {top_n: [] for top_n in top_ns} for k in num_neighbors}

            for k in num_neighbors:
                dist, nearest_ind = tree.query(eval_embeddings, k=k)
                accuracies = {}

                for top_n in top_ns:
                    correct = 0
                    pred_labels = []
                    for i in range(len(eval_embeddings)):
                        eval_lbl = eval_labels[i]
                        k_nearest_nbs = train_labels[nearest_ind[i]]
                        top_most_freq_lbls = topKfrequent(k_nearest_nbs, None, top_n, weighted=False)
                        pred_labels.append(top_most_freq_lbls[0])  # Top-1 predicted label from nearest neighbors

                        if eval_lbl in top_most_freq_lbls:
                            correct += 1

                    preds[k][top_n] = pred_labels
                    acc = correct * 100. / len(eval_labels)
                    accuracies[top_n] = acc
                    print(f"Top-{top_n} Accuracy for {k} neighbors (L2): {acc:.2f}%")

                nn_results_l2[k] = {'accuracies': accuracies}
            
            evaluation_metrics['nearest_neighbor_l2'] = nn_results_l2
    print(len(eval_labels), len(preds))
    # label_drug_dict = {v: k for k,v in .items()}
    plot_cm_save_fig(eval_labels, preds[100][1], label_drug_dict=label_drug_dict, make_plot=True, embeddings_dir=embeddings_dir)
    # cm = confusion_matrix(eval_labels, preds[100][1], labels=) 
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.savefig(f"{embeddings_dir}/confusion_matrix.png")
    # plt.close()
    # print(f"Saved confusion matrix to {embeddings_dir}/confusion_matrix.png")
    
    return {
    #     "embedding_filenames": embedding_filenames,
    #     "drug_label_list": drug_label_list,
        "drug_name_list": drug_name_list,
        "knn_entropy_per_embedding": all_neighbor_entropies,
        "mean_knn_entropy": mean_knn_entropy,
        "knn_label_distributions": all_neighbor_label_distributions,
        "evaluation_metrics": evaluation_metrics
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
    
    cfg.save_path = "checkpoint_all_drugs_300"
    model_path = f"{cfg.save_path}/z_predictor_incremental.pth"
    aggregate_method = "max"
    # cfg.val_filepaths_2 = cfg.val_filepaths[0:]
    cfg.val_filepaths = []
    pick_folders = [
        # "20240729-1",#control
        # "20240805-1",#H2O2
        # "20240802-1",#tbhp
        # "20240814-1",#valinomycin
        # "20240911-1",#nigericin
        # "20240826-1",#nocodazole
        # "20240830-1",#colchicine
        # "20240816-1",#mitomycinC
        # "20240905-1",#cisplatin
        # "20240823-1",#mdivi1
        # Add more folder names or date-based identifiers as needed
    ]

    if len(pick_folders) == 0:
        pick_folders = [i for i in os.listdir(cfg.master_base_path) if os.path.isdir(os.path.join(cfg.master_base_path, i))]
        print(f"Picked folders: {pick_folders}")
    else:
        print(f"Picked folders: {pick_folders}")

    device = "cuda:0"
    visualize_umap = False
    save_embeddings = True
    concatenate_embeddings = True
    exp_name = cfg.save_path.split("_")[1]
    save_umap_path = f"umap_validation_{exp_name}"

    filtered_base_paths = [bp for bp in cfg.base_paths if any(bp.endswith(pick) for pick in pick_folders[11:])]
    train_split = 400
    # print("Picked folders:", filtered_base_paths)
    for base_path in filtered_base_paths:
        # print("Checking:", base_path)
        # print("Loading data from:", base_path)
        files = sorted(os.listdir(base_path))
        for i in files[train_split:]:
            cfg.val_filepaths.append(f"{base_path}/{i}")
    # cfg.val_filepaths_2 = cfg.val_filepaths[i:i+100]
    batch_size = 50

    print("debug train split", train_split)

    for folder in pick_folders:
        # first select the filepaths for the folder
        idxs = [i for i, fpath in enumerate(cfg.val_filepaths) if folder in fpath]
        if not idxs:
            print(f"⚠️ No filepaths found for folder: {folder}")
            continue
        print("debugging", len(idxs))
        # now select the filepaths for the batch
        for start in range(0, len(idxs), batch_size):
            selected_idxs = idxs[start:start+batch_size]
            if not selected_idxs:
                continue
            cfg.val_filepaths_2 = [cfg.val_filepaths[i] for i in selected_idxs]
            print("debugging", cfg.val_filepaths_2[0], cfg.val_filepaths_2[-1])
            print(f"Folder: {folder} | Batch size: {len(cfg.val_filepaths_2)}")
            validate_saved_model(model_path, 
                                device =device, 
                                visualize_umap = visualize_umap,
                                save_embeddings = save_embeddings,
                                save_umap_path = save_umap_path,
                                concatenate_embeddings = concatenate_embeddings,
                                aggregate_method = aggregate_method,
                                cfg = cfg)
    # validate_saved_model(model_path, 
    #                     device =device, 
    #                     visualize_umap = visualize_umap,
    #                     save_embeddings = save_embeddings,
    #                     save_umap_path = save_umap_path,
    #                     concatenate_embeddings = concatenate_embeddings,
    #                     cfg = cfg)