import numpy as np
import os
import os.path as osp
from simclr.models_simple import Lightweight3DResNet
from utils.utils import load_config
from train_simclr import SimCLRRunner
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from multiprocessing import Pool

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_video(video_fpath):
    data = np.load(video_fpath).astype(np.float32)
    data[:, 0] = np.clip(data[:, 0], 0, 25000) / 25000.
    data[:, 1] = np.clip(data[:, 1], 0, 10000) / 10000.
    video_tensor = torch.from_numpy(data)  # (T,C,D,H,W)  # This is for kinetics data

    return video_tensor

# ----------------- Dataset -----------------
class FrameIndexProbeDataset(Dataset):
    def __init__(self, videos, model, workers=16, shuffle_frames=False, num_classes=None):
        """
        videos: list of file paths to videos (T,C,D,H,W)
        model: your trained model
        shuffle_frames: if True, randomly shuffle frame order before extracting embeddings
        num_classes: number of classes (if None, will be determined from max frames)
        Creates embeddings per frame and labels them with their frame index (as class)
        """
        self.embeddings = []
        self.labels = []
        max_frames = 0

        # First pass: find maximum number of frames (if num_classes not provided)
        if num_classes is None:
            with Pool(workers) as pool:
                pbar = tqdm(pool.imap(preprocess_video, videos),
                            total=len(videos),
                            desc="Finding max frames")
                max_frames = 20
            self.num_classes = max_frames
        else:
            self.num_classes = num_classes

        print(f"Using {self.num_classes} classes for {'shuffled' if shuffle_frames else 'normal'} dataset")

        # Second pass: extract embeddings and create labels
        with Pool(workers) as pool:
            pbar = tqdm(pool.imap(preprocess_video, videos),
                        total=len(videos),
                        desc=f"Preparing {'Shuffled' if shuffle_frames else 'Normal'} Frame Index Probe Dataset")

            for video_tensor in pbar:
                # ---- GPU forward pass happens here in master process ----
                T = video_tensor.shape[0]
                
                # Shuffle frames if requested
                if shuffle_frames:
                    perm = torch.randperm(T)
                    video_tensor = video_tensor[perm]
                    # Labels remain in original order (we want to predict original frame index)
                    frame_indices = torch.arange(T, dtype=torch.long)
                else:
                    frame_indices = torch.arange(T, dtype=torch.long)
                
                video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, D, H, W)

                with torch.no_grad():
                    _, _, emb = model(video_tensor)  # emb: (1, T, embedding_dim)

                emb = emb.squeeze(0).cpu()  # (T, embedding_dim)
                
                self.embeddings.append(emb)
                self.labels.append(frame_indices)

        # ---- combine everything ----
        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ----------------- Linear Probe -----------------
class FrameIndexProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Classification: output logits for each frame index class

    def forward(self, x):
        x = self.fc(x)
        return x

# ----------------- Evaluation Function -----------------
def evaluate_probe(probe, dataset, dataset_name="Validation"):
    """Evaluate probe on a dataset"""
    probe.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = probe(x_batch)
            loss = criterion(logits, y_batch)
            
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    avg_loss = total_loss / total_samples
    
    return accuracy, avg_loss, all_preds, all_labels

# ----------------- Training Function -----------------
def train_probe(model, train_videos, val_videos_normal, val_videos_shuffled, epochs=20, batch_size=64, lr=1e-3):
    # Create datasets
    train_dataset = FrameIndexProbeDataset(train_videos, model, shuffle_frames=False)
    val_dataset_normal = FrameIndexProbeDataset(val_videos_normal, model, shuffle_frames=False, 
                                                num_classes=train_dataset.num_classes)
    val_dataset_shuffled = FrameIndexProbeDataset(val_videos_shuffled, model, shuffle_frames=True,
                                                   num_classes=train_dataset.num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = train_dataset.embeddings.shape[1]
    num_classes = train_dataset.num_classes
    probe = FrameIndexProbe(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # Classification loss

    for epoch in range(epochs):
        # Training
        probe.train()
        correct = 0
        total = 0
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = probe(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            total_loss += loss.item() * y_batch.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Validation
        val_acc_normal, val_loss_normal, _, _ = evaluate_probe(probe, val_dataset_normal, "Normal Val")
        val_acc_shuffled, val_loss_shuffled, _, _ = evaluate_probe(probe, val_dataset_shuffled, "Shuffled Val")
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss={train_loss:.6f}, Train Acc={train_acc:.4f}, "
              f"Val Normal Acc={val_acc_normal:.4f}, Val Shuffled Acc={val_acc_shuffled:.4f}")

    return probe, train_dataset, val_dataset_normal, val_dataset_shuffled

def setup():
    cfg = load_config('/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml')
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    data_root = '/mnt/aquila/others/MitoSpace4D/data/aligned/'
    # data_root = '/mnt/aquila/others/MitoSpace4D/2025_summer_new'
    device = 'cuda'
    samples = []

    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=False).to(device)

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=278-step=59985-val_loss=0.00.ckpt"
    dataset_name = cfg["evaluate"]["dataset"]

    # print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )
    model.eval()

    k = 150

    drug_labels = {}
    with open('/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt', 'r') as f:
        drugs_to_labels = f.readlines()
        for line in drugs_to_labels:
            folder, drug, label = line.split()
            drug_labels[folder] = {'drug': drug, 'label': int(label)}
    # drug_folders = sorted([x for x in os.listdir(data_root) if osp.isdir(osp.join(data_root, x))])
    drug_folders = list(drug_labels.keys())

    for drug_folder in drug_folders:
        folder_path = osp.join(data_root, drug_folder)
        filenames = sorted([file for file in os.listdir(folder_path) if osp.isfile(osp.join(folder_path, file))])
        num_samples = len(filenames)
        # print(f"Drug: {drug_folder}, Label: {drug_labels[drug_folder]['label']}, Number of samples: {num_samples}")

        # select k random samples to perturb
        selected_indices = np.random.choice(num_samples, size=min(k, num_samples), replace=False)
        for idx in selected_indices:
            file_path = osp.join(folder_path, filenames[idx])
            samples.append(file_path)

    return model.model, samples

if __name__ == "__main__":
    model, samples = setup()

    # Split into train and validation sets (80/20 split)
    np.random.seed(42)
    torch.manual_seed(42)
    indices = np.random.permutation(len(samples))
    split_idx = int(0.8 * len(samples))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_videos = [samples[i] for i in train_indices]
    val_videos = [samples[i] for i in val_indices]
    
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    probe, train_dataset, val_dataset_normal, val_dataset_shuffled = train_probe(
        model, train_videos, val_videos, val_videos, epochs=200, batch_size=64
    )

    # Final evaluation on both validation sets
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    
    val_acc_normal, val_loss_normal, preds_normal, labels_normal = evaluate_probe(
        probe, val_dataset_normal, "Normal Validation"
    )
    val_acc_shuffled, val_loss_shuffled, preds_shuffled, labels_shuffled = evaluate_probe(
        probe, val_dataset_shuffled, "Shuffled Validation"
    )
    
    print(f"\nNormal Validation Set:")
    print(f"  Accuracy: {val_acc_normal:.4f}")
    print(f"  Loss: {val_loss_normal:.6f}")
    
    print(f"\nShuffled Validation Set:")
    print(f"  Accuracy: {val_acc_shuffled:.4f}")
    print(f"  Loss: {val_loss_shuffled:.6f}")

    # Plot results for both validation sets
    fig = plt.figure(figsize=(15, 10))
    
    # Normal validation set plots
    logits_normal = probe(val_dataset_normal.embeddings.to(device))
    probs_normal = F.softmax(logits_normal, dim=1).cpu().numpy()
    
    plt.subplot(2, 3, 1)
    plt.scatter(labels_normal, preds_normal, alpha=0.3, s=1)
    plt.plot([labels_normal.min(), labels_normal.max()], 
             [labels_normal.min(), labels_normal.max()], 'r--', label='Perfect prediction')
    plt.xlabel("True Frame Index")
    plt.ylabel("Predicted Frame Index")
    plt.title(f"Normal Val: Frame Index Classification (Acc={val_acc_normal:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    correct_mask_normal = preds_normal == labels_normal
    if correct_mask_normal.sum() > 0:
        correct_confidences_normal = probs_normal[correct_mask_normal].max(axis=1)
        incorrect_confidences_normal = probs_normal[~correct_mask_normal].max(axis=1) if (~correct_mask_normal).sum() > 0 else np.array([])
        
        if len(correct_confidences_normal) > 0:
            plt.hist(correct_confidences_normal, bins=30, alpha=0.7, label='Correct', density=True)
        if len(incorrect_confidences_normal) > 0:
            plt.hist(incorrect_confidences_normal, bins=30, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Density")
    plt.title("Normal Val: Confidence Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    unique_labels_normal = np.unique(labels_normal)
    per_class_acc_normal = []
    for label in unique_labels_normal:
        mask = labels_normal == label
        if mask.sum() > 0:
            per_class_acc_normal.append(np.mean(preds_normal[mask] == labels_normal[mask]))
    plt.bar(unique_labels_normal, per_class_acc_normal, alpha=0.7)
    plt.xlabel("Frame Index")
    plt.ylabel("Accuracy")
    plt.title("Normal Val: Per-Class Accuracy")
    plt.grid(True, alpha=0.3)

    # Shuffled validation set plots
    logits_shuffled = probe(val_dataset_shuffled.embeddings.to(device))
    probs_shuffled = F.softmax(logits_shuffled, dim=1).cpu().numpy()
    
    plt.subplot(2, 3, 4)
    plt.scatter(labels_shuffled, preds_shuffled, alpha=0.3, s=1)
    plt.plot([labels_shuffled.min(), labels_shuffled.max()], 
             [labels_shuffled.min(), labels_shuffled.max()], 'r--', label='Perfect prediction')
    plt.xlabel("True Frame Index")
    plt.ylabel("Predicted Frame Index")
    plt.title(f"Shuffled Val: Frame Index Classification (Acc={val_acc_shuffled:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    correct_mask_shuffled = preds_shuffled == labels_shuffled
    if correct_mask_shuffled.sum() > 0:
        correct_confidences_shuffled = probs_shuffled[correct_mask_shuffled].max(axis=1)
        incorrect_confidences_shuffled = probs_shuffled[~correct_mask_shuffled].max(axis=1) if (~correct_mask_shuffled).sum() > 0 else np.array([])
        
        if len(correct_confidences_shuffled) > 0:
            plt.hist(correct_confidences_shuffled, bins=30, alpha=0.7, label='Correct', density=True)
        if len(incorrect_confidences_shuffled) > 0:
            plt.hist(incorrect_confidences_shuffled, bins=30, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Density")
    plt.title("Shuffled Val: Confidence Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    unique_labels_shuffled = np.unique(labels_shuffled)
    per_class_acc_shuffled = []
    for label in unique_labels_shuffled:
        mask = labels_shuffled == label
        if mask.sum() > 0:
            per_class_acc_shuffled.append(np.mean(preds_shuffled[mask] == labels_shuffled[mask]))
    plt.bar(unique_labels_shuffled, per_class_acc_shuffled, alpha=0.7)
    plt.xlabel("Frame Index")
    plt.ylabel("Accuracy")
    plt.title("Shuffled Val: Per-Class Accuracy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
