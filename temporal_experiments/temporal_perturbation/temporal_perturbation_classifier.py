import numpy as np
import os
import os.path as osp
import hashlib
import random
from functools import partial
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


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_video(video_fpath, base_seed=42):
    data = np.load(video_fpath).astype(np.float32)
    data[:, 0] = np.clip(data[:, 0], 0, 25000) / 25000.
    data[:, 1] = np.clip(data[:, 1], 0, 10000) / 10000.
    video_tensor = torch.from_numpy(data)

    T = video_tensor.shape[0]
    # Use video path to create deterministic but unique seed per video
    # Use hashlib for deterministic hashing across Python runs
    video_hash = int(hashlib.md5(video_fpath.encode()).hexdigest()[:8], 16)
    video_seed = (video_hash % (2**31)) + base_seed
    generator = torch.Generator()
    generator.manual_seed(video_seed)
    perm = torch.randperm(T, generator=generator)

    return video_tensor, perm

# ----------------- Dataset -----------------
class ShuffleProbeDataset(Dataset):
    def __init__(self, videos, model, per_timestep=True, workers=16, cache_dir=None, seed=42):
        """
        videos: list of file paths to videos (T,C,D,H,W)
        model: your trained model
        per_timestep: if True, return embeddings per frame
        cache_dir: directory to save/load embeddings cache. If None, no caching.
        seed: random seed for reproducibility
        """
        # Create cache file path if cache_dir is provided
        cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            # Create a hash based on video paths, per_timestep setting, and seed
            video_str = '\n'.join(sorted(videos)) + f'_per_timestep_{per_timestep}_seed_{seed}'
            cache_hash = hashlib.md5(video_str.encode()).hexdigest()
            cache_path = osp.join(cache_dir, f'embeddings_{cache_hash}.pt')
        
        # Try to load from cache
        if cache_path is not None and osp.exists(cache_path):
            print(f"Loading embeddings from cache: {cache_path}")
            cache_data = torch.load(cache_path)
            self.embeddings = cache_data['embeddings']
            self.labels = cache_data['labels']
            print(f"Loaded {len(self.labels)} embeddings from cache")
            return
        
        # Compute embeddings if not cached
        self.embeddings = []
        self.labels = []

        # Create a partial function to pass seed to preprocess_video
        preprocess_func = partial(preprocess_video, base_seed=seed)
        
        with Pool(workers) as pool:
            pbar = tqdm(pool.imap(preprocess_func, videos),
                        total=len(videos),
                        desc="Preparing Shuffle Probe Dataset")

            for video_tensor, perm in pbar:
                # ---- GPU forward pass happens here in master process ----
                video_tensor = video_tensor.unsqueeze(0).to(device)
                shuf_tensor = video_tensor[:, perm].to(device)

                with torch.no_grad():
                    _, _, orig_emb = model(video_tensor)
                    _, _, shuf_emb = model(shuf_tensor)

                orig_emb = orig_emb.squeeze(0).cpu()
                shuf_emb = shuf_emb.squeeze(0).cpu()

                if per_timestep:
                    last_orig = orig_emb[9]
                    # unshuff = shuf_emb[torch.argsort(perm)]
                    last_unshuff = shuf_emb[9]

                    self.embeddings.append(last_orig.unsqueeze(0))
                    self.labels.append(torch.zeros(1))
                    self.embeddings.append(last_unshuff.unsqueeze(0))
                    self.labels.append(torch.ones(1))

                else:
                    self.embeddings.append(orig_emb.mean(0, keepdim=True))
                    self.labels.append(torch.zeros(1))
                    self.embeddings.append(shuf_emb.mean(0, keepdim=True))
                    self.labels.append(torch.ones(1))

        # ---- combine everything ----
        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.labels = torch.cat(self.labels, dim=0).long()
        
        # Save to cache if cache_dir is provided
        if cache_path is not None:
            print(f"Saving embeddings to cache: {cache_path}")
            torch.save({
                'embeddings': self.embeddings,
                'labels': self.labels
            }, cache_path)
            print(f"Saved {len(self.labels)} embeddings to cache")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ----------------- Linear Probe -----------------
class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

# ----------------- Training Function -----------------
def train_probe(model, videos, epochs=20, batch_size=64, per_timestep=True, lr=1e-3, cache_dir=None, seed=42):
    # Set seed for DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataset = ShuffleProbeDataset(videos, model, per_timestep, cache_dir=cache_dir, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    input_dim = dataset.embeddings.shape[1]
    probe = LinearProbe(input_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        correct = 0
        total = 0
        for x_batch, y_batch in loader:
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
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}, Acc={acc:.4f}")

    return probe, dataset

def setup(seed=42):
    # Set seed for reproducibility in setup
    set_seed(seed)
    
    cfg = load_config('/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml')
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    data_root = '/mnt/aquila/others/MitoSpace4D/data/aligned/'
    # data_root = '/mnt/aquila/others/MitoSpace4D/2025_summer_new'
    device = 'cuda'
    samples = []

    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=False).to(device)

    # checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/epoch=287-step=83534-val_loss=0.00.ckpt"
    checkpoint_path = f"{proj_dir}/runs/lightning_logs/resnetbilistm_encoder_consistent_temporal/checkpoints/epoch=278-step=59985-val_loss=0.00.ckpt"
    dataset_name = cfg["evaluate"]["dataset"]

    # print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )
    model.eval()

    k = 100

    # drug_labels = {}
    # with open('/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt', 'r') as f:
    #     drugs_to_labels = f.readlines()
    #     for line in drugs_to_labels:
    #         folder, drug, label = line.split()
    #         drug_labels[folder] = {'drug': drug, 'label': int(label)}
    drug_folders = sorted([x for x in os.listdir(data_root) if osp.isdir(osp.join(data_root, x))])
    # drug_folders = list(drug_labels.keys())

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
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    
    model, samples = setup(seed=seed)
    
    # Set cache directory - embeddings will be saved/loaded here
    cache_dir = osp.join(osp.dirname(__file__), 'embeddings_cache')

    probe, dataset = train_probe(model, samples, epochs=100, batch_size=256, cache_dir=cache_dir, seed=seed)

    with torch.no_grad():
        embeddings = dataset.embeddings.to(device)
        labels = dataset.labels.cpu().numpy()
        logits = probe(embeddings)
        probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()

    plt.figure(figsize=(6,4))
    plt.hist(probs[labels==0], bins=20, alpha=0.5, label="original")
    plt.hist(probs[labels==1], bins=20, alpha=0.5, label="shuffled")
    plt.xlabel("Predicted prob of shuffled")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Linear probe distribution")
    plt.show()
