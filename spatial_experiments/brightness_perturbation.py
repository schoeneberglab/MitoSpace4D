import numpy as np
import os
import os.path as osp
from simclr.models_simple import Lightweight3DResNet
from utils.utils import load_config
from train_simclr import SimCLRRunner
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Utilities -----------------
def get_embeddings(model, video_tensor):
    """
    Returns per-timestep embeddings (b,t,d)
    video_tensor: (1, T, C, D, H, W)
    """
    model.eval()
    with torch.no_grad():
        x, out, lstm_hidden = model(video_tensor.to(device))
    return lstm_hidden.squeeze(0).cpu()  # (T, D)

def cosine_per_timestep(a, b):
    """
    a, b: (T,D)
    returns: (T,) cosine similarity per timestep
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def perturb_shuffle(video_tensor):
    """
    video_tensor: (T,C,D,H,W)
    returns shuffled video and permutation used
    """
    T = video_tensor.shape[0]
    perm = np.random.permutation(T)
    shuffled = video_tensor[perm]
    return shuffled, perm

# ----------------- Experiment -----------------
def brightness_sensitivity_experiment(model, video, plot=False):
    video_cpu = video.cpu() if video.is_cuda else video
    perturb_video = video_cpu.clone()
    d_brightness_tmrm = 0.3 * video[:, 0].std()
    d_brightness_mtg = 0.3 * video[:, 1].std()
    # perturb_video[:, 0] = perturb_video[:, 0] + d_brightness_tmrm  # changing TMRM only
    # perturb_video[:, 1] = perturb_video[:, 1] + d_brightness_mtg  # changing MTG only
    perturb_video[:, 1] = perturb_video[:, 1] - d_brightness_mtg  # subtracting from MTG only
    perturb_video[:, 0] = perturb_video[:, 0] - d_brightness_tmrm  # subtracting from TMRM only


    perturb_video = perturb_video.float().unsqueeze(0)
    video_batch = video_cpu.unsqueeze(0).float()

    emb_orig = get_embeddings(model, video_batch)          # (T,D)
    emb_shuffled = get_embeddings(model, perturb_video)  # (T,D)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cos_per_frame = cosine_per_timestep(emb_orig, emb_shuffled)

    if plot:
        plt.figure(figsize=(8,3))
        plt.plot(cos_per_frame.numpy(), marker='o')
        plt.title("Cosine similarity per timestep (original vs shuffled)")
        plt.xlabel("Frame index")
        plt.ylabel("Cosine similarity")
        plt.ylim(0,1)
        plt.grid(True)
        plt.show()

    print(f"Mean cosine similarity: {cos_per_frame.mean().item():.4f}")
    print(f"Min cosine similarity: {cos_per_frame.min().item():.4f}")
    print(f"Max cosine similarity: {cos_per_frame.max().item():.4f}")

    return cos_per_frame

if __name__ == '__main__':
    cfg = load_config('/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml')
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    # data_root = '/mnt/aquila/others/MitoSpace4D/2025_summer_new'
    data_root = '/mnt/aquila/others/MitoSpace4D/data/aligned/'
    device = 'cuda'

    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=False).to(device)

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/resnetbilistm_encoder_consistent_temporal/checkpoints/epoch=278-step=59985-val_loss=0.00.ckpt"
    dataset_name = cfg["evaluate"]["dataset"]

    # print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )
    model.eval()

    k = 10  # number of samples per drug to test

    drug_labels = {}
    with open('/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt', 'r') as f:
        drugs_to_labels = f.readlines()
        for line in drugs_to_labels:
            folder, drug, label = line.split()
            drug_labels[folder] = {'drug': drug, 'label': int(label)}
    drug_folders = list(drug_labels.keys())[:26]
    # drug_folders = [x for x in os.listdir(data_root) if osp.isdir(osp.join(data_root, x))]

    drug_cosine_map = {}
    for drug_folder in drug_folders:
        folder_path = osp.join(data_root, drug_folder)
        filenames = sorted([file for file in os.listdir(folder_path) if osp.isfile(osp.join(folder_path, file))])
        num_samples = len(filenames)
        print(f"Drug: {drug_folder}, Label: {drug_labels[drug_folder]['label']}, Number of samples: {num_samples}")

        # select k random samples to perturb
        selected_indices = np.random.choice(num_samples, size=min(k, num_samples), replace=False)
        drug_cosine_map[drug_folder] = []
        for idx in selected_indices:
            file_path = osp.join(folder_path, filenames[idx])
            data = np.load(file_path).astype(np.float32)  # (t, c, z, y, x) for old v1 data
            data[:, 0] = np.clip(data[:, 0], 0, 25000) / 25000.
            data[:, 1] = np.clip(data[:, 1], 0, 10000) / 10000.
            data = torch.tensor(data)  # Keep on CPU to avoid GPU memory issues

            # T = data.shape[0]
            # permutation = np.random.permutation(T)
            # perturbed_data = data[permutation]
            #
            # with torch.no_grad():
            #     emb, _ = model.model(torch.tensor(data[None, ...], dtype=torch.float32).to(device)) # (1, t, emb_dim)
            #     perturbed_emb, _ = model.model(torch.tensor(perturbed_data[None, ...], dtype=torch.float32).to(device)) # (1, t, emb_dim)
            #
            # emb_t = emb[0]  # shape (T, D)
            # pert_emb_t = perturbed_emb[0]  # shape (T, D)
            #
            # # sort perturbed back to original order
            # pert_back = pert_emb_t[permutation.argsort()]
            #
            # emb_t = F.normalize(emb_t, dim=-1)
            # pert_back = F.normalize(pert_back, dim=-1)
            #
            # cosine_dist = (emb_t * pert_back).sum(-1)
            #
            # print(f"Sample: {filenames[idx]}, Cosine similarity between original and perturbed embedding: {cosine_dist.mean()}")

            cosine_per_frame = brightness_sensitivity_experiment(model.model, data)
            drug_cosine_map[drug_folder].append(cosine_per_frame.cpu().numpy())
            
            # Clear GPU cache to free up memory between samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    drug_means = {}
    all_drugs = []
    all_vectors = []

    for drug, vectors in drug_cosine_map.items():
        # stack: (num_samples, T)
        arr = np.stack(vectors, axis=0)
        mean_vec = arr.mean(axis=0)  # (T,)
        drug_means[drug] = mean_vec

        all_drugs.append(drug)
        all_vectors.append(mean_vec)

    # convert to matrix for heatmap: (num_drugs, T)
    drug_matrix = np.stack(all_vectors, axis=0)

    # ---------------------------------------------------------
    # Plot Heatmap
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))

    ax = sns.heatmap(
        drug_matrix,
        xticklabels=True,
        yticklabels=all_drugs,
        cmap="RdBu_r",  # red <-> blue diverging
        vmin=0,
        vmax=1,
        annot=True,  # show numbers
        fmt=".2f",  # 2 decimal places
        annot_kws={"size": 6},  # small enough to fit in cells
        cbar=True
    )

    plt.title("Drug-Level Temporal Shuffling Sensitivity (Mean Cosine Similarity)", fontsize=14)
    plt.xlabel("Frame index", fontsize=12)
    plt.ylabel("Drug", fontsize=12)

    # Rotate x-labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\n==== Drug-Level Temporal Sensitivity Stats ====\n")
    for drug, mean_vec in drug_means.items():
        print(f"Drug: {drug:<25} "
              f"Mean={mean_vec.mean():.4f}  "
              f"Min={mean_vec.min():.4f}  "
              f"Max={mean_vec.max():.4f}  "
              f"Var={mean_vec.var():.4f}")
