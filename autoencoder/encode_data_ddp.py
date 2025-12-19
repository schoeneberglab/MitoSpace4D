# import os
# import os.path as osp
# import numpy as np
# import torch
# from tqdm import tqdm

# from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
# from autoencoder.autoencoder_runner import AutoEncoderRunner
# from einops import rearrange

# if __name__ == "__main__":
#     print("Encoding data...")

#     # Local paths
#     # ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/autoencoder/runs/autoencoder/lightning_logs/version_0/checkpoints/epoch=9-step=5000.ckpt"
#     # src_root  = "/mnt/aquila0/others/MitoSpace4D/data/aligned"      # Summer 2024
#     # src_root = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/summer_2025_new"  # Summer 2025
#     # dst_root  = "/home/earkfeld/Projects/MitoSpace4D/data"
    
#     # Delta Paths
#     ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1081149/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
#     src_root = "/work/nvme/begq/MitoSpace4D/data/2025_data/"
#     dst_root = "/work/nvme/begq/MitoSpace4D/data/2025_data_encoded/"
    
#     # Model parameters
#     # latent_dim = 4
#     batch_size = 4

#     # Data normalization settings
#     normalize_data = False
#     max_value_tmrm = 25_000
#     max_value_tracker = 10_000

#     # Traverse source directory, mirror to destination, collect all npy files
#     infiles, outfiles = [], []
#     for src_dir in sorted(os.listdir(src_root)):
#         src_dir_path = osp.join(src_root, src_dir)
#         if not osp.isdir(src_dir_path):
#             continue
#         dst_dir_path = osp.join(dst_root, src_dir)
#         os.makedirs(dst_dir_path, exist_ok=True)
#         for file in sorted(os.listdir(src_dir_path)):
#             if file.endswith(".npy"):
#                 infiles.append(osp.join(src_dir_path, file))
#                 outfiles.append(osp.join(dst_dir_path, file))

#     # ---- Shard files across Slurm tasks (one GPU per task) ----
#     # Works for single-node or multi-node srun: set ntasks = number of GPUs
#     rank = int(os.environ.get("SLURM_PROCID", 0))
#     world_size = int(os.environ.get("SLURM_NTASKS", 1))

#     pairs_all = list(zip(infiles, outfiles))
#     pairs_shard = pairs_all[rank::world_size]
#     if pairs_shard:
#         infiles, outfiles = zip(*pairs_shard)
#     else:
#         infiles, outfiles = [], []

#     print(f"[rank {rank}/{world_size}] Processing {len(infiles)} files.")

#     # Model setup
#     model = MitoSpace3DAutoencoder()
#     runner = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
#     encoder = runner.model.encoder

#     encoder.eval()
#     for p in encoder.parameters():
#         p.requires_grad = False

#     # With srun --gpus-per-task=1, CUDA_VISIBLE_DEVICES is set per task, so cuda:0 is correct
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     encoder.to(device)

#     # Encode all the files assigned to this rank
#     for infile, outfile in tqdm(list(zip(infiles, outfiles)), total=len(infiles), desc=f"rank {rank}", leave=False):
#         image = np.load(infile)  # (T,C,Z,Y,X)
        
#         if normalize_data:
#             # assume channels [0]=TMRM, [1]=tracker
#             image[:, 0] = np.clip(image[:, 0], 0, max_value_tmrm) / max_value_tmrm
#             image[:, 1] = np.clip(image[:, 1], 0, max_value_tracker) / max_value_tracker

#         # Swap to (C,T,Z,Y,X)
#         image = rearrange(image, "t c z y x -> c t z y x")

#         with torch.no_grad():
#             x = torch.from_numpy(image).unsqueeze(0).float().to(device)     # (B=1,T,C,Z,Y,X)
#             z = encoder(x)                                                  # (1,T,latent_dim,D,H,W)
#             encoded = z.squeeze(0).cpu().numpy()                            # (T,latent_dim,D,H,W)
            
#         np.save(outfile, encoded)

import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import einops

from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
from autoencoder.autoencoder_runner import AutoEncoderRunner

def chunk_pairs(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]

if __name__ == "__main__":
    print("Encoding data...")

    # ---- Paths ----
    ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1081149/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    src_root  = "/work/nvme/begq/MitoSpace4D/data/2025_data/"
    dst_root  = "/work/nvme/begq/MitoSpace4D/data/2025_data_encoded/"

    # ---- Parameters ----
    batch_size = 4
    normalize_data = False
    max_value_tmrm = 25_000
    max_value_tracker = 10_000

    # ---- Collect files & mirror directory structure ----
    infiles, outfiles = [], []
    for src_dir in sorted(os.listdir(src_root)):
        src_dir_path = osp.join(src_root, src_dir)
        if not osp.isdir(src_dir_path):
            continue
        dst_dir_path = osp.join(dst_root, src_dir)
        os.makedirs(dst_dir_path, exist_ok=True)
        for file in sorted(os.listdir(src_dir_path)):
            if file.endswith(".npy"):
                infile = osp.join(src_dir_path, file)
                outfile = osp.join(dst_dir_path, file)
                
                # Check if the outfile exists already; skip if some
                if osp.exists(outfile):
                    continue

                infiles.append(infile)
                outfiles.append(outfile)

    # ---- Shard across SLURM ranks (one GPU per task) ----
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    pairs_all = list(zip(infiles, outfiles))
    pairs_shard = pairs_all[rank::world_size]
    if pairs_shard:
        infiles, outfiles = zip(*pairs_shard)
        infiles, outfiles = list(infiles), list(outfiles)
    else:
        infiles, outfiles = [], []

    print(f"[rank {rank}/{world_size}] Processing {len(infiles)} files.")

    # ---- Model setup ----
    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
    encoder = runner.model.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    # ---- Single progress bar per rank (rank 0 only) ----
    disable_pbar = (rank != 0)
    pbar = tqdm(total=len(infiles), desc=f"rank {rank}", disable=disable_pbar)

    # ---- Batched encoding ----
    with torch.inference_mode():
        for batch in chunk_pairs(list(zip(infiles, outfiles)), batch_size):
            # Load and (optionally) normalize; expect (T, C, Z, Y, X)
            np_batch = []
            out_paths = []
            for infile, outfile in batch:
                arr = np.load(infile)  # (T, C, Z, Y, X)
                if normalize_data:
                    # assume channels [0]=TMRM, [1]=tracker
                    arr[..., 0, :, :, :]  # just to make intent explicit in comments
                    arr[:, 0] = np.clip(arr[:, 0], 0, max_value_tmrm) / max_value_tmrm
                    arr[:, 1] = np.clip(arr[:, 1], 0, max_value_tracker) / max_value_tracker
                
                arr = einops.rearrange(arr, "t c z y x -> c t z y x")

                np_batch.append(arr)   # keep (T, C, Z, Y, X)
                out_paths.append(outfile)

            # Sanity: ensure identical shapes for stacking
            shapes = {tuple(a.shape) for a in np_batch}
            if len(shapes) != 1:
                raise ValueError(f"Batched files have differing shapes: {shapes}")

            # Stack to (B, T, C, Z, Y, X) and move to device
            x = torch.from_numpy(np.stack(np_batch, axis=0)).float().to(device)  # (B,T,C,Z,Y,X)

            # Encode; expecting output (B, T, latent_dim, D, H, W)
            z = encoder(x)

            # Save each item in the batch
            z_cpu = z.cpu().numpy()
            for i, out_path in enumerate(out_paths):
                os.makedirs(osp.dirname(out_path), exist_ok=True)
                np.save(out_path, z_cpu[i])  # (T, latent_dim, D, H, W)

            # Update progress once per batch by batch size (or remaining items)
            pbar.update(len(batch))

    pbar.close()