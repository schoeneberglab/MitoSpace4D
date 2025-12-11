import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

try:
    from .autoencoder_models_resnet import MitoSpace3DAutoencoder
    from .autoencoder_runner import AutoEncoderRunner
except ImportError:
    from autoencoder_models_resnet import MitoSpace3DAutoencoder
    from autoencoder_runner import AutoEncoderRunner

if __name__ == "__main__":
    print("Encoding data...")

    # Local paths
    # ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/autoencoder/runs/autoencoder/lightning_logs/version_0/checkpoints/epoch=9-step=5000.ckpt"
    # src_root  = "/mnt/aquila0/others/MitoSpace4D/data/aligned"      # Summer 2024
    # src_root = "/mnt/aquila0/ssd_processing/Others/MitoSpace4D/summer_2025_new"  # Summer 2025
    # dst_root  = "/home/earkfeld/Projects/MitoSpace4D/data"
    
    # Delta Paths
    ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1081149/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    src_root = "/work/nvme/begq/MitoSpace4D/data/2025_data/"
    dst_root = "/work/nvme/begq/MitoSpace4D/data/2025_data_encoded/"
    
    # Model parameters
    latent_dim = 6

    # Data normalization settings
    normalize_data = False
    max_value_tmrm = 25_000
    max_value_tracker = 10_000

    # Traverse source directory, mirror to destination, collect all npy files
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
                if osp.exists(outfile):
                    continue
                
                infiles.append(osp.join(src_dir_path, file))
                outfiles.append(osp.join(dst_dir_path, file))

    print(f"Found {len(infiles)} files to encode.")

    # Model setup
    # model = MitoSpace3DAutoencoder(latent_dim=latent_dim)
    # model = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
    # encoder = model.encoder
    model = MitoSpace3DAutoencoder()
    ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1081149/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    runner = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
    print("Loaded model from checkpoint.")

    encoder = runner.model.encoder
    
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    # Encode all the files!
    for infile, outfile in tqdm(zip(infiles, outfiles), total=len(infiles)):
        image = np.load(infile)  # (T,C,Z,Y,X)
        
        if normalize_data:
            image[:, 0] = np.clip(image[:, 0], 0, max_value_tmrm) / max_value_tmrm
            image[:, 1] = np.clip(image[:, 1], 0, max_value_tracker) / max_value_tracker

        # Swap to (C,T,Z,Y,X)
        # print("Normalized image shape:", image.shape)
        image = rearrange(image, "t c z y x -> c t z y x")
        # print("Rearranged image shape:", image.shape)

        with torch.no_grad():
            x = torch.from_numpy(image).unsqueeze(0).float().to(device)     # (B=1,T,C,Z,Y,X)
            z = encoder(x)                                                  # (1,T,latent_dim,D,H,W)
            encoded = z.squeeze(0).cpu().numpy()                            # (T,latent_dim,D,H,W)
            
        np.save(outfile, encoded)

        # print(f"Encoded {infile} -> {outfile}, shape: {encoded.shape}")
        # break  # TEMPORARY: only do one file for testing

