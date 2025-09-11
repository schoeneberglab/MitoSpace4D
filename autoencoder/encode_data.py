import os
import os.path as osp
import numpy as np
import torch

from autoencoder.autoencoder_models import MitoSpace3DAutoencoder
from autoencoder.autoencoder_runner import AutoEncoderRunner

if __name__ == "__main__":
    print("Encoding data...")

    # Hardcoded checkpoint file (must be a .ckpt file, not just the folder)
    ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/autoencoder/runs/autoencoder/lightning_logs/version_0/checkpoints/epoch=9-step=5000.ckpt"

    src_root  = "/mnt/aquila0/others/MitoSpace4D/data/aligned"      # Summer 2024
    # src_root = "/mnt/aquila0/ssd_processing/Others/MitoSpace4D/summer_2025_new"  # Summer 2025
    dst_root  = "/home/earkfeld/Projects/MitoSpace4D/data"

    infiles, outfiles = [], []
    for src_dir in sorted(os.listdir(src_root)):
        src_dir_path = osp.join(src_root, src_dir)
        if not osp.isdir(src_dir_path):
            continue
        dst_dir_path = osp.join(dst_root, src_dir)
        os.makedirs(dst_dir_path, exist_ok=True)
        for file in sorted(os.listdir(src_dir_path)):
            if file.endswith(".npy"):
                infiles.append(osp.join(src_dir_path, file))
                outfiles.append(osp.join(dst_dir_path, file))

    print(f"Found {len(infiles)} files")

    # Initialize the model from the checkpoint
    model = MitoSpace3DAutoencoder()
    model = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
    encoder = model.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    max_value_tmrm = 25_000
    max_value_tracker = 10_000

    for infile, outfile in zip(infiles, outfiles):
        image = np.load(infile)  # (C,Z,Y,X) or (T,C,Z,Y,X)
        image[:, 0] = np.clip(image[:, 0], 0, max_value_tmrm) / max_value_tmrm
        image[:, 1] = np.clip(image[:, 1], 0, max_value_tracker) / max_value_tracker

        with torch.no_grad():
            if image.ndim == 5:
                # Encode per timepoint, stack
                enc_list = []
                for t in range(image.shape[0]):
                    vol = torch.from_numpy(image[t]).float().unsqueeze(0).to(device)  # (1,C,Z,Y,X)
                    code = encoder(vol).detach().cpu().numpy()
                    enc_list.append(code)
                encoded = np.concatenate(enc_list, axis=0)
            else:
                vol = torch.from_numpy(image).float().unsqueeze(0).to(device)  # (1,C,Z,Y,X)
                encoded = encoder(vol).detach().cpu().numpy()

        np.save(outfile, encoded)
