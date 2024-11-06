import os
import random

import numpy as np
from tqdm import tqdm

from autoencoder.models import MitoSpace3DAutoencoder
import torch
from torch import nn
import os.path as osp

if __name__ == '__main__':
    cpkt_path = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/runs/autoencoder/lightning_logs/version_3017473/checkpoints/epoch=5-step=51672.ckpt'
    ae_model = MitoSpace3DAutoencoder().to('cuda')
    decoder = ae_model.decoder
    encoder = ae_model.encoder
    params_dict = torch.load(cpkt_path)

    decoder_param_dict = {k.replace('model.decoder.', ''): v for k, v in params_dict['state_dict'].items() if 'model.decoder' in k}
    encoder_param_dict = {k.replace('model.encoder.', ''): v for k, v in params_dict['state_dict'].items() if 'model.encoder' in k}

    decoder.load_state_dict(decoder_param_dict)
    encoder.load_state_dict(encoder_param_dict)

    data_dir = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/2024_data/processed_data'
    enc_data_dir = '/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/data/2024_data/encoded_data'
    os.makedirs(enc_data_dir, exist_ok=True)

    drug_folders = os.listdir(data_dir)

    for drug in drug_folders:
        print(f"Processing {drug} ...")
        drug_folder_path = os.path.join(data_dir, drug)
        enc_drug_folder_path = os.path.join(enc_data_dir, drug)
        os.makedirs(enc_drug_folder_path, exist_ok=True)
        pbar = tqdm(os.listdir(drug_folder_path))

        bs = 8
        for i in range(0, len(os.listdir(drug_folder_path)), bs):
            batch = []
            for file in os.listdir(drug_folder_path)[i:i+bs]:
                img_path = osp.join(drug_folder_path, file)
                data = np.load(img_path)
                data = np.clip(data, 0, 20000)
                data = data / 20000
                data = data.astype(np.float32)
                data = torch.from_numpy(data).unsqueeze(0).to('cuda')
                batch.append(data)
            batch = torch.cat(batch, dim=0)

            enc = encoder(batch).detach().cpu().numpy()

            for j, file in enumerate(os.listdir(drug_folder_path)[i:i+bs]):
                np.save(osp.join(enc_drug_folder_path, file), enc[j])
                pbar.update(1)


