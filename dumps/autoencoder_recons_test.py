import os
import random

import numpy as np
import napari

from autoencoder.models import MitoSpace3DAutoencoder
import torch
from torch import nn
import os.path as osp

if __name__ == '__main__':
    viewer = napari.Viewer()
    cpkt_path = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/autoencoder/version_3017473/checkpoints/epoch=4-step=43060.ckpt'
    ae_model = MitoSpace3DAutoencoder()
    decoder = ae_model.decoder
    encoder = ae_model.encoder
    params_dict = torch.load(cpkt_path)

    decoder_param_dict = {k.replace('model.decoder.', ''): v for k, v in params_dict['state_dict'].items() if 'model.decoder' in k}
    encoder_param_dict = {k.replace('model.encoder.', ''): v for k, v in params_dict['state_dict'].items() if 'model.encoder' in k}

    decoder.load_state_dict(decoder_param_dict)
    encoder.load_state_dict(encoder_param_dict)

    data_dir = '/home/dhruvagarwal/projects/MitoSpace4D/data/2024_subdata/processed_data'
    drug = '20240830'

    filenames = os.listdir(os.path.join(data_dir, drug))
    idx = random.sample(range(len(filenames)), 1)[0]

    img_path = osp.join(data_dir, drug, filenames[idx])

    data = np.load(img_path)
    data = np.clip(data, 0, 20000)
    data = data / 20000
    data = data.astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(0)

    enc = encoder(data)
    dec = decoder(enc)

    original = (data.squeeze().numpy()*255).astype(np.uint8)
    dec = (dec.squeeze().detach().numpy()*255).astype(np.uint8)

    viewer.add_image(original[:, 0], name=f"Original", translate=(0, 0), colormap='cyan')
    viewer.add_image(original[:, 1], name=f"Original", translate=(0, 256+10), colormap='cyan')
    viewer.add_image(dec[:, 0], name=f"Recon", translate=(256+10, 0), colormap='cyan')
    viewer.add_image(dec[:, 1], name=f"Recon", translate=(256+10, 256+10), colormap='cyan')

    napari.run()