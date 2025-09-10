import torch
import numpy as np
from autoencoder.autoencoder import AutoEncoderRunner
from autoencoder.models import MitoSpace3DAutoencoder
import einops
import os.path as osp
import os
import napari

if __name__ == "__main__":
    ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/MitospaceAutoencoder.ckpt"
    data_dir = "/mnt/aquila0/ssd_processing/Others/MitoSpace4D/2025_summer_new/20250722-2"

    file = np.random.choice(os.listdir(data_dir))
    # infile = osp.join(data_dir, file)
    infile = osp.join(data_dir, "000000-0.npy")

    device = torch.device("cuda")
    model = MitoSpace3DAutoencoder()
    runner = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
    encoder = runner.model.encoder
    decoder = runner.model.decoder

    encoder.eval()
    decoder.eval()

    for param in decoder.parameters():
        param.requires_grad = False

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.to(device)
    decoder.to(device)

    in_data = np.load(infile) # C, T, Z, Y, X
    print(in_data.shape)
    in_data = einops.rearrange(in_data, 'c t z y x -> 1 t c z y x')  # B, T, C, Z, Y, X
    print(in_data.shape)
    in_data = torch.from_numpy(in_data)
    print(in_data.shape)
    in_data = in_data.to(device)
    with torch.no_grad():
        enc_data = encoder(in_data)
        out_data = decoder(enc_data)

    out_data = out_data.detach().cpu().numpy()
    in_data = in_data.detach().cpu().numpy()
    out_data = einops.rearrange(out_data, '1 t c z y x -> c t z y x')
    in_data = einops.rearrange(in_data, '1 t c z y x -> c t z y x')
    
    print("in_data shape:", in_data.shape)
    print("out_data shape:", out_data.shape)
    
    in_data0 = in_data[0]
    in_data1 = in_data[1]
    out_data0 = out_data[0]
    out_data1 = out_data[1]

    # Set up the viewer
    viewer = napari.Viewer(ndisplay=3)

    # Translate images so they don't overlap
    viewer.add_image(in_data0, name='Original Channel 0', colormap='green')
    # viewer.add_image(out_data0, name='Reconstructed Channel 0', colormap='green', translate=(0, in_data0.shape[3]+10, 0))
    # viewer.add_image(in_data1, name='Original Channel 1', colormap='magenta', translate=(0, 0, in_data0.shape[2]+10))
    # viewer.add_image(out_data1, name='Reconstructed Channel 1', colormap='magenta', translate=(0, in_data0.shape[3]+10, in_data0.shape[2]+10))

    viewer.add_image(in_data1, name='Original Channel 1', colormap='magenta', translate=(0, in_data0.shape[2]+10, 0))

    viewer.add_image(out_data0, name='Reconstructed Channel 0', colormap='green', translate=(0, 0, in_data0.shape[2]+10))
    viewer.add_image(out_data1, name='Reconstructed Channel 1', colormap='magenta', translate=(0, in_data0.shape[3]+10, in_data0.shape[2]+10))


    napari.run()
    viewer.close()
