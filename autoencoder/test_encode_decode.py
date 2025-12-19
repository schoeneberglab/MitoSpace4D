import torch
import numpy as np
from autoencoder.autoencoder_runner import AutoEncoderRunner
from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
import einops
import os.path as osp
import os
import napari
import tifffile
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # #-- Local paths
    ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/autoencoder/mitospace_resnet_autoencoder_20251018.ckpt"
    data_root = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_summer_new/"
    # data_root = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_summer_new/"

    #-- Delta Paths
    # ckpt_path = "/u/earkfeld/MitoSpace4D/autoencoder/runs/1081149/lightning_logs/kinetics_autoencoder/checkpoints/last.ckpt"
    # data_dir = "/work/nvme/begq/MitoSpace4D/data/2025_data/20250722-2"

    data_dir = "20240826-1"
    # data_dir = "20250902-1"
    # file = "000000-0.npy"

    visualize = True
    save_imgs = False
    plot_histograms = False

    file = np.random.choice(os.listdir(osp.join(data_root, data_dir)))
    infile = osp.join(data_root, data_dir, file)

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
    in_data = einops.rearrange(in_data, 'c t z y x -> 1 t c z y x')  # B, T, C, Z, Y, X
    in_data = torch.from_numpy(in_data)
    in_data = in_data.to(device)
    with torch.no_grad():
        enc_data = encoder(in_data)
        out_data = decoder(enc_data)

    out_data = out_data.detach().cpu().numpy()
    in_data = in_data.detach().cpu().numpy()
    out_data = einops.rearrange(out_data, '1 t c z y x -> c t z y x')
    in_data = einops.rearrange(in_data, '1 t c z y x -> c t z y x')
    
    # print("in_data shape:", in_data.shape)
    # print("out_data shape:", out_data.shape)

    if plot_histograms:
        # Plot histograms of original and reconstructed data for each channel side by side
        channels = ['TMRM', 'Morphology']
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for i in range(2):
            ax[i, 0].hist(in_data[i].flatten(), bins=25, color='blue', alpha=0.7)
            ax[i, 0].set_title(f'Original {channels[i]} Histogram')
            ax[i, 1].hist(out_data[i].flatten(), bins=25, color='orange', alpha=0.7)
            ax[i, 1].set_title(f'Reconstructed {channels[i]} Histogram')
        plt.tight_layout()
        plt.show()
    
    if save_imgs:
        # Save as tiff for viewing
        tifffile.imwrite("original.tiff", in_data, imagej=True)
        tifffile.imwrite("reconstructed.tiff", out_data, imagej=True)

    if visualize:
        original_tmrm = in_data[0]
        original_morph = in_data[1]
        recon_tmrm = out_data[0]
        recon_morph = out_data[1]    

        # Set up the viewer
        viewer = napari.Viewer(ndisplay=3)

        # Translate images so they don't overlap
        viewer.add_image(original_morph, name='Original Morphology', colormap='green')
        viewer.add_image(original_tmrm, name='Original TMRM', colormap='magenta', translate=(0, original_morph.shape[2]+10, 0))
        viewer.add_image(recon_morph, name='Reconstructed Morphology', colormap='green', translate=(0, 0, original_morph.shape[2]+10))
        viewer.add_image(recon_tmrm, name='Reconstructed TMRM', colormap='magenta', translate=(0, original_morph.shape[3]+10, original_morph.shape[2]+10))

        viewer.reset_view()
        napari.run()
        viewer.close()
