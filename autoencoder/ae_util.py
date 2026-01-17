import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm

# UPDATE THESE IF NEEDED
from autoencoder.autoencoder_models_resnet import MitoSpace3DAutoencoder
from autoencoder.autoencoder_runner import AutoEncoderRunner
CKPT_PATH = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/mitospace_resnet_autoencoder_20251018.ckpt"

class AEUtil:
    def __init__(self, ckpt_path=CKPT_PATH):
        model = MitoSpace3DAutoencoder()
        runner = AutoEncoderRunner.load_from_checkpoint(ckpt_path, model=model)
        print("Loaded model from checkpoint.")

        self.decoder = runner.model.decoder
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        # Free up some memory
        del model
        del runner

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder.to(self.device)
        
    def load(self, path) -> np.ndarray:
        """
        Decode a 5D numpy array (T,latent_dim,D,H,W) into its reconstructed image.

        Args:
            path: Path to the .npy file containing the encoded data.

        Returns:
            np.ndarray: Decoded image of shape (T,C,Z,Y,X).
        """

        encoded = np.load(path)  # (T,latent_dim,D,H,W)
        # print(encoded.shape)
        with torch.no_grad():
            z = torch.from_numpy(encoded).unsqueeze(0).float().to(self.device)  # (1,T,latent_dim,D,H,W)
            x_recon = self.decoder(z)                                           # (1,T,C,Z,Y,X)
            decoded = x_recon.squeeze(0).cpu().numpy()                          # (T,C,Z,Y,X)
        decoded = np.transpose(decoded, (1,0,2,3,4))  # (C,T,Z,Y,X)
        return decoded

# Example usage:
# ae_util = AEUtil(ckpt_path=<path_to_checkpoint>)
# decoded_image = ae_util.load(<path_to_encoded_npy_file>)  # decoded_image shape: (C,T,Z,Y,X)

if __name__ == "__main__":
    ae_util = AEUtil(ckpt_path=CKPT_PATH)
    sample_encoded_path = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2024_data_encoded/20240729-1/000000-0.npy"  # Replace with actual path
    decoded_image = ae_util.load(sample_encoded_path)
    print(f"Decoded image shape: {decoded_image.shape}")  # Should be (C,T,Z,Y,X)