from application.gen_embedding import prepare_data_for_model
from simclr.models_simple import Lightweight3DResNet
from simclr.simclr import SimCLRRunner
from utils.utils import load_config
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_map(real_image, encoder):
    # Select an image
    image = real_image.to('cuda')

    # Compute original embedding
    original_embedding, _ = encoder(image)
    original_embedding = original_embedding.detach()

    # Create heatmap by perturbing the image
    heatmap = torch.zeros_like(image[0, 0, 0, 0])  # Only 2D heatmap for visualization

    perturbation_strength = 0.05
    patch_size = 3

    pbar = tqdm(total=(image.shape[-1] // patch_size) * (image.shape[-2] // patch_size))
    for x in range(0, image.shape[-1], patch_size):
        for y in range(0, image.shape[-2], patch_size):
            perturbed_image = image.clone()
            perturbed_image[:, :, :, :, x:x + patch_size, y:y + patch_size] += perturbation_strength
            perturbed_embedding, _ = encoder(perturbed_image)

            # Compute difference in embedding space

            # compute difference as cosine similarity
            diff = torch.nn.functional.cosine_similarity(original_embedding, perturbed_embedding, dim=1)
            diff = -diff.mean()  # Negative because we want to maximize the difference
            heatmap[x:x + patch_size, y:y + patch_size] = diff.item()

            pbar.update(1)

    plt.imshow(np.max(real_image[0, 0, 0].cpu().numpy(), axis=0))
    plt.show()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    plt.imshow(heatmap.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.title("Contrastive Explanation Heatmap")
    plt.show()


def get_model(cfg, ckpt_path):
    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'],
                                apply_aug=False)
    # print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        ckpt_path, model=model, cfg=cfg
    ).cuda()
    model.eval()

    return model.model


if __name__ == '__main__':
    fpath_1 = '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data_5/processed_data/20250203_Oligomycin/000308.npy'
    image = prepare_data_for_model(fpath_1)

    cfg = load_config('/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml')
    ckpt_path = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/checkpoints/epoch=287-step=83534-val_loss=0.00.ckpt'
    model = get_model(cfg, ckpt_path)

    get_map(image, model)