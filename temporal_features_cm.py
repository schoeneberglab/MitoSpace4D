import numpy as np
from utils.utils import load_config
from simclr.models_simple import Lightweight3DResNet
from simclr.simclr import SimCLRRunner
import torch
import os
import joblib
import einops

def cosine_distance(eval_embeddings, train_embeddings, weighted=False, temperature=1.):
    dist_matrix = eval_embeddings @ train_embeddings.T
    if weighted:
        dist_matrix = dist_matrix / temperature
        dist_matrix = np.exp(dist_matrix)

    # dist_matrix_idxs = (-1 * dist_matrix).argsort(1)  # because we want to sort in descending order of distances
    # dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)

    # return dist_matrix_sorted, dist_matrix_idxs
    return dist_matrix

def setup_model(cfg, model_ckpt_path, decoder_ckpt_path=None, device='cuda'):
    
    # Build and load model
    model = Lightweight3DResNet(embedding_size=2048, cfg_aug=cfg['data_params']['transforms'], apply_aug=False, decoder_checkpoint_path=decoder_ckpt_path)
    model = SimCLRRunner.load_from_checkpoint(model_ckpt_path, model=model, cfg=cfg, strict=False).model
    model.eval().to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model

if __name__ == "__main__":
    model_ckpt_path = "/home/earkfeld/Projects/MitoSpace4D/checkpoints/resnetbilstm_encoded_resumed2024ckpt_kinetics_warmup-5eps_decoupled-tmrm_50eps_r20251026.ckpt"
    decoder_ckpt_path = None
    cfg_path = "/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml"

    dataset_root = "/mnt/aquila/SSD_processing/Others/MitoSpace4D/2025_summer_new/"
    data_root = "20250724-1"

    data_dir = os.path.join(dataset_root, data_root)
    sample_file = np.random.choice(os.listdir(data_dir))
    sample_file = os.path.join(data_dir, sample_file)

    cfg = load_config(cfg_path)

    model = setup_model(cfg, model_ckpt_path, decoder_ckpt_path, device='cuda')

    # Load the sample
    data = np.load(sample_file)
    data_tensor = torch.from_numpy(data).unsqueeze(0).to('cuda')  # Add batch dimension
    data_tensor = einops.rearrange(data_tensor, 'b c t d w h -> b t c d w h')
    with torch.no_grad():
        embeddings, _ = model(data_tensor)  # (1, t, emb_dim)
    embeddings = embeddings.squeeze(0).cpu().numpy()  # (t, emb_dim)

    # Create a cosine distance matrix from every frame to every other frame
    dist_matrix = cosine_distance(embeddings, np.flip(embeddings, axis=0), weighted=False)
    
    # plot= the confusion matrix with frame indices as the ticks
    import matplotlib.pyplot as plt
    plt.imshow(dist_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=np.arange(dist_matrix.shape[0]), labels=np.arange(dist_matrix.shape[0]))
    plt.yticks(ticks=np.arange(dist_matrix.shape[0]), labels=np.arange(dist_matrix.shape[0]))
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    plt.title('Cosine Distance Matrix Between Frames')
    plt.show()