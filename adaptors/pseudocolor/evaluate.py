import torch
import os.path as osp
import wandb
import matplotlib.pyplot as plt

from model import ConditionedUNet3D
from dataset import get_dataloaders
from visualization_utils import generate_prediction_visuals


def generate_visuals(image_paths, embeddings, model_path, output_dir, n_samples=3, use_wandb=False):
    """
    Generate evaluation visuals using a trained model.

    Args:
        image_paths: List of paths to image files
        embeddings: Embeddings array
        model_path: Path to trained model checkpoint
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
        use_wandb: Whether to log to wandb
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device} loading from {model_path}")

    # Re-create validation loader (Deterministic split ensures these are unseen files)
    _, val_loader = get_dataloaders(image_paths, embeddings, batch_size=1)
    dataset = val_loader.dataset

    # Load Model
    model = ConditionedUNet3D(n_channels=1, n_classes=1, embedding_dim=embeddings.shape[-1]).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load from full training checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load legacy direct state dict
        model.load_state_dict(checkpoint)

    model.eval()

    # Generate visualizations using consolidated function
    fig = generate_prediction_visuals(
        model=model,
        dataset=dataset,
        device=device,
        n_samples=n_samples,
        epoch=None
    )

    # Save to disk
    save_path = osp.join(output_dir, 'evaluation_results.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visuals saved to {save_path}")

    # Log to wandb if enabled
    if use_wandb:
        wandb.log({"evaluation_results": wandb.Image(save_path)})

    # Clean up
    plt.close(fig)
