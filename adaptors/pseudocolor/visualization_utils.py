"""
Shared visualization utilities for pseudocolor adaptor.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def generate_prediction_visuals(model, dataset, device, indices=None, n_samples=3, epoch=None):
    """
    Generate visualization comparing input morphology, actual TMRM, and predicted TMRM.

    Args:
        model: The trained/training model
        dataset: Dataset to sample from (should support __getitem__ and __len__)
        device: torch device to run inference on
        indices: Optional specific indices to visualize. If None, random samples are chosen.
        n_samples: Number of samples to visualize (used if indices is None)
        epoch: Optional epoch number to include in title

    Returns:
        matplotlib figure with visualizations
    """
    model.eval()

    # Random sampling if indices not provided
    if indices is None:
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    fig = plt.figure(figsize=(10, 3 * len(indices)))

    for i, idx in enumerate(indices):
        img, emb, target = dataset[idx]

        img_in = img.unsqueeze(0).to(device)
        emb_in = emb.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_in, emb_in).cpu().squeeze().numpy()

        truth = target.squeeze().numpy()
        input_morph = img.squeeze().numpy()

        # Max Intensity Projection (Axis 0 = Depth)
        plt.subplot(len(indices), 3, i * 3 + 1)
        plt.imshow(np.max(input_morph, axis=0), cmap='viridis')
        if i == 0:
            if epoch is not None:
                plt.title(f"Morphology (Epoch {epoch})")
            else:
                plt.title("Morphology")
        plt.axis('off')

        plt.subplot(len(indices), 3, i * 3 + 2)
        plt.imshow(np.max(truth, axis=0), cmap='hot')
        if i == 0:
            plt.title("Actual TMRM")
        plt.axis('off')

        plt.subplot(len(indices), 3, i * 3 + 3)
        plt.imshow(np.max(pred, axis=0), cmap='hot')
        if i == 0:
            plt.title("Predicted TMRM")
        plt.axis('off')

    plt.tight_layout()
    return fig
