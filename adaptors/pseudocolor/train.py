import torch
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
from tqdm import trange
import wandb
import matplotlib.pyplot as plt

from model import ConditionedUNet3D
from dataset import get_dataloaders
from visualization_utils import generate_prediction_visuals

def threshold_otsu(image, nbins=256):
    """
    GPU-accelerated Otsu's thresholding in PyTorch.
    Matches scikit-image's output and behavior.
    """
    # Ensure image is a float tensor for calculation
    image = image.to(torch.float32)

    # 1. Calculate the histogram
    min_val, max_val = image.min(), image.max()
    hist = torch.histc(image, bins=nbins, min=min_val, max=max_val)

    # 2. Get bin centers (to map the optimal bin back to a threshold value)
    bin_centers = torch.linspace(min_val, max_val, nbins, device=image.device)

    # 3. Calculate probabilities and cumulative sums
    # weight1 (w0): probability of the background
    # weight2 (w1): probability of the foreground
    # mean1 (mu0): mean of the background
    # mean2 (mu1): mean of the foreground

    hist_norm = hist / hist.sum()
    weight1 = torch.cumsum(hist_norm, dim=0)
    weight2 = 1.0 - weight1

    mean1 = torch.cumsum(hist_norm * bin_centers, dim=0) / weight1
    # We use a small epsilon or flip/cumsum to avoid division by zero
    mean2 = (mean1[-1] - mean1 * weight1) / weight2

    # 4. Calculate Between-Class Variance
    # sigma_b^2 = w1 * w2 * (mu1 - mu2)^2
    variance12 = weight1 * weight2 * (mean1 - mean2) ** 2

    # 5. Find the threshold that maximizes between-class variance
    # Handle NaNs that occur due to division by zero at the boundaries
    variance12[torch.isnan(variance12)] = 0

    max_idx = torch.argmax(variance12)
    threshold = bin_centers[max_idx]

    return threshold

def MaskedL1Loss(x_morphology, y_tmrm_predicted, y_tmrm_target):
    threshold = threshold_otsu(x_morphology)
    mask = (x_morphology > threshold)
    return nn.L1Loss()(y_tmrm_predicted[mask], y_tmrm_target[mask])

def train_model(image_paths, embeddings, run_dir, epochs=50, batch_size=1, grad_accum_steps=4, lr=1e-4, use_wandb=False, resume_checkpoint=None):
    """
    Train the 3D Inpainter with Gradient Accumulation.
    effective_batch_size = batch_size * grad_accum_steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Physical Batch Size: {batch_size} | Accumulation Steps: {grad_accum_steps}")
    print(f"Effective Batch Size: {batch_size * grad_accum_steps}")

    if not osp.exists(run_dir):
        os.makedirs(run_dir)

    # Get standard loaders
    train_loader, val_loader = get_dataloaders(image_paths, embeddings, batch_size=batch_size)

    model = ConditionedUNet3D(
        n_channels=1,
        n_classes=1,
        embedding_dim=embeddings.shape[-1]
    ).to(device)

    # criterion = nn.L1Loss()
    criterion = MaskedL1Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_model_path = osp.join(run_dir, 'best_inpainter_3d.pth')
    start_epoch = 0

    if resume_checkpoint:
        if osp.exists(resume_checkpoint):
            print(f"Resuming training from {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)

            # Handle full checkpoint vs legacy state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                print(f"Resumed from epoch {start_epoch} with val_loss {best_val_loss}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded legacy model state dict. Starting fresh training epoch sequence.")
        else:
            print(f"Warning: Checkpoint {resume_checkpoint} not found. Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Initialize gradients

        with trange(len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch") as pbar:
            for i, (batch_img, batch_emb, batch_target) in enumerate(train_loader):
                batch_img = batch_img.to(device)
                batch_emb = batch_emb.to(device)
                batch_target = batch_target.to(device)

                # Forward Pass
                outputs = model(batch_img, batch_emb)

                # Normalize loss by accumulation steps so the gradients aren't huge
                # loss = criterion(outputs, batch_target) / grad_accum_steps # Normal L1 Loss
                loss = criterion(batch_img, outputs, batch_target) # Custom Masked L1 Loss

                # Backward Pass (Accumulate Gradients)
                loss.backward()

                # Optimizer Step (Only every N steps)
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                # Logging: Scale loss back up for display purposes
                current_loss = loss.item() * grad_accum_steps
                train_loss += current_loss * batch_img.size(0)

                pbar.set_postfix(loss=current_loss)
                pbar.update()

        # Validation Loop (No accumulation needed here)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_img, v_emb, v_target in val_loader:
                v_img = v_img.to(device)
                v_emb = v_emb.to(device)
                v_target = v_target.to(device)

                v_out = model(v_img, v_emb)
                # val_loss += criterion(v_out, v_target).item() * v_img.size(0)
                val_loss += criterion(v_img, v_out, v_target).item() * v_img.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        # Generate and log visualizations
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        # Generate visualizations every epoch (change to (epoch + 1) % 5 == 0 for every 5 epochs)
        if True:
            print(f"Generating visualizations for epoch {epoch + 1}...")
            fig = generate_prediction_visuals(
                model=model,
                dataset=val_loader.dataset,
                device=device,
                n_samples=3,
                epoch=epoch + 1
            )

            # Save locally
            vis_path = osp.join(run_dir, f'epoch_{epoch + 1:03d}_predictions.png')
            fig.savefig(vis_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {vis_path}")

            # Log to wandb
            if use_wandb:
                log_dict["predictions"] = wandb.Image(fig)

            plt.close(fig)

        if use_wandb:
            wandb.log(log_dict)

        # Save latest checkpoint every epoch to allow resuming if crashed
        latest_path = osp.join(run_dir, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }, latest_path)

        if avg_val_loss < best_val_loss:
            print(f"Saving best model to {best_model_path}...")
            best_val_loss = avg_val_loss
            # Save full checkpoint now, so we can resume from best if needed
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, best_model_path)

            if use_wandb:
                wandb.log({"best_val_loss": best_val_loss})
