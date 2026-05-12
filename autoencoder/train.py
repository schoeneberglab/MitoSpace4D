import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
from dataset import get_dataloaders
from loss import ReconstructionLoss
from model import MitoSpace3DAutoencoder
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for 3D Autoencoder with wandb logging."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        runs_dir = Path(config["logging"]["run_dir"])
        runs_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=config["logging"]["wandb_project"],
            config=config,
            name=config["logging"].get("run_name", None),
            dir=str(runs_dir),
        )

        run_name = wandb.run.name

        self.checkpoint_dir = runs_dir / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = MitoSpace3DAutoencoder
        self.model.to(self.device)
        self.criterion = ReconstructionLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.optimizer.zero_grad()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
            eta_min=config["training"].get("min_lr", 1e-6),
        )

        self.train_loader, self.val_loader = self._build_dataloaders()

        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders."""
        data_config = self.config["data"]
        training_config = self.config["training"]

        train_loader, val_loader = get_dataloaders(
            data_root=data_config["manifest_path"],
            batch_size=training_config["batch_size"],
            val_split=data_config["val_split"],
            seed=data_config.get("seed", 1123),
            num_workers=data_config.get("num_workers", 4),
            channel_index=data_config.get("channel_index", None),
            num_samples=data_config.get("n_samples", None),
            prefetch_factor=data_config.get("prefetch_factor", None),
        )

        return train_loader, val_loader

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        grad_acc_steps = self.config["training"].get("grad_acc_steps", 1)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=True,
            dynamic_ncols=True,
        )

        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)

            recon = self.model(data)
            loss, _ = self.criterion(recon, data)
            loss = loss / grad_acc_steps
            loss.backward()

            if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == len(
                self.train_loader
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * grad_acc_steps
            num_batches += 1

            avg_loss_so_far = total_loss / num_batches
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * grad_acc_steps:.6f}",
                    "avg_loss": f"{avg_loss_so_far:.6f}",
                    "accum": f"{(batch_idx % grad_acc_steps) + 1}/{grad_acc_steps}",
                }
            )

            if batch_idx % self.config["logging"]["log_interval"] == 0:
                wandb.log(
                    {
                        "train/loss": loss.item() * grad_acc_steps,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": self.current_epoch,
                        "batch": batch_idx,
                    }
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        total_val_batches = len(self.val_loader)
        random_batch_idx = np.random.randint(0, total_val_batches)

        selected_batch_input = None
        selected_batch_recon = None

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=True,
            dynamic_ncols=True,
        )

        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)

            recon = self.model(data)
            loss, _ = self.criterion(recon, data)

            total_loss += loss.item()
            num_batches += 1

            avg_loss_so_far = total_loss / num_batches
            pbar.set_postfix(
                {"loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss_so_far:.6f}"}
            )

            if batch_idx == random_batch_idx:
                selected_batch_input = data
                selected_batch_recon = recon

        avg_loss = total_loss / num_batches

        wandb.log(
            {
                "val/loss": avg_loss,
                "epoch": self.current_epoch,
            }
        )

        if selected_batch_input is not None:
            self._log_reconstructions(selected_batch_input, selected_batch_recon)

        return avg_loss

    def _log_reconstructions(self, inputs: torch.Tensor, recons: torch.Tensor):
        num_images_config = self.config["logging"].get("num_images", 1)
        num_images = min(num_images_config, inputs.shape[0])  # Don't exceed batch size

        images_to_log = []

        for i in range(num_images):
            input_sample = inputs[i].cpu()
            recon_sample = recons[i].cpu()

            input_maxproj = torch.max(input_sample, dim=1)[0]
            recon_maxproj = torch.max(recon_sample, dim=1)[0]

            if input_maxproj.shape[0] > 1:
                input_maxproj = input_maxproj.mean(dim=0, keepdim=True)
                recon_maxproj = recon_maxproj.mean(dim=0, keepdim=True)

            input_img = input_maxproj.squeeze().numpy()
            recon_img = recon_maxproj.squeeze().numpy()

            input_img = (input_img - input_img.min()) / (
                input_img.max() - input_img.min() + 1e-8
            )
            recon_img = (recon_img - recon_img.min()) / (
                recon_img.max() - recon_img.min() + 1e-8
            )

            side_by_side = np.concatenate([input_img, recon_img], axis=1)

            images_to_log.append(
                wandb.Image(side_by_side, caption=f"Sample_{i}: Input | Reconstruction")
            )

        wandb.log(
            {
                "val/reconstructions": images_to_log,
                "epoch": self.current_epoch,
            }
        )

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss={self.best_val_loss:.6f}")

        if self.current_epoch % self.config["logging"]["save_interval"] == 0:
            periodic_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        batch_size = self.config["training"]["batch_size"]
        grad_acc_steps = self.config["training"].get("grad_acc_steps", 1)
        effective_batch_size = batch_size * grad_acc_steps
        print(
            f"Batch size: {batch_size} \nGradient accumulation steps: {grad_acc_steps} \nEffective batch size: {effective_batch_size}"
        )

        for epoch in range(self.current_epoch, self.config["training"]["epochs"]):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")

            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.6f}")

            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.6f}")

            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

            # Log epoch summary
            wandb.log(
                {
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "epoch": epoch,
                }
            )

        print(f"\nTraining complete! Best val_loss: {self.best_val_loss:.6f}")
        wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train 3D Autoencoder")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
