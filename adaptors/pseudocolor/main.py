import numpy as np
import os
import os.path as osp
import argparse
import wandb
import train
import evaluate

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="3D Microscopy Inpainting with SSL Embeddings")

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'all'],
                        help="Execution mode: train, eval, or both (all)")

    parser.add_argument('--workdir', type=str, required=True,
                        help="Path to the directory containing embeddings and image paths")

    parser.add_argument('--run_name', type=str, default="experiment_1",
                        help="Name of the output directory for checkpoints and logs")

    parser.add_argument('--grad_accum', type=int, default=4,
                        help="Number of gradient accumulation steps (effective batch size = batch_size * grad_accum)")

    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")

    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument('--wandb_project', type=str, default="pseudocolor",
                        help="Weights & Biases project name")

    parser.add_argument('--no_wandb', action='store_true',
                        help="Disable wandb logging")

    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to a checkpoint file to resume training from")

    args = parser.parse_args()

    # --- Setup Directories ---
    run_dir = osp.join("runs", args.run_name)
    if not osp.exists(run_dir):
        os.makedirs(run_dir)

    model_path = osp.join(run_dir, "best_inpainter_3d.pth")

    # --- Load Data ---
    print(f"Loading data from {args.workdir}...")

    # 1. Load Embeddings
    emb_path = osp.join(args.workdir, "embeddings_raw.npy")
    if not osp.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found at {emb_path}")

    embeddings = np.load(emb_path)

    # 2. Load Image Paths
    # Expects a CSV or text file listing the full paths to the .npy image files
    paths_file = osp.join(args.workdir, "image_paths.csv")

    if osp.exists(paths_file):
        image_paths = np.loadtxt(paths_file, delimiter=",", dtype=str)
    else:
        print(f"Warning: {paths_file} not found. Generating dummy filenames for testing/debugging structure.")
        # Fallback for debugging if actual files aren't present
        image_paths = np.array([osp.join(args.workdir, f"sample_{i}.npy") for i in range(len(embeddings))])

    print(f"Found {len(embeddings)} embeddings and {len(image_paths)} image paths.")

    # --- Execution ---
    use_wandb = not args.no_wandb

    # Train Mode
    if args.mode in ['train', 'all']:
        print(f"\n--- Starting Training (Run: {args.run_name}) ---")
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    # Training hyperparameters
                    "epochs": args.epochs,
                    "batch_size": 1,
                    "grad_accum_steps": args.grad_accum,
                    "effective_batch_size": args.grad_accum,
                    "learning_rate": args.lr,
                    "optimizer": "AdamW",
                    "weight_decay": 1e-5,
                    "scheduler": "ReduceLROnPlateau",
                    "scheduler_factor": 0.5,
                    "scheduler_patience": 3,
                    
                    # Model architecture
                    "model": "ConditionedUNet3D",
                    "embedding_dim": embeddings.shape[-1],
                    "n_channels": 1,
                    "n_classes": 1,
                    "encoder_channels": [32, 64, 128, 256],
                    "bottleneck_channels": 512,
                    "conditioning": "FiLM",
                    
                    # Data configuration
                    "n_samples": len(embeddings),
                    "n_image_paths": len(image_paths),
                    "embedding_shape": embeddings.shape,
                    "has_time_embeddings": len(embeddings.shape) == 3,
                    "timepoints_per_file": 20,
                    "test_split": 0.1,
                    
                    # Loss function
                    "loss_function": "L1Loss",
                    
                    # Data paths
                    "workdir": args.workdir,
                    "run_name": args.run_name,
                }
            )

        train.train_model(
            image_paths,
            embeddings,
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=1,  # Keep physical batch size at 1 for 3D memory safety
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
            use_wandb=use_wandb,
            resume_checkpoint=args.resume_checkpoint
        )

        if use_wandb:
            wandb.finish()

    # Evaluation Mode
    if args.mode in ['eval', 'all']:
        print(f"\n--- Starting Evaluation (Run: {args.run_name}) ---")

        if not osp.exists(model_path):
            print(f"Error: Model checkpoint not found at {model_path}. Skipping evaluation.")
        else:
            evaluate.generate_visuals(
                image_paths,
                embeddings,
                model_path,
                output_dir=run_dir,
                n_samples=3,
                use_wandb=use_wandb
            )
            print("Evaluation complete.")