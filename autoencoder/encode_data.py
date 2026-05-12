import torch
import numpy as np
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm

from model import MitoSpace3DAutoencoder

@torch.no_grad()
def normalize_channel(tensor: torch.Tensor):
    """Normalize tensor to [0, 1] range."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)


def preprocess_data(data: np.ndarray):
    if data.ndim == 4:
        data = data[:, np.newaxis, ...]

    original_depth = data.shape[2]
    if original_depth < 64:
        pad_width = ((0, 0), (0, 0), (0, 64 - original_depth), (0, 0), (0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)

    return data, original_depth

@torch.no_grad()
def encode_file(file_path: Path, model: MitoSpace3DAutoencoder, device: torch.device, output_dir: Path):
    data = np.load(file_path, mmap_mode='r')
    data, original_depth = preprocess_data(data)

    data_tensor = torch.from_numpy(data).float().to(device)

    data_tensor = normalize_channel(data_tensor)

    encoded_data = []
    for i in range(data_tensor.shape[0]):
        encoded = model.encoder(data_tensor[i:i+1])
        encoded_data.append(encoded.cpu().numpy())

    encoded = np.concatenate(encoded_data, axis=0)

    relative_path = file_path.relative_to(file_path.parents[2])
    output_file = output_dir / relative_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, encoded)


def main():
    parser = argparse.ArgumentParser(description='Encode dataset using trained autoencoder')
    parser.add_argument('--checkpoint', type=str, default='./runs/2024v3_ft_sequence-norm/latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str,
                        default='/mnt/aquila/ssd_processing/Others/MitoSpace4D/2024v3_data/processed_data/',
                        help='Root directory of processed data')
    parser.add_argument('--pattern', type=str, default='2024*/*-0-1.npy',
                        help='Glob pattern for finding input files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    data_root = Path(args.data_root)
    files = sorted(glob(str(data_root / args.pattern)))
    print(f"Found {len(files)} files to encode")

    if len(files) == 0:
        print("No files found! Check your data_root and pattern.")
        return

    output_dir = data_root.parent / 'encoded_data'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\nEncoding dataset...")
    for file_path in tqdm(files, desc="Encoding", unit="file"):
        try:
            encode_file(Path(file_path), model, device, output_dir)
        except Exception as e:
            print(f"\n✗ Failed to encode {file_path}: {e}")
            continue


if __name__ == '__main__':
    main()
