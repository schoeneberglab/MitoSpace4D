"""
Generate MitoSpace-3D embeddings for the reproducibility dataset.

The 3D SimCLR model (`archive/models_simple_3d_2024v3.py`, `Lightweight3DResNet`) was
trained on autoencoder-encoded volumes: it decodes the *last* latent frame of each movie
and embeds that single 3D volume. The reproducibility data is raw (un-encoded) 4D LLSM
data, so — exactly as `evaluate.py` does for the 4D model — the decoder is bypassed and
the raw last frame is fed straight into the ResNet trunk.

Input:  <data_path>/<YYYYMMDD-R>/<cell>-0.npy of shape (T=20, Z=60, Y=256, X=256), uint16.
Output: 2048-d (fc) and 512-d (ResNet trunk) L2-normalized embeddings, one row per movie.

Usage:
    python generate_3dms_reproducibility_embeddings.py \
        --checkpoint_path manuscript_v2/checkpoints/ms3d_2024v3_epoch=225-step=24182_best.ckpt \
        --data_path /mnt/aquila/ssd_processing/Others/MitoSpace4D/reproducibility/processed_data \
        --embeddings_dir 3dms_reproducibility
"""

import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
import tqdm
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from simclr.model import Basic3DBlock
from utils.utils import load_config

np.random.seed(0)
torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy("file_system")

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="MitoSpace-3D reproducibility embeddings")
parser.add_argument(
    "--config",
    default="/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml",
    type=str,
    help="Config path (used for model_params.channels).",
)
parser.add_argument(
    "--checkpoint_path",
    default="/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/checkpoints/ms3d_2024v3_epoch=225-step=24182_best.ckpt",
    help="SimCLR 3D checkpoint.",
)
parser.add_argument(
    "--data_path",
    default="/mnt/aquila/ssd_processing/Others/MitoSpace4D/reproducibility/processed_data",
    help="Root of the raw (un-encoded) reproducibility data.",
)
parser.add_argument(
    "--drug_labels",
    default="/home/earkfeld/Projects/MitoSpace4D/metadata/drugs_to_labels.txt",
    help="Acquisition-folder -> drug/label mapping.",
)
parser.add_argument(
    "--embeddings_dir",
    default="/home/earkfeld/Projects/MitoSpace4D/3dms_reproducibility",
    help="Directory to write embeddings into.",
)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--workers", type=int, default=10)
parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="Debug: only embed this many movies (0 = all).",
)


# --------------------------------------------------------------------------------------
# Model: the 3D trunk of archive/models_simple_3d_2024v3.py::Lightweight3DResNet.
# The frozen autoencoder decoder is not instantiated (raw volumes go in directly), so the
# `2 * x - 1` rescaling and the last-frame selection that live inside the decoder branch
# of the original forward() are applied here explicitly.
# --------------------------------------------------------------------------------------
class Lightweight3DResNet(nn.Module):
    def __init__(self, embedding_size=2048, cfg=None):
        super().__init__()

        self._channels = cfg["model_params"]["channels"]
        print(f"Using channels: {self._channels} for input.")
        in_channels = len(self._channels)
        self._channels = torch.tensor(self._channels).to(torch.int32)

        stem = nn.Sequential(
            nn.Conv3d(
                in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1),
        )

        self.resnet = nn.Sequential(
            stem,
            self._make_layer(16, 32, num_blocks=2, stride=2),
            self._make_layer(32, 64, num_blocks=2, stride=2),
            self._make_layer(64, 128, num_blocks=2, stride=2),
            self._make_layer(128, 512, num_blocks=2, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.fc = nn.Linear(512, embedding_size)

        self.proj = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [Basic3DBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(Basic3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, get_resnet_feats=False):
        """x: (B, T, C, D, H, W) in [0, 1]. Only the last timestep is embedded."""
        x = x[:, -1:, ...]  # (B, 1, C, D, H, W)
        x = 2 * x - 1  # Scale to [-1, 1]
        x = x[:, :, self._channels.to(x.device), ...]

        batch_size, time_steps, channels, depth, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, depth, height, width)

        x = self.resnet(x)
        x_resnet = x.view(batch_size, -1)  # (B, 512)

        x = self.fc(x_resnet)  # (B, 2048)
        out = self.proj(x)

        if get_resnet_feats:
            return x, x_resnet, out
        return x, out


def load_3d_model(checkpoint_path, cfg):
    """Instantiate the trunk and load the SimCLR checkpoint (decoder weights ignored)."""
    model = Lightweight3DResNet(embedding_size=2048, cfg=cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    trunk_state = {
        k[len("model.") :]: v
        for k, v in state_dict.items()
        if k.startswith("model.") and not k.startswith("model.decoder.")
    }
    missing, unexpected = model.load_state_dict(trunk_state, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint is missing trunk weights: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected trunk weights in checkpoint: {unexpected}")

    print(
        f"Loaded {len(trunk_state)} trunk tensors from {osp.basename(checkpoint_path)} "
        f"(epoch {ckpt.get('epoch')}); "
        f"{sum(1 for k in state_dict if k.startswith('model.decoder.'))} decoder tensors skipped."
    )

    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def read_drug_labels(fpath):
    """acquisition folder -> {'drug': str, 'label': int}"""
    drug_labels = {}
    with open(fpath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            folder, drug, label = line.split()
            drug_labels[folder] = {"drug": drug, "label": int(label)}
    return drug_labels


class ReproducibilityDataset(Dataset):
    """Raw 4D reproducibility movies, one .npy per cell: (T, Z, Y, X) uint16."""

    def __init__(self, root_dir, drug_labels, limit=0):
        self.root_dir = root_dir

        folders = sorted(
            d
            for d in os.listdir(root_dir)
            if osp.isdir(osp.join(root_dir, d)) and d in drug_labels
        )
        skipped = sorted(
            d
            for d in os.listdir(root_dir)
            if osp.isdir(osp.join(root_dir, d)) and d not in drug_labels
        )
        if skipped:
            print(f"WARNING: no drug label for folders {skipped}; skipping them.")

        self.filenames, self.labels, self.drugs, self.dates, self.replicates = (
            [],
            [],
            [],
            [],
            [],
        )
        for folder in folders:
            files = sorted(
                osp.join(root_dir, folder, f)
                for f in os.listdir(osp.join(root_dir, folder))
                if f.endswith(".npy")
            )
            date, replicate = folder.split("-")
            self.filenames.extend(files)
            self.labels.extend([drug_labels[folder]["label"]] * len(files))
            self.drugs.extend([drug_labels[folder]["drug"]] * len(files))
            self.dates.extend([date] * len(files))
            self.replicates.extend([replicate] * len(files))

        if limit:
            self.filenames = self.filenames[:limit]
            self.labels = self.labels[:limit]
            self.drugs = self.drugs[:limit]
            self.dates = self.dates[:limit]
            self.replicates = self.replicates[:limit]

        print(f"Found {len(self.filenames)} movies across {len(folders)} acquisitions.")
        for drug in sorted(set(self.drugs)):
            print(f"  {drug}: {self.drugs.count(drug)} movies")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        movie = np.load(img_name).astype(np.float32)  # (T, Z, Y, X)

        # Min-max over the whole movie, matching the normalization the autoencoder used
        # when the training latents were produced (autoencoder/encode_data.py).
        movie = (movie - movie.min()) / (movie.max() - movie.min() + 1e-9)

        last_frame = movie[-1][np.newaxis, np.newaxis, ...]  # (T=1, C=1, Z, Y, X)

        return {
            "images": last_frame,
            "classes": self.labels[idx],
            "image_paths": img_name,
        }


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)

    embeddings_dir = args.embeddings_dir
    os.makedirs(embeddings_dir, exist_ok=True)

    with open(osp.join(embeddings_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    with open(osp.join(embeddings_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    print(f"Ckpt Path: {args.checkpoint_path}")
    print(f"Data Path: {args.data_path}")

    model = load_3d_model(args.checkpoint_path, cfg)

    drug_labels = read_drug_labels(args.drug_labels)
    dataset = ReproducibilityDataset(args.data_path, drug_labels, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )

    embeddings_list, resnet_embeddings_list, labels_list, img_pth_list = [], [], [], []

    for batch in tqdm.tqdm(loader, total=len(loader)):
        im, lbl, img_pth = batch["images"], batch["classes"], batch["image_paths"]
        img_pth_list.extend(img_pth)

        with torch.no_grad():
            features, resnet_features, _ = model(
                im.to(device, non_blocking=True).float(), get_resnet_feats=True
            )
            features = F.normalize(features, dim=-1)  # (B, 2048)
            resnet_features = F.normalize(resnet_features, dim=-1)  # (B, 512)

        embeddings_list.extend(features.cpu().numpy().astype(np.float32))
        resnet_embeddings_list.extend(resnet_features.cpu().numpy().astype(np.float32))
        labels_list.extend(lbl.numpy().reshape(-1).astype(np.int32))

    embeddings_arr = np.asarray(embeddings_list, dtype=np.float32)
    resnet_embeddings_arr = np.asarray(resnet_embeddings_list, dtype=np.float32)
    labels_arr = np.asarray(labels_list, dtype=np.int32)

    np.save(osp.join(embeddings_dir, "embeddings.npy"), embeddings_arr)
    np.save(osp.join(embeddings_dir, "embeddings_resnet.npy"), resnet_embeddings_arr)
    np.save(osp.join(embeddings_dir, "labels.npy"), labels_arr)
    np.save(
        osp.join(embeddings_dir, "label_names.npy"),
        np.array(sorted({d["drug"] for d in drug_labels.values()})),
    )

    with open(osp.join(embeddings_dir, "image_paths.csv"), "w") as f:
        for pth in img_pth_list:
            f.write(f"{pth}\n")

    # Per-sample metadata: the day/replicate columns drive the leave-one-day-out folds.
    # The loader is unshuffled, so row i corresponds to dataset index i.
    assert img_pth_list == dataset.filenames, "Loader returned paths out of order"
    meta = pd.DataFrame(
        {
            "image_paths": img_pth_list,
            "labels": labels_arr,
            "label_names": dataset.drugs,
            "date": dataset.dates,
            "replicate": dataset.replicates,
        }
    )
    meta.to_csv(osp.join(embeddings_dir, "metadata.csv"), index=False)

    print(f"\nSaved embeddings {embeddings_arr.shape} -> {embeddings_dir}")
    print(meta.groupby(["date", "label_names"]).size().to_string())


if __name__ == "__main__":
    main()
