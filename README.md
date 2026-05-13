
# MitoSpace4D: Training Pipeline
[mitospace.ai](https://mitospace.ai)

This repository contains the training pipeline for **MitoSpace4D**, a 4D deep learning model designed to capture mitochondrial morphological changes across time and treatments.

## 📦 Prerequisites

- **Python**: Version 3.8 or higher.
- **Dependencies**: Install all required packages via `pip`.

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/mitospace4d.git
   cd mitospace4d
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup**:  
   Ensure you have access to the dataset and place it in the appropriate directory, as specified in the configuration file below.

---

## ⚙️ Configuration

Edit the configuration file located at:

```
/tscc/nfs/home/d5agarwal/projects/MitoSpace4D/simclr/config.yaml
```

---

## 🏋️ Training

Before training, ensure that the model uses **encoded data** from the autoencoder:

1. Open `data_aug/mitospace_dataset.py`.
2. In the constructor, replace all occurrences of `processed_data` with `encoded_data`.
3. In the `__getitem__` method, **comment out** normalization lines (the encoded data is already normalized).

Start training:

```bash
python -m train_simclr
```

To run training on SLURM:

```bash
sbatch run.sb
```

Make sure that:
- `--ntasks-per-node` is set to the number of GPUs you want per node.
- `--gres=gpu:<n>` matches that same number (`<n>` = GPUs per node).

---

## 🧠 Training Details

- **System**: SDSC (San Diego Supercomputer Center) server  
- **Queue**: `hotel-gpu`  
- **GPUs**: V100  
- **Setup**: 15 nodes × 4 GPUs = **60 GPUs total**  
- **Batch size**: 2 per GPU → **Total batch size = 120**  
- **Epochs**: 300  
- **Training duration**: ~3 days  
- **SLURM config**: See `run.sb` for detailed job submission parameters.

---

## 🔍 Inference / Embedding Space Generation

To generate embeddings from **raw data** (not encoded):

1. Open `data_aug/mitospace_dataset.py`.
2. In the constructor, replace `encoded_data` with `processed_data`.
3. In the `__getitem__` method, **uncomment** the normalization lines:
   - TMRM: Clip and normalize by 25,000
   - MitoTracker: Clip and normalize by 10,000

Run the generation script:

```bash
python -m generate_space
```

### Optional Arguments

- `--save_embeddings` (True/False): Whether to save the embeddings.
- `--visualise_space` (True/False): Whether to visualize the space using Open3D.
