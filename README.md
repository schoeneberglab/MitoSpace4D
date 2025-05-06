# MitoSpace4D: Training Pipeline

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

---
