import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
import joblib
from tqdm import trange
import os.path as osp


class HurdleRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_prob=0.3):
        super(HurdleRegressor, self).__init__()

        # Shared feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )

        # Head 1: Regression (Predicts value IF > 0)
        self.regressor_head = nn.Linear(hidden_dim // 2, 1)

        # Head 2: Classifier (Predicts Probability of being > 0)
        self.classifier_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        features = self.encoder(x)
        val_pred = self.regressor_head(features)
        cls_logits = self.classifier_head(features)
        return val_pred, cls_logits


def train_model(embeddings, targets, labels, epochs=100, batch_size=32, lr=1e-3, hidden_dim=256, dropout_rate=0.3,
                test_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Generate Binary Targets (0 if zero, 1 if > 0)
    # Using a small epsilon for float safety
    targets_binary = (targets > 1e-6).astype(float)

    # Split data
    X_train, X_val, y_train, y_val, y_bin_train, y_bin_val, labels_train, labels_val = train_test_split(
        embeddings, targets, targets_binary, labels, test_size=test_split, random_state=42
    )

    # Scale the continuous targets (we fit on all data, but only regress on positives later)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))

    # Create Datasets (Now including the binary target)
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(y_bin_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(y_bin_val)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = HurdleRegressor(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout_prob=dropout_rate
    ).to(device)

    # Two loss functions
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    best_val_loss = float('inf')

    save_dir = '../20260117_3DMS_tmrm_regressor_kinetics-raw'
    if not osp.exists(save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)

    for epoch in trange(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y_val, batch_y_bin in train_loader:
            batch_x = batch_x.to(device)
            batch_y_val = batch_y_val.to(device)
            batch_y_bin = batch_y_bin.to(device).unsqueeze(1)  # [Batch, 1]

            optimizer.zero_grad()

            val_preds, cls_logits = model(batch_x)

            # 1. Classification Loss (All data)
            loss_cls = bce_criterion(cls_logits, batch_y_bin)

            # 2. Regression Loss (Only where target > 0)
            mask = batch_y_bin.bool()
            if mask.sum() > 0:
                loss_reg = mse_criterion(val_preds[mask], batch_y_val[mask])
            else:
                loss_reg = 0.0

            # Combined Loss
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy_val, vy_bin in val_loader:
                vx = vx.to(device)
                vy_val = vy_val.to(device)
                vy_bin = vy_bin.to(device).unsqueeze(1)

                v_out, c_out = model(vx)

                l_cls = bce_criterion(c_out, vy_bin)

                mask = vy_bin.bool()
                if mask.sum() > 0:
                    l_reg = mse_criterion(v_out[mask], vy_val[mask])
                else:
                    l_reg = 0.0

                val_loss += (l_cls + l_reg).item() * vx.size(0)

        avg_train_loss = train_loss / len(X_train)
        avg_val_loss = val_loss / len(X_val)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), osp.join(save_dir, 'best_regressor.pth'))

    joblib.dump(target_scaler, osp.join(save_dir, 'target_scaler.pkl'))
    return model, target_scaler, X_val, y_val, labels_val


def evaluate_and_plot(model, X_val, y_val, val_labels, label_names, target_scaler):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = torch.FloatTensor(X_val).to(device)
        # Get both heads
        pred_vals_scaled, pred_logits = model(inputs)

        # CPU conversion
        pred_vals_scaled = pred_vals_scaled.cpu().numpy()
        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()

    # --- Hurdle Combination Logic ---
    # 1. Inverse transform the regression values
    pred_vals_orig = target_scaler.inverse_transform(pred_vals_scaled).flatten()

    # 2. Apply Classification Gate (Threshold 0.5)
    # If prob < 0.5, we predict exactly 0.0
    gate_mask = (pred_probs > 0.5).flatten()
    final_preds = pred_vals_orig * gate_mask

    # Force zeros where classifier said zero
    final_preds[~gate_mask] = 0.0

    # --- Global Metrics ---
    r2_global = r2_score(y_val, final_preds)
    mae_global = np.mean(np.abs(y_val - final_preds))
    std_global = np.std(np.abs(y_val - final_preds))

    # Classifier Metrics
    y_val_binary = (y_val > 1e-6).astype(int)
    pred_binary = gate_mask.astype(int)
    acc_cls = accuracy_score(y_val_binary, pred_binary)
    f1_cls = f1_score(y_val_binary, pred_binary)

    print("\n" + "=" * 40)
    print("      FINAL PERFORMANCE METRICS")
    print("=" * 40)
    print(f"Hurdle Accuracy (Zero vs Signal): {acc_cls:.4f}")
    print(f"Hurdle F1 Score: {f1_cls:.4f}")
    print("-" * 40)
    print(f"Overall R²: {r2_global:.4f}")
    print(f"Overall MAE: {mae_global:.4f} ± {std_global:.4f}")
    print("-" * 40)

    # --- Per-Condition Metrics ---
    print(f"{'Condition':<30} | {'Count':<6} | {'R²':<7} | {'MAE':<15}")
    print("-" * 65)

    unique_labels = np.unique(val_labels)
    unique_labels.sort()

    for lbl_idx in unique_labels:
        mask = (val_labels == lbl_idx)
        if np.sum(mask) < 2:
            continue

        y_subset = y_val[mask]
        preds_subset = final_preds[mask]

        r2_cond = r2_score(y_subset, preds_subset)
        mae_cond = np.mean(np.abs(y_subset - preds_subset))
        std_cond = np.std(np.abs(y_subset - preds_subset))

        try:
            cond_name = label_names[int(lbl_idx)]
        except IndexError:
            cond_name = f"Label_{int(lbl_idx)}"

        print(f"{str(cond_name):<30} | {np.sum(mask):<6} | {r2_cond:<7.4f} | {mae_cond:.4f} ± {std_cond:.4f}")

    print("=" * 40)

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_val, y=final_preds, scatter_kws={'alpha': 0.1, 's': 2}, line_kws={'color': 'red'})
    plt.xlabel('Actual Normalized Intensity')
    plt.ylabel('Predicted Normalized Intensity')
    plt.title(f'TMRM Hurdle Prediction (Global R²: {r2_global:.4f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('prediction_results.png')
    plt.show()

    return r2_global, mae_global


if __name__ == "__main__":
    # Settings
    balance_classes = False
    pick_labels = None

    # 2024v2 3D
    workdir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"

    embeddings = np.load(osp.join(workdir, "embeddings_raw.npy"))
    tmrm_intensities = np.load(osp.join(workdir, "tmrm_intensities.npy"))
    labels = np.load(osp.join(workdir, "labels.npy"))
    label_names = np.load(osp.join(workdir, "label_names.npy"))
    # Keep as numpy array for masking
    img_paths = np.loadtxt(osp.join(workdir, 'image_paths.csv'), dtype=str)

    if pick_labels:
        print("Filtering for specific labels:", pick_labels)
        mask = np.isin(labels, pick_labels)
        embeddings = embeddings[mask]
        tmrm_intensities = tmrm_intensities[mask]
        labels = labels[mask]
        img_paths = img_paths[mask]

    print(f"Original Shapes -> Embeddings: {embeddings.shape}, TMRM: {tmrm_intensities.shape}, Labels: {labels.shape}")

    # embeddings = embeddings[:, -1, :]
    embeddings = np.mean(embeddings, axis=1)

    # tmrm_intensities = tmrm_intensities[:, -1]
    tmrm_intensities = np.mean(tmrm_intensities, axis=-1)

    # --- Optional: Balance Classes ---
    if balance_classes:
        print("\n--- Balancing Classes ---")
        # Use pandas for easier grouping and sampling
        df = pd.DataFrame({'label': labels})
        min_count = df['label'].value_counts().min()
        print(f"Minimum class count found: {min_count}")

        # Sample min_count from each group
        balanced_indices = df.groupby('label').sample(n=min_count, random_state=42).index

        # Apply filter
        embeddings = embeddings[balanced_indices]
        tmrm_intensities = tmrm_intensities[balanced_indices]
        labels = labels[balanced_indices]
        img_paths = img_paths[balanced_indices]
        print(f"Balanced Shapes -> Embeddings: {embeddings.shape}, Labels: {labels.shape}\n")

    tmrm_intensities = np.log1p(tmrm_intensities)

    if tmrm_intensities.max() - tmrm_intensities.min() != 0:
        tmrm_intensities = (tmrm_intensities - tmrm_intensities.min()) / (
                tmrm_intensities.max() - tmrm_intensities.min())

    # Plot a histogram of intensities
    plt.figure(figsize=(8, 6))
    sns.histplot(data=tmrm_intensities, kde=True)
    plt.xlabel('Normalized TMRM Intensity')
    plt.ylabel('Frequency')
    plt.title('Distribution of TMRM Intensities')
    plt.show()

    # --- Print 5 Representative Images ---
    sorted_idxs = np.argsort(tmrm_intensities)
    step_indices = np.linspace(0, len(sorted_idxs) - 1, 5, dtype=int)

    print("\n" + "=" * 80)
    print("REPRESENTATIVE IMAGES ALONG TMRM DISTRIBUTION")
    print("=" * 80)
    print(f"{'Percentile':<15} | {'Norm TMRM':<12} | {'Image Path'}")
    print("-" * 80)

    labels_pct = ["0% (Min)", "25%", "50% (Median)", "75%", "100% (Max)"]
    for i, idx in enumerate(step_indices):
        real_idx = sorted_idxs[idx]
        print(f"{labels_pct[i]:<15} | {tmrm_intensities[real_idx]:.6f}     | {img_paths[real_idx]}")
    print("=" * 80 + "\n")

    print(f"TMRM Intensities Variance: {np.var(tmrm_intensities)}")

    # Pass flattened data to train_model
    model, scaler, X_val, y_val, labels_val = train_model(
        embeddings,
        tmrm_intensities,
        labels,
        epochs=200,
        batch_size=4096,
        lr=1e-3,
        hidden_dim=1048,
        dropout_rate=0.2,
        test_split=0.1
    )

    print("Model trained successfully!")

    evaluate_and_plot(model, X_val, y_val, labels_val, label_names, scaler)