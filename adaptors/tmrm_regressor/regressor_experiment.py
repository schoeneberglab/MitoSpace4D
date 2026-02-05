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
from sklearn.metrics import r2_score
import joblib
from tqdm import trange
import os.path as osp


class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_prob=0.3):
        super(Regressor, self).__init__()

        self.regressor = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            # Output Layer
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.regressor(x)


def train_model(embeddings, targets, labels, epochs=100, batch_size=32, lr=1e-3, hidden_dim=256, dropout_rate=0.3,
                test_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Split data (now flattened by time)
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        embeddings, targets, labels, test_size=test_split, random_state=42
    )

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_scaled))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = Regressor(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout_prob=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    best_val_loss = float('inf')

    # Ensure directory exists
    save_dir = '20260117_3DMS_tmrm_regressor_kinetics-raw'
    if not osp.exists(save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)

    for epoch in trange(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += criterion(v_out, vy).item() * vx.size(0)

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
        preds_scaled = model(inputs).cpu().numpy()

    preds_orig = target_scaler.inverse_transform(preds_scaled).flatten()

    # --- Global Metrics ---
    r2_global = r2_score(y_val, preds_orig)
    mae_global = np.mean(np.abs(y_val - preds_orig))
    std_global = np.std(np.abs(y_val - preds_orig))

    print("\n" + "=" * 40)
    print("      FINAL PERFORMANCE METRICS")
    print("=" * 40)
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
        preds_subset = preds_orig[mask]

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
    sns.regplot(x=y_val, y=preds_orig, scatter_kws={'alpha': 0.1, 's': 2}, line_kws={'color': 'red'})
    plt.xlabel('Actual Normalized Intensity')
    plt.ylabel('Predicted Normalized Intensity')
    plt.title(f'TMRM Prediction Results (Global R²: {r2_global:.4f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('prediction_results.png')
    plt.show()

    return r2_global, mae_global


if __name__ == "__main__":
    # Settings
    balance_classes = True
    pick_labels = [6]

    # 2024v2 3D
    workdir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"

    embeddings = np.load(osp.join(workdir, "embeddings_raw.npy"))
    tmrm_intensities = np.load(osp.join(workdir, "tmrm_intensities.npy"))
    labels = np.load(osp.join(workdir, "labels.npy"))
    label_names = np.load(osp.join(workdir, "label_names.npy"))

    if pick_labels:
        print("Filtering for specific labels:", pick_labels)
        mask = np.isin(labels, pick_labels)
        embeddings = embeddings[mask]
        tmrm_intensities = tmrm_intensities[mask]
        labels = labels[mask]

    print(f"Original Shapes -> Embeddings: {embeddings.shape}, TMRM: {tmrm_intensities.shape}, Labels: {labels.shape}")

    # --- Reshaping (Folding Time into Batch) ---
    N, T, D = embeddings.shape

    # 1. Flatten embeddings: (N, T, D) -> (N*T, D)
    embeddings = embeddings.reshape(N * T, D)

    # 2. Flatten intensities: (N, T) -> (N*T)
    tmrm_intensities = tmrm_intensities.reshape(N * T)

    # 3. Repeat labels for each timepoint: (N) -> (N*T)
    labels = np.repeat(labels, T)

    print(f"Folded Shapes   -> Embeddings: {embeddings.shape}, TMRM: {tmrm_intensities.shape}, Labels: {labels.shape}")

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
        print(f"Balanced Shapes -> Embeddings: {embeddings.shape}, Labels: {labels.shape}\n")

    # Min/Max normalization of TMRM intensities
    if tmrm_intensities.max() - tmrm_intensities.min() != 0:
        tmrm_intensities = (tmrm_intensities - tmrm_intensities.min()) / (
                    tmrm_intensities.max() - tmrm_intensities.min())
    tmrm_intensities = np.log1p(tmrm_intensities)

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