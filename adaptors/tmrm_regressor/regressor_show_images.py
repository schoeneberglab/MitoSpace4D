# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score
# import joblib
# from tqdm import trange
# import os.path as osp
#
#
# class Regressor(nn.Module):
#     def __init__(self, input_dim, hidden_dim=256, dropout_prob=0.3):
#         super(Regressor, self).__init__()
#
#         self.regressor = nn.Sequential(
#             # Layer 1
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_prob),
#
#             # Layer 2
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_prob),
#
#             # Output Layer
#             nn.Linear(hidden_dim // 2, 1)
#         )
#
#     def forward(self, x):
#         return self.regressor(x)
#
#
# def train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=32, lr=1e-3,
#                 hidden_dim=256, dropout_rate=0.3):
#     """
#     Updated to accept pre-split data.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Training on: {device}")
#
#     # Scale targets (Fit on train, transform val)
#     target_scaler = StandardScaler()
#     y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
#     y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))
#
#     train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
#     val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_scaled))
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size)
#
#     model = Regressor(
#         input_dim=X_train.shape[1],
#         hidden_dim=hidden_dim,
#         dropout_prob=dropout_rate
#     ).to(device)
#
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
#
#     best_val_loss = float('inf')
#
#     # Ensure directory exists
#     save_dir = '20260117_3DMS_tmrm_regressor_kinetics-raw'
#     if not osp.exists(save_dir):
#         import os
#         os.makedirs(save_dir, exist_ok=True)
#
#     for epoch in trange(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * batch_x.size(0)
#
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for vx, vy in val_loader:
#                 vx, vy = vx.to(device), vy.to(device)
#                 v_out = model(vx)
#                 val_loss += criterion(v_out, vy).item() * vx.size(0)
#
#         avg_train_loss = train_loss / len(X_train)
#         avg_val_loss = val_loss / len(X_val)
#         scheduler.step(avg_val_loss)
#
#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
#
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), osp.join(save_dir, 'best_regressor.pth'))
#
#     joblib.dump(target_scaler, osp.join(save_dir, 'target_scaler.pkl'))
#
#     # Return model and scaler to Main
#     return model, target_scaler
#
#
# def evaluate_and_plot(model, X_val, y_val, val_labels, label_names, target_scaler):
#     model.eval()
#     device = next(model.parameters()).device
#
#     with torch.no_grad():
#         inputs = torch.FloatTensor(X_val).to(device)
#         preds_scaled = model(inputs).cpu().numpy()
#
#     preds_orig = target_scaler.inverse_transform(preds_scaled).flatten()
#
#     # --- Global Metrics ---
#     r2_global = r2_score(y_val, preds_orig)
#     mae_global = np.mean(np.abs(y_val - preds_orig))
#     std_global = np.std(np.abs(y_val - preds_orig))
#
#     print("\n" + "=" * 40)
#     print("      FINAL PERFORMANCE METRICS")
#     print("=" * 40)
#     print(f"Overall R²: {r2_global:.4f}")
#     print(f"Overall MAE: {mae_global:.4f} ± {std_global:.4f}")
#     print("-" * 40)
#
#     # --- Per-Condition Metrics ---
#     print(f"{'Condition':<30} | {'Count':<6} | {'R²':<7} | {'MAE':<15}")
#     print("-" * 65)
#
#     unique_labels = np.unique(val_labels)
#     unique_labels.sort()
#
#     for lbl_idx in unique_labels:
#         mask = (val_labels == lbl_idx)
#         if np.sum(mask) < 2:
#             continue
#
#         y_subset = y_val[mask]
#         preds_subset = preds_orig[mask]
#
#         r2_cond = r2_score(y_subset, preds_subset)
#         mae_cond = np.mean(np.abs(y_subset - preds_subset))
#         std_cond = np.std(np.abs(y_subset - preds_subset))
#
#         try:
#             cond_name = label_names[int(lbl_idx)]
#         except IndexError:
#             cond_name = f"Label_{int(lbl_idx)}"
#
#         print(f"{str(cond_name):<30} | {np.sum(mask):<6} | {r2_cond:<7.4f} | {mae_cond:.4f} ± {std_cond:.4f}")
#
#     print("=" * 40)
#
#     # Visualization
#     plt.figure(figsize=(8, 6))
#     sns.regplot(x=y_val, y=preds_orig, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
#     plt.xlabel('Actual Normalized Intensity')
#     plt.ylabel('Predicted Normalized Intensity')
#     plt.title(f'TMRM Prediction Results (Global R²: {r2_global:.4f})')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.savefig('prediction_results.png')
#     plt.show()
#
#     return r2_global, mae_global
#
#
# if __name__ == "__main__":
#     # Settings
#     balance_classes = False
#     pick_labels = None
#     test_split_ratio = 0.1  # Set here so it matches everywhere
#
#     # 2024v2 3D
#     workdir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"
#
#     embeddings = np.load(osp.join(workdir, "embeddings_raw.npy"))
#     tmrm_intensities = np.load(osp.join(workdir, "tmrm_intensities.npy"))
#     labels = np.load(osp.join(workdir, "labels.npy"))
#     label_names = np.load(osp.join(workdir, "label_names.npy"))
#     # Keep as numpy array for masking
#     img_paths = np.loadtxt(osp.join(workdir, 'image_paths.csv'), dtype=str)
#
#     if pick_labels:
#         print("Filtering for specific labels:", pick_labels)
#         mask = np.isin(labels, pick_labels)
#         embeddings = embeddings[mask]
#         tmrm_intensities = tmrm_intensities[mask]
#         labels = labels[mask]
#         img_paths = img_paths[mask]
#
#     print(f"Original Shapes -> Embeddings: {embeddings.shape}, TMRM: {tmrm_intensities.shape}, Labels: {labels.shape}")
#
#     # Average embeddings and intensities over time
#     embeddings = np.mean(embeddings, axis=1)
#     tmrm_intensities = np.mean(tmrm_intensities, axis=-1)
#
#     # --- Optional: Balance Classes ---
#     if balance_classes:
#         print("\n--- Balancing Classes ---")
#         df = pd.DataFrame({'label': labels})
#         min_count = df['label'].value_counts().min()
#         print(f"Minimum class count found: {min_count}")
#
#         balanced_indices = df.groupby('label').sample(n=min_count, random_state=42).index
#
#         embeddings = embeddings[balanced_indices]
#         tmrm_intensities = tmrm_intensities[balanced_indices]
#         labels = labels[balanced_indices]
#         img_paths = img_paths[balanced_indices]
#         print(f"Balanced Shapes -> Embeddings: {embeddings.shape}, Labels: {labels.shape}\n")
#
#     # Processing Intensities
#     tmrm_intensities = np.log1p(tmrm_intensities)
#
#     if tmrm_intensities.max() - tmrm_intensities.min() != 0:
#         tmrm_intensities = (tmrm_intensities - tmrm_intensities.min()) / (
#                 tmrm_intensities.max() - tmrm_intensities.min())
#
#     # --- SPLIT DATA BEFORE VISUALIZATION ---
#     print("\nSplitting Data into Train and Validation...")
#     # Include img_paths in the split so we can visualize VALIDATION images specifically
#     X_train, X_val, y_train, y_val, labels_train, labels_val, paths_train, paths_val = train_test_split(
#         embeddings,
#         tmrm_intensities,
#         labels,
#         img_paths,
#         test_size=test_split_ratio,
#         random_state=37
#     )
#     print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
#
#     # Plot a histogram of validation intensities
#     plt.figure(figsize=(8, 6))
#     sns.histplot(data=y_val, kde=True)
#     plt.xlabel('Normalized TMRM Intensity (Validation Set)')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of TMRM Intensities (Val)')
#     plt.show()
#
#     # --- Visualize 5 Representative Images (FROM VALIDATION SET) ---
#     print("\n" + "=" * 80)
#     print("GENERATING REPRESENTATIVE IMAGE PLOTS (VALIDATION SET ONLY)")
#     print("=" * 80)
#
#     # Sort based on Validation Targets (y_val)
#     sorted_idxs = np.argsort(y_val)
#     step_indices = np.linspace(0, len(sorted_idxs) - 1, 8, dtype=int)
#     # labels_pct = ["0% (Min)", "25%", "50% (Median)", "75%", "100% (Max)"]
#
#     f, axarr = plt.subplots(2, 8, figsize=(40, 8))
#     f.suptitle("Representative Images along TMRM Distribution (Validation Set)", fontsize=24)
#
#     for i, idx in enumerate(step_indices):
#         real_idx = sorted_idxs[idx]
#
#         # Access from VALIDATION arrays
#         path = paths_val[real_idx]
#         val = y_val[real_idx]
#         lbl_idx = labels_val[real_idx]
#
#         try:
#             lbl_name = label_names[int(lbl_idx)]
#         except:
#             lbl_name = str(lbl_idx)
#
#         # print(f"Loading {labels_pct[i]}: {path}")
#
#         try:
#             # Load the .npy file
#             img_4d = np.load(path)
#             img_4d = img_4d[:, -1, ...].max(axis=1)  # MIP last frame
#             # --- Identify Channels ---
#             # Assuming format (C, H, W).
#             if img_4d.shape[0] > 1:
#                 img_tmrm = img_4d[0]  # Channel 0
#                 img_mito = img_4d[1]  # Channel 1
#             else:
#                 img_tmrm = img_4d[0]
#                 img_mito = np.zeros_like(img_tmrm)
#
#             # --- Plot TMRM (Top Row) ---
#             # axarr[0, i].imshow(img_tmrm, cmap='hot', vmin=0, vmax=np.percentile(img_tmrm, 99.5))
#             axarr[0, i].imshow(img_tmrm, cmap='hot', vmin=0, vmax=np.percentile(img_tmrm, 99.5))
#             axarr[0, i].set_title(f"\nVal: {val:.4f}", fontsize=14)
#             if i == 0:
#                 axarr[0, i].set_ylabel("TMRM Channel", fontsize=16)
#             axarr[0, i].set_xticks([])
#             axarr[0, i].set_yticks([])
#
#             # --- Plot Mito (Bottom Row) ---
#             axarr[1, i].imshow(img_mito, cmap='viridis', vmin=0, vmax=np.percentile(img_mito, 99.5))
#             axarr[1, i].set_title(f"{lbl_name}", fontsize=12)
#             if i == 0:
#                 axarr[1, i].set_ylabel("Mito Channel", fontsize=16)
#             axarr[1, i].set_xticks([])
#             axarr[1, i].set_yticks([])
#
#         except Exception as e:
#             print(f"Error loading or plotting {path}: {e}")
#             axarr[0, i].text(0.5, 0.5, "Error Loading", ha='center')
#             axarr[1, i].text(0.5, 0.5, "Image", ha='center')
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig('representative_images_viz.png')
#     plt.show()
#     print("=" * 80 + "\n")
#
#     print(f"TMRM Intensities Variance (Val): {np.var(y_val)}")
#
#     # Pass PRE-SPLIT data to train_model
#     model, scaler = train_model(
#         X_train, X_val,
#         y_train, y_val,
#         epochs=200,
#         batch_size=4096,
#         lr=1e-3,
#         hidden_dim=1048,
#         dropout_rate=0.2
#     )
#
#     print("Model trained successfully!")
#
#     evaluate_and_plot(model, X_val, y_val, labels_val, label_names, scaler)

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
from skimage.filters import threshold_otsu

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


def train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=32, lr=1e-3,
                hidden_dim=256, dropout_rate=0.3):
    """
    Updated to accept pre-split data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Scale targets (Fit on train, transform val)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_scaled))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = Regressor(
        input_dim=X_train.shape[1],
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

    # Return model and scaler to Main
    return model, target_scaler


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
    sns.regplot(x=y_val, y=preds_orig, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.xlabel('Actual Normalized Intensity')
    plt.ylabel('Predicted Normalized Intensity')
    plt.title(f'TMRM Prediction Results (Global R²: {r2_global:.4f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('prediction_results.png')
    plt.show()

    return r2_global, mae_global


if __name__ == "__main__":
    # Settings
    balance_classes = False
    pick_labels = None
    test_split_ratio = 0.1  # Set here so it matches everywhere

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

    # Average embeddings and intensities over time
    embeddings = np.mean(embeddings, axis=1)
    tmrm_intensities = np.mean(tmrm_intensities, axis=-1)

    # --- Optional: Balance Classes ---
    if balance_classes:
        print("\n--- Balancing Classes ---")
        df = pd.DataFrame({'label': labels})
        min_count = df['label'].value_counts().min()
        print(f"Minimum class count found: {min_count}")

        balanced_indices = df.groupby('label').sample(n=min_count, random_state=42).index

        embeddings = embeddings[balanced_indices]
        tmrm_intensities = tmrm_intensities[balanced_indices]
        labels = labels[balanced_indices]
        img_paths = img_paths[balanced_indices]
        print(f"Balanced Shapes -> Embeddings: {embeddings.shape}, Labels: {labels.shape}\n")

    # Processing Intensities
    tmrm_intensities = np.log1p(tmrm_intensities)

    if tmrm_intensities.max() - tmrm_intensities.min() != 0:
        tmrm_intensities = (tmrm_intensities - tmrm_intensities.min()) / (
                tmrm_intensities.max() - tmrm_intensities.min())

    # --- SPLIT DATA BEFORE VISUALIZATION ---
    print("\nSplitting Data into Train and Validation...")
    # Include img_paths in the split so we can visualize VALIDATION images specifically
    X_train, X_val, y_train, y_val, labels_train, labels_val, paths_train, paths_val = train_test_split(
        embeddings,
        tmrm_intensities,
        labels,
        img_paths,
        test_size=test_split_ratio,
        random_state=1123
    )
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")

    # Plot a histogram of validation intensities
    plt.figure(figsize=(8, 6))
    sns.histplot(data=y_val, kde=True)
    plt.xlabel('Normalized TMRM Intensity (Validation Set)')
    plt.ylabel('Frequency')
    plt.title('Distribution of TMRM Intensities (Val)')
    plt.show()

    # --- Visualize 5 Representative Images (FROM VALIDATION SET) ---
    print("\n" + "=" * 80)
    print("GENERATING REPRESENTATIVE IMAGE PLOTS (VALIDATION SET ONLY)")
    print("=" * 80)

    # Sort based on Validation Targets (y_val)
    sorted_idxs = np.argsort(y_val)
    step_indices = np.linspace(0, len(sorted_idxs) - 1, 8, dtype=int)

    # --- PHASE 1: Load images and find GLOBAL Max intensity ---
    loaded_images = []
    all_tmrm_pixels = []

    for idx in step_indices:
        real_idx = sorted_idxs[idx]
        path = paths_val[real_idx]
        val = y_val[real_idx]
        lbl_idx = labels_val[real_idx]

        try:
            lbl_name = label_names[int(lbl_idx)]
        except:
            lbl_name = str(lbl_idx)

        data_packet = {
            "path": path,
            "val": val,
            "lbl_name": lbl_name,
            "img_tmrm": None,
            "img_mito": None,
            "error": False
        }

        try:
            # Load the .npy file
            img_4d = np.load(path)
            img_4d = img_4d[:, 0, ...].max(axis=1)  # MIP last frame

            # --- Identify Channels ---
            if img_4d.shape[0] > 1:
                img_tmrm = img_4d[0]  # Channel 0
                img_mito = img_4d[1]  # Channel 1
            else:
                img_tmrm = img_4d[0]
                img_mito = np.zeros_like(img_tmrm)

            # otsu threshold the mito image
            thr = threshold_otsu(img_mito)
            mask = img_mito > thr

            data_packet["img_tmrm"] = img_tmrm
            data_packet["img_mito"] = img_mito

            # Collect pixels for max calculation
            all_tmrm_pixels.append(img_tmrm[mask].mean())

        except Exception as e:
            print(f"Error loading {path}: {e}")
            data_packet["error"] = True

        loaded_images.append(data_packet)

    # Calculate Global Maximum across all loaded TMRM images
    if len(all_tmrm_pixels) > 0:
        global_tmrm_max = np.max(all_tmrm_pixels)
        print(f"Global TMRM Max for visualization: {global_tmrm_max:.4f}")
    else:
        global_tmrm_max = 1.0

    # --- PHASE 2: Plotting ---
    f, axarr = plt.subplots(2, 8, figsize=(40, 8))
    f.suptitle("Representative Images along TMRM Distribution (Validation Set)", fontsize=24)

    for i, data in enumerate(loaded_images):
        if not data["error"]:
            img_tmrm = data["img_tmrm"]
            img_mito = data["img_mito"]
            val = data["val"]
            lbl_name = data["lbl_name"]

            # --- Plot TMRM (Top Row) - NORMALIZED BY GLOBAL MAX ---
            # Using global_tmrm_max for vmax ensures relative brightness is preserved
            axarr[0, i].imshow(img_tmrm, cmap='hot')
            axarr[0, i].set_title(f"\nVal: {val:.4f}", fontsize=14)
            if i == 0:
                axarr[0, i].set_ylabel("TMRM Channel", fontsize=16)
            axarr[0, i].set_xticks([])
            axarr[0, i].set_yticks([])

            # --- Plot Mito (Bottom Row) - Keep individual normalization for structure ---
            # Structural channel usually benefits from individual scaling to see morphology
            axarr[1, i].imshow(img_mito, cmap='viridis', vmin=0, vmax=np.percentile(img_mito, 99.5))
            axarr[1, i].set_title(f"{lbl_name}", fontsize=12)
            if i == 0:
                axarr[1, i].set_ylabel("Mito Channel", fontsize=16)
            axarr[1, i].set_xticks([])
            axarr[1, i].set_yticks([])
        else:
            axarr[0, i].text(0.5, 0.5, "Error Loading", ha='center')
            axarr[1, i].text(0.5, 0.5, "Image", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('representative_images_viz.png')
    plt.show()
    print("=" * 80 + "\n")

    print(f"TMRM Intensities Variance (Val): {np.var(y_val)}")

    # Pass PRE-SPLIT data to train_model
    model, scaler = train_model(
        X_train, X_val,
        y_train, y_val,
        epochs=200,
        batch_size=4096,
        lr=1e-3,
        hidden_dim=1048,
        dropout_rate=0.2
    )

    print("Model trained successfully!")

    evaluate_and_plot(model, X_val, y_val, labels_val, label_names, scaler)