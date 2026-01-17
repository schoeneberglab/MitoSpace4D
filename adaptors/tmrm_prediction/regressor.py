import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# No other changes were made to imports or model architecture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
from tqdm import trange


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


def train_model(embeddings, targets, epochs=100, batch_size=32, lr=1e-3, hidden_dim=256, dropout_rate=0.3,
                test_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, targets, test_size=test_split, random_state=42
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

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    best_val_loss = float('inf')

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
            torch.save(model.state_dict(), '20260117_3DMS_tmrm_regressor_kinetics-raw/best_regressor.pth')

    joblib.dump(target_scaler, '20260117_3DMS_tmrm_regressor_kinetics-raw/target_scaler.pkl')
    return model, target_scaler, X_val, y_val


def evaluate_and_plot(model, X_val, y_val, target_scaler):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = torch.FloatTensor(X_val).to(device)
        preds_scaled = model(inputs).cpu().numpy()

    preds_orig = target_scaler.inverse_transform(preds_scaled).flatten()

    r2 = r2_score(y_val, preds_orig)
    mean_absolute_error = np.abs(y_val - preds_orig)
    mae = mean_absolute_error.mean()
    std = mean_absolute_error.std()

    print("\n--- Final Performance Metrics ---")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f} ± {std:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_val, y=preds_orig, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.xlabel('Actual Normalized Intensity')
    plt.ylabel('Predicted Normalized Intensity')
    plt.title(f'TMRM Prediction Results (R²: {r2:.4f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('prediction_results.png')
    plt.show()

    return r2, mae


if __name__ == "__main__":
    # embeddings = np.load("/home/earkfeld/Projects/MitoSpace4D/adaptors/tmrm_prediction/2024v2_embeddings.npy")
    # embeddings = np.load("/home/earkfeld/Projects/MitoSpace4D/adaptors/tmrm_prediction/2024v2_embeddings_resnet.npy")
    # embeddings = np.load("/home/earkfeld/Projects/MitoSpace4D/runs/20260112_2024v2-all_embeddings_resnet3d-kinetics-300eps_ablated-tmrm/embeddings_raw.npy")
    # tmrm_intensities = np.load("/home/earkfeld/Projects/MitoSpace4D/adaptors/tmrm_prediction/2024v2_tmrm_otsu.npy")

    # 3D Kinetics Embeddings (Raw Data)
    embeddings = np.load("/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm/embeddings_raw.npy")
    tmrm_intensities = np.load("/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm/tmrm_intensities.npy")

    # embeddings = embeddings[:, -1, :]
    embeddings = np.mean(embeddings, axis=1)

    # tmrm_intensities = tmrm_intensities[:, -1]
    tmrm_intensities = np.mean(tmrm_intensities, axis=-1)
    tmrm_intensities = np.log1p(tmrm_intensities)
    #
    # # min/max normalization of TMRM intensities
    tmrm_intensities = (tmrm_intensities - tmrm_intensities.min()) / (tmrm_intensities.max() - tmrm_intensities.min())

    model, scaler, X_val, y_val = train_model(embeddings,
                                              tmrm_intensities,
                                              epochs=200,
                                              batch_size=2048,
                                              # batch_size=512,
                                              lr=1e-3,
                                              hidden_dim=1048,
                                              dropout_rate=0.2,
                                              test_split=0.1)

    print("Model trained successfully!")
    evaluate_and_plot(model, X_val, y_val, scaler)