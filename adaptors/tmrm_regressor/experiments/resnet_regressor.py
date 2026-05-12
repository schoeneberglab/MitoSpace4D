import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
from tqdm import trange


# --- IMPROVEMENT 1: Residual Block Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),  # GELU often outperforms ReLU in deep networks
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # The "skip connection": x + block(x)
        return self.activation(x + self.block(x))


class ResNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_prob=0.3, num_blocks=3):
        super(ResNetRegressor, self).__init__()

        # Project input to hidden dimension
        self.embedding_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # Stack Residual Blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_prob) for _ in range(num_blocks)
        ])

        # Final Output Head
        self.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.embedding_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def train_model(embeddings, targets, epochs=100, batch_size=32, lr=1e-3,
                hidden_dim=512, dropout_rate=0.3, test_split=0.1):
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

    # Initialize the new ResNet model
    model = ResNetRegressor(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout_prob=dropout_rate,
        num_blocks=3  # Depth of network
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # Slightly higher decay for ResNet

    # --- IMPROVEMENT 3: OneCycleLR Scheduler ---
    # This scheduler updates the LR every batch, not every epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3  # Spend 30% of time warming up
    )

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
            scheduler.step()  # Step per batch for OneCycle
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

        # Note: No scheduler.step() here, it's done in the batch loop

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '../20260117_3DMS_tmrm_regressor_kinetics-raw/best_regressor.pth')

    joblib.dump(target_scaler, '../20260117_3DMS_tmrm_regressor_kinetics-raw/target_scaler.pkl')
    return model, target_scaler, X_val, y_val


def evaluate_and_plot(model, X_val, y_val, target_scaler):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = torch.FloatTensor(X_val).to(device)
        preds_scaled = model(inputs).cpu().numpy()

    preds_orig = target_scaler.inverse_transform(preds_scaled).flatten()

    r2 = r2_score(y_val, preds_orig)
    mape = np.mean(np.abs((y_val - preds_orig) / (y_val + 1e-10))) * 100

    print("\n--- Final Performance Metrics ---")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_val, y=preds_orig, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.xlabel('Actual Intensity (Log-Transformed)')
    plt.ylabel('Predicted Intensity')
    plt.title(f'Prediction Results (R²: {r2:.4f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('prediction_results.png')
    plt.show()

    return r2, mape


if __name__ == "__main__":
    embeddings = np.load("/adaptors/tmrm_regressor/2024v2_embeddings.npy")
    tmrm_intensities = np.load("/adaptors/tmrm_regressor/2024v2_tmrm_otsu.npy")

    # --- IMPROVEMENT 2: Global Average Pooling ---
    # Instead of taking just the last frame [-1], we average across time [axis 1]
    # If shape is (N, T, D), this becomes (N, D)
    print(f"Original shape: {embeddings.shape}")
    embeddings = np.mean(embeddings, axis=1)
    print(f"Pooled shape: {embeddings.shape}")

    tmrm_intensities = tmrm_intensities[:, -1]
    # tmrm_intensities = np.mean(tmrm_intensities, axis=-1)
    tmrm_intensities = np.log1p(tmrm_intensities)

    # Note: Increased batch size and epochs for OneCycle convergence
    model, scaler, X_val, y_val = train_model(embeddings,
                                              tmrm_intensities,
                                              epochs=200,
                                              batch_size=512,
                                              lr=3e-3,  # OneCycle allows higher max LRs
                                              hidden_dim=512,
                                              dropout_rate=0.2,  # Lower dropout for ResNets
                                              test_split=0.1)

    print("Model trained successfully!")
    evaluate_and_plot(model, X_val, y_val, scaler)