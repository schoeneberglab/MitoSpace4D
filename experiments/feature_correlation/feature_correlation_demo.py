import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from sklearn.decomposition import PCA


# --- Helper: Vector ACF for (N, T, D) ---
def compute_batch_vector_acf(X, max_lag=10):
    """
    X: Shape (N, T, D)
    Computes cosine similarity averaged across all N samples and T-lag time steps.
    """
    N, T, D = X.shape

    # 1. Normalize vectors (L2 norm) so we measure Direction, not Magnitude
    norms = np.linalg.norm(X, axis=2, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X / norms

    acf_means = []
    acf_stds = []  # To show variance across the N samples

    for lag in range(max_lag + 1):
        if lag == 0:
            # Correlation with self is always 1
            acf_means.append(1.0)
            acf_stds.append(0.0)
        else:
            # Vectorized slice:
            # v1: All samples, from t=0 to t=End-lag
            # v2: All samples, from t=lag to t=End
            v1 = X_norm[:, :-lag, :]  # Shape (N, T-lag, D)
            v2 = X_norm[:, lag:, :]  # Shape (N, T-lag, D)

            # Dot product along feature dim (D) -> Shape (N, T-lag)
            dots = np.sum(v1 * v2, axis=2)

            # Average over time first (get one score per sample)
            sample_corrs = np.mean(dots, axis=1)  # Shape (N,)

            # Record stats across the batch (N)
            acf_means.append(np.mean(sample_corrs))
            acf_stds.append(np.std(sample_corrs))

    return np.array(acf_means), np.array(acf_stds)


# --- Helper: Batch PACF via PCA ---
def compute_batch_pacf(X, max_lag=8):
    """
    Since PACF is univariate, we:
    1. Flatten X to (N*T, D) and run PCA to find the dominant signal (PC1).
    2. Reshape PC1 back to (N, T).
    3. Compute PACF for each of the N samples individually.
    4. Average the PACF curves.
    """
    N, T, D = X.shape

    # 1. PCA to reduce D=2048 to D=1
    X_flat = X.reshape(-1, D)
    pca = PCA(n_components=1)
    # Fit on the flattened data to find global patterns
    pc1_flat = pca.fit_transform(X_flat)
    pc1 = pc1_flat.reshape(N, T)

    # 2. Compute PACF for each sample
    batch_pacfs = []
    for i in range(N):
        # We must limit lags because T=20 is short
        # method='ywm' is often more stable for short series
        try:
            p = pacf(pc1[i], nlags=max_lag, method='ywm')
            batch_pacfs.append(p)
        except:
            pass  # Skip samples if convergence fails (rare)

    if not batch_pacfs:
        return np.zeros(max_lag + 1), np.zeros(max_lag + 1)

    batch_pacfs = np.array(batch_pacfs)  # Shape (N, lags+1)

    return np.mean(batch_pacfs, axis=0), np.std(batch_pacfs, axis=0)


# --- Helper: Plotting ---
def plot_batch_acf(means, stds, ax, title="ACF"):
    lags = len(means)
    # Plot Mean
    ax.vlines(range(lags), 0, means, colors='tab:blue', linewidth=2)
    ax.scatter(range(lags), means, color='tab:blue', s=30, zorder=2)

    # Plot Standard Deviation (Shaded region)
    ax.fill_between(range(lags), means - stds, means + stds, color='tab:blue', alpha=0.15, label='Std Dev (across N)')

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(title)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks(range(lags))
    ax.grid(axis='y', linestyle='--', alpha=0.3)


# ---------------------------------------------------------
# 1. Generate Synthetic Data (N, T=20, D=2048)
# ---------------------------------------------------------
np.random.seed(42)
N = 100
T = 20
D = 2048

# Case A: High-Dim White Noise
data_noise = np.random.normal(0, 1, size=(N, T, D))

# Case B: High-Dim Random Walk
# We generate a walk, then reshape to (N, T, D)
# Note: With short T, the "walk" aspect is subtle but detectable
walks = []
for _ in range(N):
    # Create a random walk in D dimensions
    w = np.cumsum(np.random.normal(0, 1, size=(T, D)), axis=0)
    walks.append(w)
data_walk = np.array(walks)

# ---------------------------------------------------------
# 2. Run Analysis
# ---------------------------------------------------------
L_max = 9  # Max lag < T/2 is safest

# Noise Analysis
acf_mean_n, acf_std_n = compute_batch_vector_acf(data_noise, max_lag=L_max)
pacf_mean_n, pacf_std_n = compute_batch_pacf(data_noise, max_lag=L_max)

# Walk Analysis
acf_mean_w, acf_std_w = compute_batch_vector_acf(data_walk, max_lag=L_max)
pacf_mean_w, pacf_std_w = compute_batch_pacf(data_walk, max_lag=L_max)

# ---------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
plt.subplots_adjust(hspace=0.4)

# Row 1: White Noise
plot_batch_acf(acf_mean_n, acf_std_n, axes[0, 0], title=f"Vector ACF (White Noise)\nAvg across N={N}")
plot_batch_acf(pacf_mean_n, pacf_std_n, axes[0, 1], title=f"PACF of PC1 (White Noise)\nAvg across N={N}")

# Row 2: Random Walk
plot_batch_acf(acf_mean_w, acf_std_w, axes[1, 0], title=f"Vector ACF (Random Walk)\nAvg across N={N}")
plot_batch_acf(pacf_mean_w, pacf_std_w, axes[1, 1], title=f"PACF of PC1 (Random Walk)\nAvg across N={N}")

plt.suptitle(f"Autocorrelation Analysis for (N={N}, T={T}, D={D})", fontsize=14)
plt.show()