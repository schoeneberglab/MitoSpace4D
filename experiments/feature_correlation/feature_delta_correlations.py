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

    # Normalize vectors (L2 norm) so we measure Direction, not Magnitude
    norms = np.linalg.norm(X, axis=2, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X / norms

    acf_means = []
    acf_stds = []

    for lag in range(max_lag + 1):
        if lag == 0:
            acf_means.append(1.0)
            acf_stds.append(0.0)
        else:
            # Vectorized slice
            v1 = X_norm[:, :-lag, :]
            v2 = X_norm[:, lag:, :]

            # Dot product along feature dim (D) -> Shape (N, T-lag)
            dots = np.sum(v1 * v2, axis=2)

            # Average over time first (get one score per sample)
            sample_corrs = np.mean(dots, axis=1)  # Shape (N,)

            # Record stats across the batch
            acf_means.append(np.mean(sample_corrs))
            acf_stds.append(np.std(sample_corrs))

    return np.array(acf_means), np.array(acf_stds)


# --- Helper: Multi-PC PACF ---
def compute_multi_pc_pacf(X, n_components=3, max_lag=8):
    """
    1. PCA on the flattened data to find top 'n_components' modes.
    2. Compute PACF for each component separately.
    3. Return means and stds for each PC.
    """
    N, T, D = X.shape

    # 1. PCA: Flatten (N, T, D) -> (N*T, D)
    X_flat = X.reshape(-1, D)

    # Fit PCA
    pca = PCA(n_components=n_components)
    components_flat = pca.fit_transform(X_flat)  # Shape (N*T, n_comps)

    # Reshape back to (N, T, n_comps)
    components = components_flat.reshape(N, T, n_components)

    # Dictionary to store results
    results = {}

    # 2. Loop through each PC
    for c in range(n_components):
        pc_series = components[:, :, c]  # Shape (N, T)

        batch_pacfs = []
        for i in range(N):
            # Calculate PACF for this specific sample and this specific PC
            try:
                # method='ywm' is robust for short time series (T=20)
                p = pacf(pc_series[i], nlags=max_lag, method='ywm')
                batch_pacfs.append(p)
            except:
                pass

        batch_pacfs = np.array(batch_pacfs)

        # Store Mean and Std for this component
        results[f'PC{c + 1}'] = {
            'mean': np.mean(batch_pacfs, axis=0),
            'std': np.std(batch_pacfs, axis=0),
            'explained_var': pca.explained_variance_ratio_[c]
        }

    return results


# --- Plotting Helper ---
def plot_pc_grid(pacf_results, acf_mean, acf_std, title_prefix):
    pcs = list(pacf_results.keys())
    n_pcs = len(pcs)

    # Create grid: 1 extra slot for the Vector ACF
    fig, axes = plt.subplots(1, n_pcs + 1, figsize=(4 * (n_pcs + 1), 4), sharey=True)

    # --- Plot 1: Overall Vector ACF ---
    ax_acf = axes[0]
    lags_acf = len(acf_mean)
    ax_acf.vlines(range(lags_acf), 0, acf_mean, colors='tab:blue', linewidth=2)
    ax_acf.scatter(range(lags_acf), acf_mean, color='tab:blue', s=30, zorder=2)
    ax_acf.fill_between(range(lags_acf), acf_mean - acf_std, acf_mean + acf_std, color='tab:blue', alpha=0.15)
    ax_acf.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax_acf.set_title("Total Vector ACF\n(Directional Consistency)")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Correlation")
    ax_acf.set_ylim(-1.1, 1.1)
    ax_acf.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Plot 2..N: Component-wise PACF ---
    for idx, pc_name in enumerate(pcs):
        ax = axes[idx + 1]  # Offset by 1
        data = pacf_results[pc_name]
        means = data['mean']
        stds = data['std']
        lags = len(means)

        # Plot bars and dots
        ax.vlines(range(lags), 0, means, colors='tab:red', linewidth=2)
        ax.scatter(range(lags), means, color='tab:red', s=30, zorder=2)

        # Plot Std Dev Shade
        ax.fill_between(range(lags), means - stds, means + stds, color='red', alpha=0.1)

        # Styling
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        # 95% confidence interval approximation line (for N samples)
        # Note: We use (T-1) here because velocities have 1 fewer time step
        conf = 1.96 / np.sqrt(N * (T - 1))
        ax.axhline(conf, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(-conf, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(f"{pc_name}\n(Expl. Var: {data['explained_var']:.1%})")
        ax.set_xlabel("Lag")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle(f"{title_prefix} - Dynamics Analysis", fontsize=14, y=1.05)
    plt.tight_layout()


# ---------------------------------------------------------
# 1. Load Data (N, T, D)
# ---------------------------------------------------------
np.random.seed(42)

# Load your specific data file
try:
    data = np.load(
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_pten_embeddings_resnet3d-kinetics-300eps_ablated-tmrm_modified_labels/embeddings_raw.npy"
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260109_pten-t4_resnet_embeddings_2024v2-model_modified_labels/embeddings_resnet.npy"
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260110_kinetics_morphology_resnet_embeddings_val-set_tscrambled/embeddings_resnet.npy"
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260108_kinetics_morphology_resnet_embeddings_val-set/embeddings_resnet.npy"
        "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-val-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm/embeddings_raw.npy"
    )
    N, T, D = data.shape
    print(f"Loaded raw data shape: {data.shape}")
except FileNotFoundError:
    print("File not found. Generating synthetic fallback data...")
    N, T, D = 100, 20, 2048
    data = np.random.normal(0, 1, size=(N, T, D))

# --- CALCULATE VELOCITIES ---
# V_t = X_t - X_{t-1}
# New shape will be (N, T-1, D)
velocities = data[:, 1:, :] - data[:, :-1, :]
print(f"Computed velocities shape: {velocities.shape}")

# ---------------------------------------------------------
# 2. Run Analysis on VELOCITIES
# ---------------------------------------------------------

# A. Run Vector ACF (Scalar Metric)
vector_acf_mean, vector_acf_std = compute_batch_vector_acf(velocities, max_lag=9)

# B. Run Multi-PC PACF
pc_results = compute_multi_pc_pacf(velocities, n_components=3, max_lag=9)

# ---------------------------------------------------------
# 3. Plot Both
# ---------------------------------------------------------
plot_pc_grid(pc_results, vector_acf_mean, vector_acf_std, "Velocity Dynamics Analysis")
plt.show()