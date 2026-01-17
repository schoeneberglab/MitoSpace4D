#!/usr/bin/env python3
"""
Differential perturbation via Wasserstein-flavored metric:
PCA -> Sliced Wasserstein (SW2) -> within-condition permutation test
Modified to include Z-score sensitivity and PCA dimension calculation.
"""
import numpy as np
from sklearn.decomposition import PCA
from tqdm import trange
import ot

# def exact_wasserstein_sensitivity(X, Y, robust=True):
#     """
#     Calculates the Exact Wasserstein Sensitivity Index (Beta_W2).
#     Corrected to be scale-invariant (Signal and Noise both scale linearly).
#
#     Parameters:
#     - X, Y: (n_samples, n_features) arrays
#     - robust: If True, uses Median Absolute Deviation (MAD) for dispersion.
#
#     Returns:
#     - w2_dist: Exact Wasserstein-2 distance.
#     - beta_w2: Sensitivity Index (Signal / Noise).
#     """
#     X = np.asarray(X, dtype=np.float64)
#     Y = np.asarray(Y, dtype=np.float64)
#
#     # --- 1. Signal: Exact Wasserstein-2 Distance ---
#     n_x, n_y = len(X), len(Y)
#     a, b = np.ones(n_x) / n_x, np.ones(n_y) / n_y
#
#     # CRITICAL FIX: Use 'sqeuclidean' (Squared Euclidean) distance.
#     # M_ij = ||x_i - y_j||^2
#     # If inputs scale by 10, M scales by 100.
#     M = ot.dist(X, Y, metric='sqeuclidean')
#
#     # Solve EMD. returns the transport cost.
#     # Since M is squared distance, Cost = W_2^2 (Squared Wasserstein)
#     w2_sq = ot.emd2(a, b, M)
#
#     # Take sqrt to get W_2.
#     # If inputs scale by 10 -> w2_sq scales by 100 -> w2_dist scales by 10.
#     w2_dist = np.sqrt(w2_sq)
#
#     # --- 2. Noise: Total Dispersion (Multivariate Spread) ---
#     if robust:
#         # Robust "Total Variance": Sum of squared rescaled MADs
#         # MAD scales linearly (x10). Squared MAD scales x100.
#         med_x = np.median(X, axis=0)
#         med_y = np.median(Y, axis=0)
#
#         mad_x = np.median(np.abs(X - med_x), axis=0)
#         mad_y = np.median(np.abs(Y - med_y), axis=0)
#
#         # Rescaling factor for normal consistency
#         k = 1.4826
#         total_var_x = np.sum((k * mad_x) ** 2)
#         total_var_y = np.sum((k * mad_y) ** 2)
#
#     else:
#         # Classical Total Variance: Trace of Covariance
#         # Variance scales by 100.
#         total_var_x = np.sum(np.var(X, axis=0, ddof=1))
#         total_var_y = np.sum(np.var(Y, axis=0, ddof=1))
#
#     # Combined dispersion (Noise denominator)
#     # sqrt(Variance) scales by 10.
#     noise_spread = np.sqrt(total_var_x + total_var_y)
#
#     # --- 3. Sensitivity Ratio ---
#     # Ratio: (10 * Signal) / (10 * Noise) = 1 (Invariant)
#     eps = 1e-12
#     beta_w2 = w2_dist / (noise_spread + eps)
#
#     return w2_dist, beta_w2

import numpy as np
import ot

def exact_wasserstein_sensitivity(X, Y, robust=True, normalize=True, log_transform=False):
    """
    Calculates the Exact Wasserstein Sensitivity Index (Beta_W2).
    Includes optional preprocessing to normalize features safely.

    Parameters:
    - X, Y: (n_samples, n_features) arrays
    - robust: If True, uses Median Absolute Deviation (MAD) for dispersion.
    - normalize: If True, scales features using POOLED statistics (Median/MAD).
                 Essential if features have different units/scales.
    - log_transform: If True, applies log1p to data before normalization.
                     Recommended for highly skewed intensity data.

    Returns:
    - w2_dist: Exact Wasserstein-2 distance (on the processed data).
    - beta_w2: Sensitivity Index (Signal / Noise).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # --- 0. Preprocessing (Normalization) ---
    if log_transform:
        # Handle skewness common in biological data
        X = np.log1p(np.maximum(X, 0))
        Y = np.log1p(np.maximum(Y, 0))

    if normalize:
        # CRITICAL: Normalize using POOLED statistics.
        # This brings all features to the same scale (e.g., sigma=1)
        # without erasing the shift between X and Y.
        pooled = np.vstack([X, Y])

        if robust:
            # Robust Scaling (Median / MAD) to handle outliers
            med = np.median(pooled, axis=0)
            # MAD = median(|x - median|)
            mad = np.median(np.abs(pooled - med), axis=0)
            # Avoid division by zero for constant features
            scale_factor = 1.4826 * mad
            scale_factor[scale_factor < 1e-12] = 1.0

            X = (X - med) / scale_factor
            Y = (Y - med) / scale_factor
        else:
            # Standard Scaling (Mean / SD)
            mu = np.mean(pooled, axis=0)
            sigma = np.std(pooled, axis=0)
            sigma[sigma < 1e-12] = 1.0

            X = (X - mu) / sigma
            Y = (Y - mu) / sigma

    # --- 1. Signal: Exact Wasserstein-2 Distance ---
    n_x, n_y = len(X), len(Y)
    a, b = np.ones(n_x) / n_x, np.ones(n_y) / n_y

    # CRITICAL FIX: Use 'sqeuclidean' (Squared Euclidean) distance.
    # M_ij = ||x_i - y_j||^2
    # If inputs scale by 10, M scales by 100.
    M = ot.dist(X, Y, metric='sqeuclidean')

    # Solve EMD. returns the transport cost.
    # Since M is squared distance, Cost = W_2^2 (Squared Wasserstein)
    w2_sq = ot.emd2(a, b, M)

    # Take sqrt to get W_2.
    # If inputs scale by 10 -> w2_sq scales by 100 -> w2_dist scales by 10.
    w2_dist = np.sqrt(w2_sq)

    # --- 2. Noise: Total Dispersion (Multivariate Spread) ---
    if robust:
        # Robust "Total Variance": Sum of squared rescaled MADs
        # MAD scales linearly (x10). Squared MAD scales x100.
        med_x = np.median(X, axis=0)
        med_y = np.median(Y, axis=0)

        mad_x = np.median(np.abs(X - med_x), axis=0)
        mad_y = np.median(np.abs(Y - med_y), axis=0)

        # Rescaling factor for normal consistency
        k = 1.4826
        total_var_x = np.sum((k * mad_x) ** 2)
        total_var_y = np.sum((k * mad_y) ** 2)

    else:
        # Classical Total Variance: Trace of Covariance
        # Variance scales by 100.
        total_var_x = np.sum(np.var(X, axis=0, ddof=1))
        total_var_y = np.sum(np.var(Y, axis=0, ddof=1))

    # Combined dispersion (Noise denominator)
    # sqrt(Variance) scales by 10.
    noise_spread = np.sqrt(total_var_x + total_var_y)

    # --- 3. Sensitivity Ratio ---
    # Ratio: (10 * Signal) / (10 * Noise) = 1 (Invariant)
    eps = 1e-12
    beta_w2 = w2_dist / (noise_spread + eps)

    return w2_dist, beta_w2

def calculate_ssmd(x, y, robust=True):
    """
    Calculates Strictly Standardized Mean Difference (SSMD) per dimension.
    Based on Zhang (2007).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if robust:
        # Robust estimate using Median and MAD [cite: 141-146]
        mu1, mu2 = np.median(x, axis=0), np.median(y, axis=0)

        # Rescaled MAD to estimate sigma [cite: 147]
        # 1.4826 * median(|x - median(x)|)
        s1 = 1.4826 * np.median(np.abs(x - mu1), axis=0)
        s2 = 1.4826 * np.median(np.abs(y - mu2), axis=0)

        # Robust SSMD formula [cite: 142]
        # denom is sqrt(s1^2 + s2^2)
        denom = np.sqrt(s1 ** 2 + s2 ** 2)
    else:
        # Classical MLE estimate [cite: 118]
        mu1, mu2 = np.mean(x, axis=0), np.mean(y, axis=0)

        # Sample variances (using ddof=1 for unbiased estimator)
        var1 = np.var(x, axis=0, ddof=1)
        var2 = np.var(y, axis=0, ddof=1)

        # The paper uses a specific MLE form, but for simple QC,
        # sqrt(var1 + var2) is the standard approximation for independent populations.
        denom = np.sqrt(var1 + var2)

    # Avoid division by zero
    eps = 1e-12
    beta = (mu1 - mu2) / (denom + eps)

    return beta

def get_pca_dim(data, threshold=0.95):
    """
    Calculates the number of principal components needed to capture
    a certain fraction of the variance (e.g. 0.95).
    """
    data = np.asarray(data, dtype=np.float64)
    pca = PCA().fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    return n_components


def _sliced_w2(x, y, n_proj=512, rng=None, return_squared=False):
    """
    Sliced 2-Wasserstein between two empirical distributions with equal weights.
    Uses random 1D projections + exact 1D W2 via sorting.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Match sample counts for simple equal-weight 1D W2
    n = min(len(x), len(y))
    if len(x) != n:
        x = x[rng.choice(len(x), size=n, replace=False)]
    if len(y) != n:
        y = y[rng.choice(len(y), size=n, replace=False)]
    d = x.shape[1]
    proj = rng.normal(size=(n_proj, d))
    proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)

    # Project to 1D: (n_proj, n)
    xp = proj @ x.T
    yp = proj @ y.T
    xp.sort(axis=1)
    yp.sort(axis=1)

    # 1D W2^2 for equal weights is mean squared diff between sorted samples
    w2_sq_per_proj = np.mean((xp - yp) ** 2, axis=1)
    sw2_sq = float(np.mean(w2_sq_per_proj))
    return sw2_sq if return_squared else np.sqrt(sw2_sq)


def sw2(x, y, pca_dim=100, n_proj=512, n_perm=2000, seed=0):
    """
    Wasserstein-flavored distance with permutation test and Z-score.
    Returns: (S_obs, p_value, z_score)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pooled = np.vstack([x, y]).astype(np.float64)

    # PCA + Whitening
    pca = PCA(n_components=min(pca_dim, pooled.shape[1]), svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(pooled)
    eps = 1e-12
    Z = Z / np.sqrt(pca.explained_variance_ + eps)

    zx = Z[: len(x)]
    zy = Z[len(x):]

    # Observed distance
    S_obs = _sliced_w2(zx, zy, n_proj=n_proj, rng=rng, return_squared=False)

    # Permutation test
    n0 = len(zx)
    n1 = len(zy)
    Z_pool = np.vstack([zx, zy])
    S_null = np.empty(n_perm, dtype=np.float64)
    for t in trange(n_perm, desc="Permuting SW2"):
        idx = rng.permutation(len(Z_pool))
        X0 = Z_pool[idx[:n0]]
        X1 = Z_pool[idx[n0:n0 + n1]]
        S_null[t] = _sliced_w2(X0, X1, n_proj=n_proj, rng=rng, return_squared=False)

    p = (np.sum(np.abs(S_null) >= abs(S_obs)) + 1.0) / (n_perm + 1.0)

    # Calculate Z-score (Sensitivity Index)
    mu_null = np.mean(S_null)
    std_null = np.std(S_null)
    z_score = (S_obs - mu_null) / (std_null + eps)

    return S_obs, p, z_score


def differential_sw2(
        A, Aprime, B, Bprime,
        pca_dim=100,
        n_proj=512,
        n_perm=2000,
        seed=0,
):
    """
    Compares two differential conditions.
    Returns: (SA, SB, T_obs, p_value)
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A);
    Aprime = np.asarray(Aprime)
    B = np.asarray(B);
    Bprime = np.asarray(Bprime)

    pooled = np.vstack([A, Aprime, B, Bprime]).astype(np.float64)
    pca = PCA(n_components=min(pca_dim, pooled.shape[1]), svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(pooled)

    eps = 1e-12
    Z = Z / np.sqrt(pca.explained_variance_ + eps)

    nA, nAp, nB, nBp = len(A), len(Aprime), len(B), len(Bprime)
    zA = Z[0:nA]
    zAprime = Z[nA:nA + nAp]
    zB = Z[nA + nAp:nA + nAp + nB]
    zBprime = Z[nA + nAp + nB:nA + nAp + nB + nBp]

    SA = _sliced_w2(zA, zAprime, n_proj=n_proj, rng=rng, return_squared=False)
    SB = _sliced_w2(zB, zBprime, n_proj=n_proj, rng=rng, return_squared=False)
    T_obs = SA - SB

    A_pool = np.vstack([zA, zAprime])
    B_pool = np.vstack([zB, zBprime])
    nA0, nA1 = len(zA), len(zAprime)
    nB0, nB1 = len(zB), len(zBprime)
    T_null = np.empty(n_perm, dtype=np.float64)
    for t in trange(n_perm, desc="Permuting Diff-SW2"):
        idxA = rng.permutation(len(A_pool))
        idxB = rng.permutation(len(B_pool))
        A0 = A_pool[idxA[:nA0]]
        A1 = A_pool[idxA[nA0:]]
        B0 = B_pool[idxB[:nB0]]
        B1 = B_pool[idxB[nB0:]]
        SA_t = _sliced_w2(A0, A1, n_proj=n_proj, rng=rng, return_squared=False)
        SB_t = _sliced_w2(B0, B1, n_proj=n_proj, rng=rng, return_squared=False)
        T_null[t] = SA_t - SB_t

    p = (np.sum(np.abs(T_null) >= abs(T_obs)) + 1.0) / (n_perm + 1.0)
    return SA, SB, T_obs, p

def _exact_w2(X, Y, return_squared=False):
    """ Calculates exact Wasserstein-2 distance between two distributions. """
    # Set up weights
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)

    # Cost matrix
    M = ot.dist(X, Y, metric="euclidean")**2

    # Transport plan
    G = ot.emd(a, b, M)

    # Calculate Wasserstein distance
    W2_sq = np.sum(G * M)
    return W2_sq if return_squared else np.sqrt(W2_sq)

def w2(X, Y, n_perm=None, seed=1123):
    """ Calculates Wasserstein-2 distance between two distributions. """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Match sample counts for simple equal-weight W2
    n = min(len(X), len(Y))
    if len(X) != n:
        X = X[rng.choice(len(X), size=n, replace=False)]
    if len(Y) != n:
        Y = Y[rng.choice(len(Y), size=n, replace=False)]

    W2_dist = _exact_w2(X, Y, return_squared=False)

    if n_perm is not None:
        n_perm = max(1, n_perm)
        Z = np.vstack([X, Y])
        W2_null = np.empty(n_perm, dtype=np.float64)
        for t in trange(n_perm):
            Z_perm = rng.permutation(Z)
            W2_null[t] = _exact_w2(Z_perm[:n], Z_perm[n:], return_squared=False)

        p = (np.sum(np.abs(W2_null) >= abs(W2_dist)) + 1.0) / (n_perm + 1.0)

        # Calculate Z-score (Sensitivity Index)
        mu_perm = np.mean(W2_null)
        std_perm = np.std(W2_null)
        z_score = (W2_dist - mu_perm) / (std_perm + 1e-12)
        print(f"Mean W2 Null: {mu_perm:.4f}")
        print(f"Std W2 Null: {std_perm:.4f}")
        print(f"S/N: {W2_dist / mu_perm:.2f}")
    else:
        p = None
        z_score = None
    return W2_dist, p, z_score


# if __name__ == "__main__":
#     print("=== Validating SW2 Sensitivity & Robustness ===\n")
#
#     rng = np.random.default_rng(42)
#
#     # ---------------------------------------------------------
#     # Test 1: Sensitivity (Can it detect signal?)
#     # ---------------------------------------------------------
#     print("--- Test 1: Sensitivity (Signal vs Noise) ---")
#     n_features = 50
#     n_samples = 100
#
#     X_base = rng.normal(0, 1, size=(n_samples, n_features))
#     X_pert = rng.normal(0, 1, size=(n_samples, n_features))
#     X_pert[:, :5] += 1.0  # Add signal
#
#     # Low perms for speed in testing
#     s_obs, p_val, z_score = sw2(X_base, X_pert, pca_dim=20, n_perm=100)
#     print(f"Signal Z-score: {z_score:.4f} (Expected > 2.0)")
#
#     if z_score > 2.0:
#         print(">> PASS: Sensitivity check passed.")
#     else:
#         print(">> FAIL: Sensitivity check failed.")
#     print()
#
#     # ---------------------------------------------------------
#     # Test 2: Robustness to Sample Imbalance
#     # ---------------------------------------------------------
#     print("--- Test 2: Sample Imbalance (200 vs 50 samples) ---")
#     X_large = rng.normal(0, 1, size=(200, n_features))
#     X_small = rng.normal(0, 1, size=(50, n_features))
#
#     try:
#         s_imbal, p_imbal, z_imbal = sw2(X_large, X_small, pca_dim=20, n_perm=50)
#         print(f"Imbalanced Z-score: {z_imbal:.4f} (Should be low/noise for same dist)")
#         print(">> PASS: Handled unequal sample sizes without error.")
#     except Exception as e:
#         print(f">> FAIL: Crashed on unequal sample sizes. Error: {e}")
#     print()
#
#     # ---------------------------------------------------------
#     # Test 3: Robustness to PCA Dimension Mismatch
#     # ---------------------------------------------------------
#     print("--- Test 3: PCA Dim > Feature Count ---")
#     # Requesting 100 PCs when data only has 10 features
#     X_low_dim = rng.normal(0, 1, size=(100, 10))
#     X_low_dim_2 = rng.normal(0, 1, size=(100, 10))
#
#     target_pca = 100
#     print(f"Requesting PCA dim={target_pca} for data with features={X_low_dim.shape[1]}")
#
#     try:
#         s_pca, _, _ = sw2(X_low_dim, X_low_dim_2, pca_dim=target_pca, n_perm=50)
#         print(f"Resulting SW2: {s_pca:.4f}")
#         print(">> PASS: Auto-corrected PCA dimension limit.")
#     except Exception as e:
#         print(f">> FAIL: Crashed on PCA dim mismatch. Error: {e}")
#     print()
#
#     # ---------------------------------------------------------
#     # Test 4: High Dimensionality Robustness
#     # ---------------------------------------------------------
#     print("--- Test 4: High Dimensionality (Features > Samples) ---")
#     # 1000 features, only 50 samples (Curse of dimensionality check)
#     X_high = rng.normal(0, 1, size=(50, 1000))
#     X_high_pert = rng.normal(0, 1, size=(50, 1000))
#     X_high_pert[:, :50] += 0.5
#
#     try:
#         # Should default to min(samples, features) -> 50 PCs (or 100 since pooled is 100 samples)
#         # Pooled samples = 100, Features = 1000. Max rank = 100.
#         s_high, p_high, z_high = sw2(X_high, X_high_pert, pca_dim=50, n_perm=50)
#         print(f"High-Dim Z-score: {z_high:.4f}")
#         print(">> PASS: Handled Features > Samples.")
#     except Exception as e:
#         print(f">> FAIL: Crashed on High Dimensions. Error: {e}")

# if __name__ == "__main__":
#     print("=== Validating SW2: Strict Signal vs Noise Comparison ===\n")
#     rng = np.random.default_rng(42)
#
#
#     def run_comparative_check(case_name, n_feat, n_pca, signal_strength=1.5):
#         """
#         Runs two checks:
#         1. Null Check (Noise vs Noise) -> Expect Low Z-score
#         2. Signal Check (Base vs Perturbed) -> Expect High Z-score
#         Asserts that Signal Z-score > Null Z-score
#         """
#         print(f"--- Case: {case_name} ---")
#         print(f"    Dim: {n_feat}, Requested PCA: {n_pca}")
#
#         n_samples = 100
#
#         # 1. Baseline Data
#         X_base = rng.normal(0, 1, size=(n_samples, n_feat))
#
#         # 2. Null Data (Just random noise, no signal)
#         X_null = rng.normal(0, 1, size=(n_samples, n_feat))
#
#         # 3. Signal Data (Perturbation added)
#         X_signal = rng.normal(0, 1, size=(n_samples, n_feat))
#         # Add signal to 10% of features
#         n_mod = max(2, int(0.1 * n_feat))
#         X_signal[:, :n_mod] += signal_strength
#
#         try:
#             # Run Null
#             _, _, z_null = sw2(X_base, X_null, pca_dim=n_pca, n_perm=1000)
#
#             # Run Signal
#             _, _, z_signal = sw2(X_base, X_signal, pca_dim=n_pca, n_perm=1000)
#
#             print(f"    Null Z-score:   {z_null:.4f}")
#             print(f"    Signal Z-score: {z_signal:.4f}")
#
#             # STRICT COMPARISON
#             if z_signal > z_null and z_signal > 2.0:
#                 print("    >> PASS: Signal correctly identified as strictly higher than noise.")
#             elif z_signal <= z_null:
#                 print("    >> FAIL: Metric failed to distinguish signal from noise.")
#             else:
#                 print("    >> WARNING: Signal detected, but confidence (Z-score) was low.")
#
#         except Exception as e:
#             print(f"    >> CRITICAL FAIL: Crashed. Error: {e}")
#         print()
#
#
#     # 1. High Data Dim / High PCA Dim
#     run_comparative_check("High Dim / High PCA", n_feat=500, n_pca=100)
#
#     # 2. High Data Dim / Low PCA Dim (Compression test)
#     run_comparative_check("High Dim / Low PCA", n_feat=500, n_pca=5)
#
#     # 3. Low Data Dim / High PCA Dim (Oversize request test)
#     run_comparative_check("Low Dim / High PCA (Oversized)", n_feat=20, n_pca=100)
#
#     # 4. Low Data Dim / Low PCA Dim
#     run_comparative_check("Low Dim / Low PCA", n_feat=20, n_pca=5)
#
#     # 5. Weak Signal Test (Check sensitivity threshold)
#     print("--- Case: Weak Signal Check ---")
#     # Reduced signal strength to see if metric still orders them correctly
#     run_comparative_check("Weak Signal Strength (0.3)", n_feat=50, n_pca=10, signal_strength=0.3)

if __name__ == "__main__":
    print("=== Validating SW2: Geometry, Alignment, and Resolution ===\n")
    rng = np.random.default_rng(42)


    def run_resolution_check(case_name, n_feat, n_pca):
        """
        Tests Resolution: Can we distinguish small steps in signal intensity?
        CRITICAL FIX: Freezes the noise background to test signal delta only.
        """
        print(f"--- Case: {case_name} (Resolution Check) ---")

        n_samples = 100
        n_mod = max(2, int(0.1 * n_feat))

        # 1. Baseline
        X_base = rng.normal(0, 1, size=(n_samples, n_feat))

        # 2. FREEZE the Noise Background
        X_noise_fixed = rng.normal(0, 1, size=(n_samples, n_feat))

        # 3. Define Signal Direction
        Signal_pattern = np.zeros((n_samples, n_feat))
        Signal_pattern[:, :n_mod] = 1.0

        amplitudes = [1.0, 1.2, 1.4, 1.6]
        z_scores = []

        try:
            for amp in amplitudes:
                # Add signal to the EXACT SAME noise background
                X_sig = X_noise_fixed + (Signal_pattern * amp)
                _, _, z = sw2(X_base, X_sig, pca_dim=n_pca, n_perm=1000)
                z_scores.append(z)

            print(f"    Amplitudes: {amplitudes}")
            print(f"    Z-Scores:   {['{:.4f}'.format(z) for z in z_scores]}")

            is_monotonic = all(x < y for x, y in zip(z_scores, z_scores[1:]))

            if is_monotonic:
                print("    >> PASS: Metric successfully resolved intensity steps.")
            else:
                print("    >> FAIL: Metric failed to resolve steps (Z-scores not strictly increasing).")

        except Exception as e:
            print(f"    >> CRITICAL FAIL: Crashed. Error: {e}")
        print()


    def run_geometry_check(case_name, n_feat, n_pca):
        """
        Tests Signal Geometry:
        1. Feature Distribution: Contiguous Block vs. Randomly Scattered Features.
           (Should be similar for standard metrics).
        2. Manifold Alignment: Signal along PC1 (Hidden) vs. Orthogonal (Exposed).
           (Orthogonal should generally be easier to detect).
        """
        print(f"--- Case: {case_name} (Geometry Check) ---")
        n_samples = 100
        X_base = rng.normal(0, 1, size=(n_samples, n_feat))

        # --- TEST 1: Spatial Distribution (Block vs Scattered) ---
        # Same total energy, just different "locations" in the array
        n_mod = int(0.2 * n_feat)
        amp = 1.5

        # A. Block Signal (Indices 0 to N)
        X_block = rng.normal(0, 1, size=(n_samples, n_feat))
        X_block[:, :n_mod] += amp

        # B. Scattered Signal (Random Indices)
        X_scatter = rng.normal(0, 1, size=(n_samples, n_feat))
        indices = rng.choice(n_feat, n_mod, replace=False)
        X_scatter[:, indices] += amp

        _, _, z_block = sw2(X_base, X_block, pca_dim=n_pca, n_perm=1000)
        _, _, z_scatt = sw2(X_base, X_scatter, pca_dim=n_pca, n_perm=1000)

        print(f"    [Distribution] Block Z: {z_block:.4f} | Scattered Z: {z_scatt:.4f}")

        # We expect them to be roughly similar (within stochastic variance)
        if abs(z_block - z_scatt) < (0.2 * z_block):
            print("    >> PASS: Metric is robust to feature permutation (Scattered ~= Block).")
        else:
            print("    >> WARNING: Metric is sensitive to feature ordering.")

        # --- TEST 2: Manifold Alignment (Aligned vs Orthogonal) ---
        # Generate correlated noise (Manifold)
        # Create a dominant direction (PC1)
        v_dominant = rng.normal(0, 1, size=(1, n_feat))
        v_dominant /= np.linalg.norm(v_dominant)

        # Base data has strong variance along v_dominant
        X_manifold = rng.normal(0, 1, size=(n_samples, n_feat)) + (rng.normal(0, 3, size=(n_samples, 1)) * v_dominant)

        # Signal A: Aligned (Hiding in the noise direction)
        # We add signal exactly along v_dominant
        X_aligned = X_manifold.copy() + (v_dominant * amp)

        # Signal B: Orthogonal (Sticking out)
        # Generate random vector, make orthogonal to v_dominant
        v_ortho = rng.normal(0, 1, size=(1, n_feat))
        v_ortho -= v_ortho.dot(v_dominant.T) * v_dominant
        v_ortho /= np.linalg.norm(v_ortho)

        X_ortho = X_manifold.copy() + (v_ortho * amp)

        _, _, z_align = sw2(X_manifold, X_aligned, pca_dim=n_pca, n_perm=1000)
        _, _, z_ortho = sw2(X_manifold, X_ortho, pca_dim=n_pca, n_perm=1000)

        print(f"    [Alignment]    Aligned Z: {z_align:.4f} | Orthogonal Z: {z_ortho:.4f}")

        if z_ortho > z_align:
            print("    >> PASS: Orthogonal signal (cleaner) detected more strongly than Aligned signal (hidden).")
        else:
            print("    >> NOTE: Aligned signal detected equally or better. (Metric might normalize variance).")
        print()


    # 1. Run Resolution Tests (With Frozen Noise Fix)
    run_resolution_check("High Res Check", n_feat=500, n_pca=100)

    # 2. Run Geometry Tests
    run_geometry_check("High Dim Geometry", n_feat=500, n_pca=50)
    run_geometry_check("Low Dim Geometry", n_feat=50, n_pca=10)