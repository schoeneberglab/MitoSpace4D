import argparse
import os
import sys
import numpy as np
import pandas as pd

def load_embeddings(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.values.astype(np.float64)
    else:
        raise ValueError("Embeddings must be .npy or .csv")

def save_embeddings(path: str, X: np.ndarray):
    if path.endswith(".npy"):
        np.save(path, X)
    elif path.endswith(".csv"):
        pd.DataFrame(X).to_csv(path, index=False)
    else:
        raise ValueError("Output path must be .npy or .csv")
    
def build_controls_mask(meta: pd.DataFrame, control_col: str, control_value: str) -> np.ndarray:
    if control_col not in meta.columns:
        raise ValueError(f"Metadata missing column '{control_col}'.")
    return (meta[control_col].astype(str).values == str(control_value))

def compute_covariance(X: np.ndarray, ledoit_wolf: bool = False) -> np.ndarray:
    # X is (n_ctrl x d), assumed centered before calling
    if ledoit_wolf:
        try:
            from sklearn.covariance import LedoitWolf
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for --ledoit-wolf. Install with `pip install scikit-learn`."
            ) from e
        lw = LedoitWolf().fit(X)
        return lw.covariance_
    # Unbiased sample covariance; (rows = observations)
    return (X.T @ X) / max(1, (X.shape[0] - 1))

#-- ZCA Whitening
def zca_whitener_from_controls(X_ctrl: np.ndarray, eps: float = 1e-6,
                               ledoit_wolf: bool = False):
    """
    Build ZCA-like whitening transform from control rows.

    Returns:
      mean_ctrl: (d,)
      W: (d x d) whitening matrix such that Y = (X - mean_ctrl) @ W
    """
    mean_ctrl = X_ctrl.mean(axis=0, dtype=np.float64)
    Xc = X_ctrl - mean_ctrl
    C = compute_covariance(Xc, ledoit_wolf=ledoit_wolf)

    # Eigen-decomposition (symmetrize for numerical stability)
    C = (C + C.T) * 0.5
    evals, evecs = np.linalg.eigh(C)  # eigh for symmetric matrices
    
    # Clamp eigenvalues
    evals_clamped = np.maximum(evals, eps)
    inv_sqrt = 1.0 / np.sqrt(evals_clamped)
    
    # ZCA: W = U * diag(inv_sqrt) * U^T
    W = (evecs * inv_sqrt) @ evecs.T
    return mean_ctrl, W


#-- TVN calcs
def tvn_global(X: np.ndarray, controls_mask: np.ndarray,
               eps: float, ledoit_wolf: bool) -> np.ndarray:
    if controls_mask.sum() < X.shape[1]:
        # Not enough controls for a full-rank covariance but will continue anyways (eps helps)
        sys.stderr.write(
            f"Warning: controls ({controls_mask.sum()}) < feature_dim ({X.shape[1]}). "
            "Result may be rank-deficient; using eps regularization.\n"
        )
    mean_ctrl, W = zca_whitener_from_controls(X[controls_mask], eps=eps, ledoit_wolf=ledoit_wolf)
    return (X - mean_ctrl) @ W

def tvn_per_batch(X: np.ndarray, meta: pd.DataFrame, batch_col: str,
                  controls_mask: np.ndarray, eps: float, ledoit_wolf: bool) -> np.ndarray:
    """
    Apply TVN separately within each batch (e.g., plate), using that batch's controls
    to compute mean/cov; then transform only rows from that batch.
    """
    X_out = np.empty_like(X, dtype=np.float64)
    batches = meta[batch_col].values
    for b in np.unique(batches):
        idx_batch = (batches == b)
        idx_ctrl_b = idx_batch & controls_mask
        if idx_ctrl_b.sum() < 2:
            # Fallback to global controls if a batch lacks enough controls
            mean_ctrl, W = zca_whitener_from_controls(X[controls_mask], eps=eps, ledoit_wolf=ledoit_wolf)
        else:
            mean_ctrl, W = zca_whitener_from_controls(X[idx_ctrl_b], eps=eps, ledoit_wolf=ledoit_wolf)
        X_out[idx_batch] = (X[idx_batch] - mean_ctrl) @ W
    return X_out

def main():
    ap = argparse.ArgumentParser(description="Typical Variation Normalization (TVN) for embeddings.")
    ap.add_argument("--embeddings", required=True, help=".npy or .csv (rows=samples, cols=features)")
    ap.add_argument("--metadata", required=True, help=".csv with at least the control column")
    ap.add_argument("--control-col", required=True, help="Column in metadata that marks controls")
    ap.add_argument("--control-value", required=True, help="Value in control-col that denotes negative controls (e.g., DMSO)")
    ap.add_argument("--batch-col", default=None, help="Optional column for per-batch TVN (e.g., plate_id)")
    ap.add_argument("--ledoit-wolf", action="store_true", help="Use Ledoit-Wolf shrinkage (requires scikit-learn)")
    ap.add_argument("--eps", type=float, default=1e-6, help="Eigenvalue floor for whitening (default: 1e-6)")
    ap.add_argument("--out", required=True, help="Output .npy or .csv")
    args = ap.parse_args()

    X = load_embeddings(args.embeddings).astype(np.float64)
    meta = pd.read_csv(args.metadata)
    if len(meta) != X.shape[0]:
        raise ValueError(f"metadata rows ({len(meta)}) != embeddings rows ({X.shape[0]}).")

    ctrl_mask = build_controls_mask(meta, args.control_col, args.control_value)
    if ctrl_mask.sum() == 0:
        raise ValueError("No negative-control rows found. Check --control-col/--control-value.")

    if args.batch_col is None:
        X_tvn = tvn_global(X, ctrl_mask, eps=args.eps, ledoit_wolf=args.ledoit_wolf)
    else:
        if args.batch_col not in meta.columns:
            raise ValueError(f"Metadata missing batch column '{args.batch_col}'.")
        X_tvn = tvn_per_batch(X, meta, args.batch_col, ctrl_mask, eps=args.eps, ledoit_wolf=args.ledoit_wolf)

    save_embeddings(args.out, X_tvn)
    print(f"Saved TVN-normalized embeddings to: {args.out}")

if __name__ == "__main__":
    # main()

    # Example usage w/ toy embeddings (100 samples, 10 features)
    np.random.seed(0)
    X = np.random.randn(100, 10)

    # Make metadata with a control column (first 20 are controls)
    meta = pd.DataFrame({
        "treatment": ["control"] * 20 + ["A"] * 40 + ["B"] * 40,
        "plate_id": ["plate1"] * 50 + ["plate2"] * 50
    })

    # Run global (approx) TVN
    ctrl_mask = (meta["treatment"].values == "control")
    X_tvn = tvn_global(X, ctrl_mask, eps=1e-6, ledoit_wolf=False)

    print("Original embeddings shape:", X.shape)
    print("TVN-normalized embeddings shape:", X_tvn.shape)
    print("First row before:\n", X[0])
    print("First row after:\n", X_tvn[0])
