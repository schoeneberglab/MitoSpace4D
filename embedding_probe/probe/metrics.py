"""Embedding-space diagnostics (adapted from what-does-openphenom-learn)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def pca_eigenvalues(x: np.ndarray, n_components: int | None = None) -> np.ndarray:
    n_components = n_components or min(x.shape) - 1
    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(x)
    return pca.explained_variance_.astype(np.float64)


def participation_ratio(eigvals: np.ndarray) -> float:
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    return float((s**2) / (eigvals**2).sum())


def cumulative_explained_variance(eigvals: np.ndarray) -> np.ndarray:
    s = eigvals.sum()
    if s <= 0:
        return np.zeros_like(eigvals)
    return np.cumsum(eigvals) / s


def components_to_reach(cev: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cev, threshold) + 1
    return int(min(idx, len(cev)))


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    xn = l2_normalize(x)
    return xn @ xn.T


def upper_triangle(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def spearman_of_pairwise_sims(x: np.ndarray, y: np.ndarray) -> float:
    sx = upper_triangle(cosine_similarity_matrix(x))
    sy = upper_triangle(cosine_similarity_matrix(y))
    return float(spearmanr(sx, sy).statistic)


def within_between_stats(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    sim = cosine_similarity_matrix(emb)
    labels = np.asarray(labels)
    iu = np.triu_indices_from(sim, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    sims = sim[iu]
    within = float(sims[same].mean()) if same.any() else float("nan")
    between = float(sims[~same].mean()) if (~same).any() else float("nan")
    return {
        "within_mean": within,
        "between_mean": between,
        "gap": within - between,
        "within_std": float(sims[same].std()) if same.any() else float("nan"),
        "between_std": float(sims[~same].std()) if (~same).any() else float("nan"),
    }


def replicate_retrieval_map(emb: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    sim = cosine_similarity_matrix(emb)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    aps = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        order = np.argsort(-sim[i])
        order = order[order != i]
        is_pos = same[order]
        cum_hits = np.cumsum(is_pos)
        ranks = np.arange(1, n)
        precision_at_hits = cum_hits[is_pos] / ranks[is_pos]
        aps.append(precision_at_hits.mean())
    return float(np.mean(aps)) if aps else float("nan")


def random_map(labels: np.ndarray, seed: int = 0, n_trials: int = 30) -> float:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    vals = []
    for _ in range(n_trials):
        scores = rng.standard_normal((n, n))
        np.fill_diagonal(scores, -np.inf)
        aps = []
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            if not same.any():
                continue
            order = np.argsort(-scores[i])
            order = order[order != i]
            is_pos = same[order]
            cum_hits = np.cumsum(is_pos)
            ranks = np.arange(1, n)
            precision_at_hits = cum_hits[is_pos] / ranks[is_pos]
            aps.append(precision_at_hits.mean())
        vals.append(np.mean(aps))
    return float(np.mean(vals))


def effect_sizes(within: np.ndarray, between: np.ndarray) -> dict[str, float]:
    from scipy.stats import mannwhitneyu

    if len(within) == 0 or len(between) == 0:
        return {"gap": 0.0, "cohen_d": 0.0, "auc": 0.5}
    mu_w = float(within.mean())
    mu_b = float(between.mean())
    pooled_std = float(
        np.sqrt(
            ((len(within) - 1) * within.var() + (len(between) - 1) * between.var())
            / max(len(within) + len(between) - 2, 1)
        )
    )
    cohen_d = (mu_w - mu_b) / max(pooled_std, 1e-12)
    try:
        u, _ = mannwhitneyu(within, between, alternative="greater")
        auc = float(u / (len(within) * len(between)))
    except Exception:
        auc = 0.5
    return {"gap": mu_w - mu_b, "cohen_d": cohen_d, "auc": auc}
