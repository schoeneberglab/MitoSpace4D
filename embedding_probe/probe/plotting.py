"""Matplotlib figures for the MS4D embedding probe."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PLOTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ms4d" / "plots"


def set_plots_dir(path: Path) -> None:
    global PLOTS_DIR
    PLOTS_DIR = path
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_hist(ax, data: np.ndarray, **kwargs) -> None:
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    span = float(data.max() - data.min())
    bins = kwargs.pop("bins", 50)
    if span <= 1e-8:
        ax.axvline(float(data.mean()), **{k: v for k, v in kwargs.items() if k in ("color", "label")})
        return
    ax.hist(data, bins=bins, **kwargs)


def _ensure_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def savefig(name: str) -> Path:
    out = _ensure_dir() / f"{name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_cumulative_variance(cev_dict: dict[str, np.ndarray], name: str, title: str) -> None:
    plt.figure(figsize=(7, 4))
    for label, cev in cev_dict.items():
        plt.plot(np.arange(1, len(cev) + 1), cev, label=label)
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    savefig(name)


def plot_sim_histogram(hist_dict: dict[str, np.ndarray], name: str, title: str) -> None:
    plt.figure(figsize=(7, 4))
    for label, sims in hist_dict.items():
        sims = np.asarray(sims, dtype=np.float64)
        span = float(np.ptp(sims))
        bins = 60 if span > 1e-8 else 1
        plt.hist(sims, bins=bins, alpha=0.5, density=True, label=label)
    plt.xlabel("Pairwise cosine similarity")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(fontsize=8)
    savefig(name)


def plot_within_vs_between(rows: list[dict], name: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    models = [r["model"] for r in rows]
    x = np.arange(len(models))
    w = 0.35
    plt.bar(x - w / 2, [r["within_mean"] for r in rows], w, label="Within drug")
    plt.bar(x + w / 2, [r["between_mean"] for r in rows], w, label="Between drug")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Mean cosine similarity")
    plt.title(title)
    plt.legend()
    savefig(name)


def plot_bar_dict(values: dict[str, float], name: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(6, 4))
    keys = list(values.keys())
    plt.bar(keys, [values[k] for k in keys], color="#1a5276")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(name)


def plot_scatter_two_encoders(x: np.ndarray, y: np.ndarray, name: str, title: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=4, alpha=0.25)
    plt.xlabel("Trained MS4D pairwise cos-sim")
    plt.ylabel("Random-init MS4D pairwise cos-sim")
    plt.title(title)
    savefig(name)


def plot_spearman_matrix(matrix: np.ndarray, labels: list[str], name: str, title: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1)
    plt.title(title)
    savefig(name)


def plot_pca_scatter(xy: np.ndarray, labels: np.ndarray, label_names: dict[int, str], name: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        m = labels == lab
        nm = label_names.get(int(lab), str(lab))
        plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, label=nm)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    savefig(name)


def plot_drug_heatmap(sim: np.ndarray, drug_names: list[str], name: str, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, xticklabels=drug_names, yticklabels=drug_names, vmin=-0.2, vmax=1.0, cmap="coolwarm")
    plt.title(title)
    savefig(name)


def plot_within_between_histograms(
    within: dict[str, np.ndarray],
    between: dict[str, np.ndarray],
    effect_sizes: dict[str, dict],
    name: str,
    title: str,
) -> None:
    n = len(within)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for ax, enc in zip(axes[0], within.keys()):
        _safe_hist(ax, between[enc], bins=50, alpha=0.5, density=True, label="Between")
        _safe_hist(ax, within[enc], bins=50, alpha=0.5, density=True, label="Within")
        es = effect_sizes[enc]
        ax.set_title(f"{enc}\nd={es['cohen_d']:.2f}, AUC={es['auc']:.2f}")
        ax.set_xlabel("Cosine similarity")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    savefig(name)


def plot_per_treatment_gaps(
    gaps: dict[str, float],
    drug_names: dict[str, str],
    name: str,
    title: str,
    top_n: int = 25,
) -> None:
    items = sorted(gaps.items(), key=lambda kv: -kv[1])[:top_n]
    labels = [drug_names.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(vals)), vals, color="#1a5276")
    plt.yticks(range(len(vals)), labels, fontsize=8)
    plt.xlabel("Within − between cosine similarity")
    plt.title(title)
    plt.gca().invert_yaxis()
    savefig(name)


def plot_scatter_xy(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, name: str, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=20, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(name)
