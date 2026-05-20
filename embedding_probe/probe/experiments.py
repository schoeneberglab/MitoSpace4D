"""MS4D embedding probe experiments (mirrors OpenPhenom probe E1, E2, E4, E6–E11)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import data_utils as du
from . import metrics as M
from . import plotting as P

PROBE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ProbeContext:
    trained_df: pd.DataFrame
    label_names: dict[int, str]
    x_trained: np.ndarray
    model_id: str = "ms4d"
    display_name: str = "MitoSpace4D (4D)"
    x_random: np.ndarray | None = None
    labels: np.ndarray | None = None
    control_label: int = 0
    seed: int = 0

    @property
    def results_dir(self) -> Path:
        return PROBE_ROOT / "results" / self.model_id

    @property
    def numbers_path(self) -> Path:
        return self.results_dir / "numbers.json"


def _save_findings(ctx: ProbeContext, name: str, payload: dict) -> None:
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    blob: dict = {}
    if ctx.numbers_path.exists():
        blob = json.loads(ctx.numbers_path.read_text())
    blob[name] = payload
    ctx.numbers_path.write_text(json.dumps(blob, indent=2, default=float))


def experiment_1_geometry(ctx: ProbeContext, subsample: int = 20000) -> dict:
    rng = np.random.default_rng(ctx.seed)
    x = ctx.x_trained
    n = x.shape[0]
    idx = rng.choice(n, size=min(subsample, n), replace=False)
    x_sub = x[idx]

    eigvals = M.pca_eigenvalues(x_sub)
    pr = M.participation_ratio(eigvals)
    cev = M.cumulative_explained_variance(eigvals)
    n90 = M.components_to_reach(cev, 0.9)
    n99 = M.components_to_reach(cev, 0.99)

    samp_idx = rng.choice(len(x_sub), size=min(2000, len(x_sub)), replace=False)
    sims = M.upper_triangle(M.cosine_similarity_matrix(x_sub[samp_idx]))

    P.plot_cumulative_variance(
        {f"MS4D trained ({len(x_sub):,} cells)": cev},
        name="e1_pca_cumulative_variance",
        title="E1. PCA cumulative explained variance (MS4D 2048-d)",
    )
    P.plot_sim_histogram(
        {f"MS4D, {len(samp_idx)} cells": sims},
        name="e1_similarity_histogram",
        title="E1. Pairwise cosine similarity distribution",
    )

    findings = {
        "n_cells_used": int(len(x_sub)),
        "embedding_dim": int(x_sub.shape[1]),
        "effective_rank_participation_ratio": pr,
        "components_for_90pct_variance": n90,
        "components_for_99pct_variance": n99,
        "top1_eigenvalue_share": float(eigvals[0] / eigvals.sum()),
        "top10_eigenvalue_share": float(eigvals[:10].sum() / eigvals.sum()),
        "pairwise_sim_mean": float(sims.mean()),
        "pairwise_sim_std": float(sims.std()),
        "pairwise_sim_p05": float(np.quantile(sims, 0.05)),
        "pairwise_sim_p95": float(np.quantile(sims, 0.95)),
    }
    _save_findings(ctx, "E1_geometry", findings)
    return findings


def experiment_2_trained_vs_random(ctx: ProbeContext, max_cells: int = 8000) -> dict:
    if ctx.x_random is None or ctx.labels is None:
        raise ValueError("E2 requires aligned random-init embeddings")

    rng = np.random.default_rng(ctx.seed)
    n = ctx.x_trained.shape[0]
    idx = rng.choice(n, size=min(max_cells, n), replace=False)
    x_t, x_r, labels = ctx.x_trained[idx], ctx.x_random[idx], ctx.labels[idx]

    rho = M.spearman_of_pairwise_sims(x_t, x_r)
    sims_t = M.upper_triangle(M.cosine_similarity_matrix(x_t))
    sims_r = M.upper_triangle(M.cosine_similarity_matrix(x_r))

    stats_t = M.within_between_stats(x_t, labels)
    stats_r = M.within_between_stats(x_r, labels)
    map_t = M.replicate_retrieval_map(x_t, labels)
    map_r = M.replicate_retrieval_map(x_r, labels)
    map_rand = M.random_map(labels, seed=ctx.seed)

    pr_t = M.participation_ratio(M.pca_eigenvalues(x_t))
    pr_r = M.participation_ratio(M.pca_eigenvalues(x_r))

    P.plot_scatter_two_encoders(sims_t, sims_r, "e2_pairwise_sim_scatter", f"E2. Trained vs random-init (Spearman ρ={rho:.3f})")
    P.plot_sim_histogram(
        {"Trained MS4D": sims_t, "Random-init MS4D": sims_r},
        "e2_pairwise_sim_overlay",
        "E2. Pairwise cosine similarity distributions",
    )
    P.plot_within_vs_between(
        [{"model": "Trained", **stats_t}, {"model": "Random-init", **stats_r}],
        "e2_within_vs_between",
        "E2. Within- vs between-drug cosine similarity",
    )
    P.plot_bar_dict(
        {"Random ranking (floor)": map_rand, "Random-init MS4D": map_r, "Trained MS4D": map_t},
        "e2_replicate_map",
        "E2. Cell replicate-retrieval mAP (same drug label)",
        "mAP",
    )

    findings = {
        "n_cells": int(len(x_t)),
        "n_drugs": int(len(np.unique(labels))),
        "spearman_pairwise_sim_rankings": rho,
        "trained_gap": stats_t["gap"],
        "random_gap": stats_r["gap"],
        "trained_replicate_map": map_t,
        "random_replicate_map": map_r,
        "random_ranking_floor_map": map_rand,
        "trained_participation_ratio": pr_t,
        "random_participation_ratio": pr_r,
        "map_lift_trained_over_random": (map_t - map_r) / max(1e-9, map_r),
    }
    _save_findings(ctx, "E2_trained_vs_random", findings)
    return findings


def experiment_4_discriminability(
    ctx: ProbeContext,
    n_classes: int = 25,
    replicates_per_class: int = 20,
) -> dict:
    idx = du.pick_balanced_indices(ctx.labels, n_classes, replicates_per_class, seed=ctx.seed)
    x = ctx.x_trained[idx]
    labels = ctx.labels[idx]

    stats = M.within_between_stats(x, labels)
    map_score = M.replicate_retrieval_map(x, labels)
    rand_floor = M.random_map(labels, seed=ctx.seed)

    P.plot_within_vs_between(
        [{"model": "MS4D (raw)", **stats}],
        "e4_within_vs_between",
        f"E4. Within vs between cos-sim ({n_classes} drugs × {replicates_per_class} cells)",
    )
    P.plot_bar_dict(
        {"Random ranking": rand_floor, "MS4D (raw)": map_score},
        "e4_replicate_map",
        "E4. Replicate-retrieval mAP (raw embeddings)",
        "mAP",
    )

    ctrl_mask = ctx.labels == ctx.control_label
    ctrl_emb = ctx.x_trained[ctrl_mask]
    if ctrl_emb.shape[0] < 50:
        ctrl_emb = ctx.x_trained

    x_corr = du.pca_center_scale_fit_transform(x, ctrl_emb)
    stats_corr = M.within_between_stats(x_corr, labels)
    map_corr = M.replicate_retrieval_map(x_corr, labels)

    P.plot_within_vs_between(
        [{"model": "Raw 2048-d", **stats}, {"model": "PCA + center/scale (control)", **stats_corr}],
        "e4_within_vs_between_corrected",
        "E4b. Effect of PCA-CenterScale (fit on control cells)",
    )
    P.plot_bar_dict(
        {"Random floor": rand_floor, "Raw": map_score, "PCA + center/scale": map_corr},
        "e4_map_corrected",
        "E4b. Replicate-retrieval mAP after post-processing",
        "mAP",
    )

    findings = {
        "n_classes": int(len(np.unique(labels))),
        "replicates_per_class": int(replicates_per_class),
        "raw_gap": stats["gap"],
        "raw_map": map_score,
        "post_pca_centerscale_gap": stats_corr["gap"],
        "post_pca_centerscale_map": map_corr,
        "random_floor_map": rand_floor,
    }
    _save_findings(ctx, "E4_discriminability", findings)
    return findings


def experiment_6_pca_viz(ctx: ProbeContext, subsample: int = 5000) -> dict:
    rng = np.random.default_rng(ctx.seed)
    n = min(subsample, len(ctx.x_trained))
    idx = rng.choice(len(ctx.x_trained), size=n, replace=False)
    x = ctx.x_trained[idx]
    labels = ctx.labels[idx]

    pca = PCA(n_components=2).fit(x)
    xy = pca.transform(x)
    P.plot_pca_scatter(xy, labels, ctx.label_names, "e6_pca_by_drug", "E6. PCA of MS4D embeddings colored by drug")

    findings = {"n_cells": int(n), "pc1_var": float(pca.explained_variance_ratio_[0]), "pc2_var": float(pca.explained_variance_ratio_[1])}
    _save_findings(ctx, "E6_pca_viz", findings)
    return findings


def experiment_7_baseline_matrix(ctx: ProbeContext, max_cells: int = 6000) -> dict:
    if ctx.x_random is None:
        raise ValueError("E7 requires random-init embeddings")

    rng = np.random.default_rng(ctx.seed)
    idx = rng.choice(len(ctx.x_trained), size=min(max_cells, len(ctx.x_trained)), replace=False)
    x_t, x_r = ctx.x_trained[idx], ctx.x_random[idx]
    labels = ctx.labels[idx]

    rho = M.spearman_of_pairwise_sims(x_t, x_r)
    map_t = M.replicate_retrieval_map(x_t, labels)
    map_r = M.replicate_retrieval_map(x_r, labels)
    map_floor = M.random_map(labels, seed=ctx.seed)

    mat = np.array([[1.0, rho], [rho, 1.0]])
    P.plot_spearman_matrix(mat, ["Trained", "Random-init"], "e7_spearman_matrix", "E7. Spearman ρ of pairwise ranking (trained vs random-init)")
    P.plot_bar_dict(
        {"Random floor": map_floor, "Random-init": map_r, "Trained": map_t},
        "e7_replicate_map_comparison",
        "E7. Replicate-retrieval mAP",
        "mAP",
    )

    findings = {"spearman_trained_vs_random": rho, "map_trained": map_t, "map_random": map_r, "map_floor": map_floor}
    _save_findings(ctx, "E7_baseline", findings)
    return findings


def experiment_8_treatment_viz(ctx: ProbeContext) -> dict:
    labels = ctx.labels
    unique = np.sort(np.unique(labels))
    mean_unit = []
    names = []
    for lab in unique:
        mask = labels == lab
        emb = M.l2_normalize(ctx.x_trained[mask])
        mu = emb.mean(axis=0)
        mean_unit.append(mu)
        names.append(ctx.label_names.get(int(lab), str(lab)))
    mean_unit = np.stack(mean_unit)
    norms = np.linalg.norm(mean_unit, axis=1, keepdims=True)
    mean_unit = mean_unit / np.clip(norms, 1e-12, None)
    sim = mean_unit @ mean_unit.T

    P.plot_drug_heatmap(sim, names, "e8_drug_cosine_heatmap", "E8. Mean-direction cosine similarity between drugs")

    findings = {"n_drugs": int(len(unique)), "mean_off_diagonal_sim": float(sim[~np.eye(len(unique), dtype=bool)].mean())}
    _save_findings(ctx, "E8_treatment_viz", findings)
    return findings


def experiment_9_constancy(ctx: ProbeContext, subsample: int = 5000) -> dict:
    rng = np.random.default_rng(ctx.seed)
    idx = rng.choice(len(ctx.x_trained), size=min(subsample, len(ctx.x_trained)), replace=False)
    x = ctx.x_trained[idx]
    per_dim_std = x.std(axis=0)
    norms = np.linalg.norm(x, axis=1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.hist(per_dim_std, bins=80, color="#1a5276")
    plt.xlabel("Per-dimension std across cells")
    plt.ylabel("Count")
    plt.title("E9. Embedding dimension variability")
    P.savefig("e9_per_dim_std")

    plt.figure(figsize=(6, 3))
    span = float(np.nanmax(norms) - np.nanmin(norms))
    if span > 1e-6:
        plt.hist(norms, bins=60, color="#1a5276")
    else:
        plt.axvline(float(norms[0]), color="#1a5276")
        plt.text(0.5, 0.5, f"Nearly constant L2 norm ≈ {float(norms[0]):.4f}", ha="center", transform=plt.gca().transAxes)
    plt.xlabel("L2 norm of stored embedding vectors")
    plt.ylabel("Count")
    plt.title("E9. Embedding L2 norm distribution")
    P.savefig("e9_l2_norm_histogram")

    findings = {
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_max": float(per_dim_std.max()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_std": float(norms.std()),
    }
    _save_findings(ctx, "E9_constancy", findings)
    return findings


def experiment_10_cossim_by_condition(ctx: ProbeContext, min_reps: int = 50, max_cells: int = 12000) -> dict:
    labels = ctx.labels
    counts = pd.Series(labels).value_counts()
    keep = counts[counts >= min_reps].index.to_numpy()
    mask = np.isin(labels, keep)
    idx = np.where(mask)[0]
    if len(idx) > max_cells:
        rng = np.random.default_rng(ctx.seed)
        idx = rng.choice(idx, size=max_cells, replace=False)
    sub_labels = labels[idx]

    encoders: dict[str, np.ndarray] = {ctx.display_name: ctx.x_trained[idx]}
    if ctx.x_random is not None:
        encoders["Random-init (same arch.)"] = ctx.x_random[idx]

    sims_within: dict[str, np.ndarray] = {}
    sims_between: dict[str, np.ndarray] = {}
    effect_sizes: dict[str, dict] = {}
    n = len(idx)
    for enc_label, emb in encoders.items():
        sim = M.cosine_similarity_matrix(emb)
        iu = np.triu_indices(n, k=1)
        same = sub_labels[iu[0]] == sub_labels[iu[1]]
        s = sim[iu]
        sims_within[enc_label] = s[same]
        sims_between[enc_label] = s[~same]
        effect_sizes[enc_label] = M.effect_sizes(sims_within[enc_label], sims_between[enc_label])

    P.plot_within_between_histograms(
        sims_within,
        sims_between,
        effect_sizes,
        "e10_cossim_within_vs_between",
        f"E10. Within-drug vs between-drug cosine similarity ({n} cells)",
    )

    per_treatment: dict[str, dict[str, float]] = {}
    trained_key = next(k for k in encoders if "random" not in k.lower())
    emb = encoders[trained_key]
    sim = M.cosine_similarity_matrix(emb)
    for t in keep:
        t_idx = np.where(sub_labels == t)[0]
        o_idx = np.where(sub_labels != t)[0]
        within_block = sim[np.ix_(t_idx, t_idx)]
        iu_t = np.triu_indices_from(within_block, k=1)
        within_vals = within_block[iu_t]
        between_vals = sim[np.ix_(t_idx, o_idx)].flatten()
        per_treatment[str(int(t))] = float(within_vals.mean() - between_vals.mean())

    name_map = {str(int(k)): ctx.label_names.get(int(k), str(k)) for k in keep}
    P.plot_per_treatment_gaps(per_treatment, name_map, "e10_per_treatment_gaps", "E10. Per-drug embedding discriminability (trained)")

    findings = {
        "n_cells": int(n),
        "n_drugs": int(len(keep)),
        "effect_sizes": effect_sizes,
    }
    _save_findings(ctx, "E10_cossim_by_condition", findings)
    return findings


def experiment_11_phenotype_vs_gap(ctx: ProbeContext, min_reps: int = 50) -> dict:
    """Correlate MitoTNT mean features with per-drug embedding gaps (pixel unusualness analogue)."""
    from scipy.stats import spearmanr

    feat_cols = du.mitotnt_mean_columns(ctx.trained_df)
    if not feat_cols:
        findings = {"note": "No MitoTNT *_mean columns in trained parquet; skipped."}
        _save_findings(ctx, "E11_phenotype_vs_gap", findings)
        return findings

    labels = ctx.labels
    counts = pd.Series(labels).value_counts()
    keep = counts[counts >= min_reps].index

    gaps: list[float] = []
    drug_ids: list[int] = []
    mitotnt_z: list[np.ndarray] = []

    grand = ctx.trained_df[feat_cols].to_numpy(dtype=np.float64)
    grand_mean = grand.mean(axis=0)
    grand_std = grand.std(axis=0) + 1e-9

    emb_all = ctx.x_trained
    sim_all = M.cosine_similarity_matrix(emb_all)

    for t in keep:
        t_mask = labels == t
        t_idx = np.where(t_mask)[0]
        o_idx = np.where(~t_mask)[0]
        within_block = sim_all[np.ix_(t_idx, t_idx)]
        iu = np.triu_indices_from(within_block, k=1)
        within_vals = within_block[iu]
        between_vals = sim_all[np.ix_(t_idx, o_idx)].flatten()
        gap = float(within_vals.mean() - between_vals.mean())
        gaps.append(gap)
        drug_ids.append(int(t))

        sub = ctx.trained_df.loc[t_mask, feat_cols].to_numpy(dtype=np.float64)
        z = (sub.mean(axis=0) - grand_mean) / grand_std
        mitotnt_z.append(z)

    mitotnt_z = np.stack(mitotnt_z)
    unusualness = np.linalg.norm(mitotnt_z, axis=1)
    rho, pval = spearmanr(unusualness, gaps)

    P.plot_scatter_xy(
        unusualness,
        np.array(gaps),
        "MitoTNT feature L2 distance from global mean (z-scored)",
        "Embedding within−between gap",
        "e11_mitotnt_unusualness_vs_gap",
        f"E11. MitoTNT unusualness vs embedding gap (Spearman ρ={rho:.3f}, p={pval:.3g})",
    )

    top = np.argsort(gaps)[-6:][::-1]
    bot = np.argsort(gaps)[:6]
    findings = {
        "spearman_mitotnt_unusualness_vs_gap": float(rho),
        "p_value": float(pval),
        "top_drugs_by_gap": [ctx.label_names.get(drug_ids[i], str(drug_ids[i])) for i in top],
        "bottom_drugs_by_gap": [ctx.label_names.get(drug_ids[i], str(drug_ids[i])) for i in bot],
    }
    _save_findings(ctx, "E11_phenotype_vs_gap", findings)
    return findings
