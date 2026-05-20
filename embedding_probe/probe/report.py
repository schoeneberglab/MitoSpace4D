"""Render per-model and comparative findings reports."""

from __future__ import annotations

import json
from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parents[1]
OPENPHENOM_REF = PROBE_ROOT / "openphenom_reference.json"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _load_numbers(model_id: str) -> dict | None:
    path = PROBE_ROOT / "results" / model_id / "numbers.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def render_model(model_id: str) -> Path:
    numbers = _load_numbers(model_id)
    if numbers is None:
        raise FileNotFoundError(f"No results for {model_id}")

    display = {"ms4d": "MitoSpace4D (4D)", "ms3d": "MitoSpace3D (3D)", "ms2d": "MitoSpace2D (2D)"}.get(
        model_id, model_id
    )
    out_dir = PROBE_ROOT / "results" / model_id
    plots_dir = out_dir / "plots"

    lines = [
        f"# {display} — embedding probe findings",
        "",
        "Diagnostics aligned with [what-does-openphenom-learn](https://github.com/drv-agwl/what-does-openphenom-learn).",
        "",
        "## Headline metrics",
        "",
    ]

    if "E1_geometry" in numbers:
        e1 = numbers["E1_geometry"]
        dim = e1.get("embedding_dim", "?")
        pr = e1.get("effective_rank_participation_ratio")
        lines += [
            f"- **Participation ratio:** {_fmt(pr)} / {dim}",
            f"- **Top-1 PC variance share:** {_fmt(e1.get('top1_eigenvalue_share'))}",
            f"- **Mean pairwise cosine similarity:** {_fmt(e1.get('pairwise_sim_mean'))}",
            "",
        ]

    if "E2_trained_vs_random" in numbers:
        e2 = numbers["E2_trained_vs_random"]
        lines += [
            f"- **Trained vs random-init Spearman ρ:** {_fmt(e2.get('spearman_pairwise_sim_rankings'))}",
            f"- **Within−between gap (trained / random):** {_fmt(e2.get('trained_gap'))} / {_fmt(e2.get('random_gap'))}",
            f"- **Replicate mAP (trained / random / floor):** {_fmt(e2.get('trained_replicate_map'))} / {_fmt(e2.get('random_replicate_map'))} / {_fmt(e2.get('random_ranking_floor_map'))}",
            "",
        ]

    if "E4_discriminability" in numbers:
        e4 = numbers["E4_discriminability"]
        lines += [
            f"- **Raw gap / mAP:** {_fmt(e4.get('raw_gap'))} / {_fmt(e4.get('raw_map'))}",
            f"- **PCA-CenterScale gap / mAP:** {_fmt(e4.get('post_pca_centerscale_gap'))} / {_fmt(e4.get('post_pca_centerscale_map'))}",
            "",
        ]

    if "E10_cossim_by_condition" in numbers:
        for enc, stats in numbers["E10_cossim_by_condition"].get("effect_sizes", {}).items():
            lines.append(
                f"- **{enc}:** Cohen's d={_fmt(stats.get('cohen_d'))}, AUROC={_fmt(stats.get('auc'))}, gap={_fmt(stats.get('gap'))}"
            )
        lines.append("")

    if "E11_phenotype_vs_gap" in numbers and "spearman_mitotnt_unusualness_vs_gap" in numbers["E11_phenotype_vs_gap"]:
        e11 = numbers["E11_phenotype_vs_gap"]
        lines.append(
            f"- **MitoTNT unusualness vs embedding gap (Spearman):** {_fmt(e11['spearman_mitotnt_unusualness_vs_gap'])} (p={_fmt(e11.get('p_value'))})"
        )
        lines.append("")

    lines += ["## Full metrics", ""]
    for name, payload in numbers.items():
        lines += [f"### {name}", "", "```json", json.dumps(payload, indent=2), "```", ""]

    if plots_dir.is_dir():
        lines += ["## Figures", ""]
        for p in sorted(plots_dir.glob("*.png")):
            lines.append(f"![{p.stem}](plots/{p.name})")
            lines.append("")

    path = out_dir / "findings.md"
    path.write_text("\n".join(lines))
    return path


def render_comparative() -> Path:
    """Cross-model + OpenPhenom summary (main findings.md)."""
    mitospace_models = ["ms4d", "ms3d", "ms2d"]
    all_nums = {m: _load_numbers(m) for m in mitospace_models}
    op = json.loads(OPENPHENOM_REF.read_text()) if OPENPHENOM_REF.is_file() else {}

    names = {
        "ms4d": "MitoSpace4D (4D)",
        "ms3d": "MitoSpace3D (3D)",
        "ms2d": "MitoSpace2D (2D)",
        "openphenom": "OpenPhenom (RxRx3-core)",
    }

    def row(metric_fn, section: str, key: str):
        cells = []
        for m in mitospace_models:
            n = all_nums.get(m)
            val = metric_fn(n.get(section, {})) if n and section in n else None
            cells.append(_fmt(val))
        op_val = metric_fn(op.get(section, {})) if op else "—"
        return f"| {key} | {op_val} | " + " | ".join(cells) + " |"

    lines = [
        "# Embedding probe — comparative findings",
        "",
        "Cross-study of **MitoSpace 4D / 3D / 2D** self-supervised embeddings vs **OpenPhenom** "
        "(Cell Painting, RxRx3-core), using the experimental framing from "
        "[what-does-openphenom-learn](https://github.com/drv-agwl/what-does-openphenom-learn) and "
        "Svatko et al. (2026) [*Deep Learning for BioImaging: What Are We Learning?*](https://arxiv.org/abs/2603.13377).",
        "",
        "> OpenPhenom numbers are from the reference run documented in the external probe repository "
        "(~2.4k wells, 300 treatments). MitoSpace numbers use the 2024v3 drug screen "
        "(~36.6k cells, 25 drugs, shared `cell_id` cohort where noted).",
        "",
        "## Summary table",
        "",
        "| Metric | OpenPhenom | MS4D (4D) | MS3D (3D) | MS2D (2D) |",
        "|--------|------------|-----------|-----------|-----------|",
    ]

    lines.append(
        row(lambda d: d.get("effective_rank_participation_ratio"), "E1_geometry", "Participation ratio (effective rank)")
    )
    lines.append(
        row(lambda d: d.get("top1_eigenvalue_share"), "E1_geometry", "Top-1 PC variance share")
    )
    lines.append(row(lambda d: d.get("pairwise_sim_mean"), "E1_geometry", "Mean pairwise cosine similarity"))
    def trained_gap(n):
        if not n:
            return None
        if "E2_trained_vs_random" in n:
            return n["E2_trained_vs_random"].get("trained_gap")
        es = n.get("E10_cossim_by_condition", {}).get("effect_sizes", {})
        for k, v in es.items():
            if "random" not in k.lower():
                return v.get("gap")
        return None

    lines.append(
        f"| Within−between gap (trained) | {_fmt(op.get('E2_trained_vs_random', {}).get('trained_gap'))} | "
        + " | ".join(_fmt(trained_gap(all_nums.get(m))) for m in mitospace_models)
        + " |"
    )
    lines.append(
        row(lambda d: d.get("spearman_pairwise_sim_rankings"), "E2_trained_vs_random", "Spearman ρ vs random baseline")
    )
    lines.append(row(lambda d: d.get("trained_replicate_map"), "E2_trained_vs_random", "Replicate retrieval mAP"))
    lines.append(row(lambda d: d.get("raw_map"), "E4_discriminability", "Replicate mAP (balanced sample, raw)"))
    lines.append(
        row(lambda d: d.get("post_pca_centerscale_map"), "E4_discriminability", "Replicate mAP after PCA-CenterScale")
    )

    def e10_auc(n):
        if not n:
            return None
        es = n.get("E10_cossim_by_condition", {}).get("effect_sizes", {})
        for k, v in es.items():
            if "random" not in k.lower():
                return v.get("auc")
        return None

    lines.append(f"| AUROC (within vs between pairs) | {_fmt(op.get('E10_cossim_by_condition', {}).get('effect_sizes', {}).get('Trained OpenPhenom', {}).get('auc'))} | " + " | ".join(_fmt(e10_auc(all_nums.get(m))) for m in mitospace_models) + " |")

    n4, n3, n2 = all_nums.get("ms4d"), all_nums.get("ms3d"), all_nums.get("ms2d")
    e11_4d = (n4 or {}).get("E11_phenotype_vs_gap", {})
    e11_2d = (n2 or {}).get("E11_phenotype_vs_gap", {})

    lines += [
        "",
        "## How to read these metrics (plain language)",
        "",
        "### Participation ratio — not the same as “% of dimensions used”",
        "",
        "The table lists **participation ratio** next to embedding size (e.g. 16 / 2048). That ratio is an "
        "**effective number of PCA directions** that carry variance: ~1 means almost all cells vary along one "
        "dominant axis (collapsed); ~16 means variance is spread across roughly sixteen directions.",
        "",
        "If you divide by embedding dimension (16/2048 ≈ 0.8% vs OpenPhenom 1.8/384 ≈ 0.5%), the gap looks tiny. "
        "That is misleading. Compare instead:",
        "",
        "| | OpenPhenom | MS4D | MS3D | MS2D |",
        "|--|------------|------|------|------|",
        "| **Absolute participation ratio** | **~2** | ~16 | ~15 | **~20** |",
        "| **Top-1 PC variance share** | **~74%** | ~13% | ~15% | ~14% |",
        "| **Mean pairwise cosine similarity** | **~0.94** | ~0.23 | ~0.22 | ~0.22 |",
        "",
        "OpenPhenom is collapsed in **absolute** effective rank and in **how similar all pairs are** (0.94). "
        "MitoSpace models use ~9–11× more effective directions and much lower average similarity. "
        "**2D has the highest participation ratio among MitoSpace models** (19.5), not the lowest.",
        "",
        "### Two questions: “separate drugs” vs “match extreme MitoTNT”",
        "",
        "The probe asks separate things:",
        "",
        "1. **Drug structure in raw cosine space** — Are cells treated with the same drug more similar to each "
        "other than to other drugs? (within−between gap, AUROC, raw mAP.)",
        "2. **Link to MitoTNT severity (E11)** — For each drug, is its embedding separation large when its "
        "MitoTNT profile is extreme compared with all cells?",
        "",
        "**2D captures most of (1):** gaps and AUROC are close to 3D/4D (e.g. raw mAP ~0.27 vs ~0.30). A single "
        "2D mitochondrial image already supports strong drug clustering.",
        "",
        "**4D is clearer on (2):** Spearman ρ between MitoTNT “unusualness” and per-drug embedding gap is "
        f"**{_fmt(e11_4d.get('spearman_mitotnt_unusualness_vs_gap'))}** "
        f"(p≈{_fmt(e11_4d.get('p_value'))}) for 4D vs "
        f"**{_fmt(e11_2d.get('spearman_mitotnt_unusualness_vs_gap'))}** "
        f"(not significant) for 2D. So 4D separation tracks severe mitochondrial remodeling more reliably; "
        "2D can separate drugs without that separation scaling as cleanly with MitoTNT outliers.",
        "",
        "### What is “PCA-CenterScale” and “replicate mAP”?",
        "",
        "**Replicate retrieval** is a practical test: pretend you “lost” the drug label of one cell, find its "
        "nearest neighbors in embedding space by cosine similarity, and ask whether those neighbors are mostly "
        "the **same drug** (technical replicates / same treatment).",
        "",
        "- **mAP (mean average precision)** summarizes how highly same-drug cells rank among neighbors. "
        "  - **0** ≈ useless (random guessing).",
        "  - **~0.05** ≈ random floor on this benchmark.",
        "  - **~0.30** ≈ strong retrieval for 25 drugs.",
        "  - **1.0** ≈ perfect.",
        "",
        "**Raw mAP** uses embeddings exactly as the model outputs them.",
        "",
        "**PCA-CenterScale** is a post-processing step (same idea as Recursion’s OpenPhenom recipe, adapted here):",
        "",
        "1. Take **control (untreated) cells** only.",
        "2. Fit **PCA** on their embeddings — captures the main shared “baseline mitochondrial look”.",
        "3. **Center and scale** in that PCA space (StandardScaler fit on controls).",
        "4. Apply that transform to **all** cells (including drugs).",
        "",
        "Intuition: many embeddings share a big common direction (imaging conditions, cell health baseline). "
        "PCA-CenterScale **subtracts that shared bulk** so neighbors are chosen more from **drug-specific residual** "
        "structure than from “everyone looks a bit like control”.",
        "",
        "On this cohort (25 drugs × 20 cells per drug in E4):",
        "",
    ]

    if n4 and n2:
        e4_4d, e4_2d = n4.get("E4_discriminability", {}), n2.get("E4_discriminability", {})
        lines += [
            f"| | MS4D (4D) | MS2D (2D) |",
            f"|--|-----------|-----------|",
            f"| Raw replicate mAP | {_fmt(e4_4d.get('raw_map'))} | {_fmt(e4_2d.get('raw_map'))} |",
            f"| mAP after PCA-CenterScale | **{_fmt(e4_4d.get('post_pca_centerscale_map'))}** | {_fmt(e4_2d.get('post_pca_centerscale_map'))} |",
            f"| Random ranking floor | ~{_fmt(e4_4d.get('random_floor_map'))} | ~{_fmt(e4_2d.get('random_floor_map'))} |",
            "",
            "4D **gains** more from correction (0.30 → 0.33) than 2D (0.27 → 0.23). So when you explicitly "
            "remove the control-dominated subspace, **4D’s neighbors are more drug-faithful** — better for "
            "“find my replicate / same treatment” workflows. 2D already separates drugs reasonably in raw space "
            "but benefits less (here, mAP slightly drops after correction, suggesting more drug signal was "
            "already in the raw 512-d directions or the control subspace differs).",
            "",
        ]

    lines += [
        "**When to care:** use **raw mAP / gap** for “does the model cluster drugs at all?”; use "
        "**PCA-CenterScale mAP** for “after removing baseline morphology, can I retrieve same-treatment cells?” "
        "— especially relevant when comparing to OpenPhenom (raw mAP ~0.02, corrected ~0.03).",
        "",
        "## Short answer",
        "",
        "**All three MitoSpace dimensionalities learn strong, wide drug-discriminative embeddings on this mitochondrial screen; "
        "OpenPhenom on RxRx3-core is comparatively collapsed and weakly discriminative in the reference probe.** "
        "3D matches or slightly exceeds 4D on raw within−between gap; 4D leads on PCA-CenterScale mAP and MitoTNT–gap correlation. "
        "2D is competitive at 512-d but weakest on replicate retrieval after post-processing.",
        "",
        "## MitoSpace 4D vs 3D vs 2D (same cells, same drugs)",
        "",
    ]

    if n4 and n3 and n2:
        lines += [
            "| Metric | MS4D (4D) | MS3D (3D) | MS2D (2D) |",
            "|--------|-----------|-----------|-----------|",
            f"| Embedding dim | {n4['E1_geometry'].get('embedding_dim')} | {n3['E1_geometry'].get('embedding_dim')} | {n2['E1_geometry'].get('embedding_dim')} |",
            f"| Participation ratio | {_fmt(n4['E1_geometry'].get('effective_rank_participation_ratio'))} | {_fmt(n3['E1_geometry'].get('effective_rank_participation_ratio'))} | {_fmt(n2['E1_geometry'].get('effective_rank_participation_ratio'))} |",
            f"| Raw within−between gap (E4/E10) | {_fmt(trained_gap(n4))} | {_fmt(trained_gap(n3))} | {_fmt(trained_gap(n2))} |",
            f"| Raw replicate mAP (E4) | {_fmt(n4['E4_discriminability'].get('raw_map'))} | {_fmt(n3['E4_discriminability'].get('raw_map'))} | {_fmt(n2['E4_discriminability'].get('raw_map'))} |",
            f"| mAP after PCA-CenterScale | {_fmt(n4['E4_discriminability'].get('post_pca_centerscale_map'))} | {_fmt(n3['E4_discriminability'].get('post_pca_centerscale_map'))} | {_fmt(n2['E4_discriminability'].get('post_pca_centerscale_map'))} |",
            f"| AUROC within vs between (E10) | {_fmt(e10_auc(n4))} | {_fmt(e10_auc(n3))} | {_fmt(e10_auc(n2))} |",
            f"| MitoTNT unusualness vs gap ρ (E11) | {_fmt(n4.get('E11_phenotype_vs_gap', {}).get('spearman_mitotnt_unusualness_vs_gap'))} | {_fmt(n3.get('E11_phenotype_vs_gap', {}).get('spearman_mitotnt_unusualness_vs_gap'))} | {_fmt(n2.get('E11_phenotype_vs_gap', {}).get('spearman_mitotnt_unusualness_vs_gap'))} |",
            "",
            "**Ranking (this panel):** discriminability **3D ≈ 4D > 2D**; post-processed replicate retrieval **4D > 3D > 2D**; "
            "link to extreme MitoTNT phenotypes **4D only (significant)**.",
            "",
        ]

    lines += [
        "## Interpretation",
        "",
        "### Geometry (E1)",
        "",
    ]

    for m in mitospace_models:
        n = all_nums.get(m)
        if not n or "E1_geometry" not in n:
            lines.append(f"- **{names[m]}:** *Not run or data unavailable.*")
            continue
        e1 = n["E1_geometry"]
        pr, dim = e1.get("effective_rank_participation_ratio"), e1.get("embedding_dim")
        lines.append(
            f"- **{names[m]}:** participation ratio **{_fmt(pr)} / {dim}**, "
            f"mean pairwise cos-sim **{_fmt(e1.get('pairwise_sim_mean'))}** "
            f"(OpenPhenom ≈ 0.94 on 384-d — highly collapsed)."
        )

    lines += [
        "",
        "### Training signal vs architectural prior (E2, E7)",
        "",
        "- **OpenPhenom:** Pairwise rankings largely shared with a **vanilla random ViT** (Spearman ρ ≈ 0.36); "
        "same-architecture random twin is *not* comparable (ρ ≈ −0.03).",
    ]

    for m in mitospace_models:
        n = all_nums.get(m)
        if not n or "E2_trained_vs_random" not in n:
            continue
        e2 = n["E2_trained_vs_random"]
        lines.append(
            f"- **{names[m]}:** Spearman ρ vs random-init = **{_fmt(e2.get('spearman_pairwise_sim_rankings'))}**; "
            f"mAP trained **{_fmt(e2.get('trained_replicate_map'))}** vs random **{_fmt(e2.get('random_replicate_map'))}** "
            f"(floor **{_fmt(e2.get('random_ranking_floor_map'))}**)."
        )

    lines += [
        "",
        "### Drug discriminability (E4, E10)",
        "",
        "- **OpenPhenom:** Small absolute gaps; subtle pharmacology is hard without PCA-CenterScale.",
    ]

    for m in mitospace_models:
        n = all_nums.get(m)
        if not n:
            continue
        if "E4_discriminability" in n:
            e4 = n["E4_discriminability"]
            lines.append(
                f"- **{names[m]}:** raw gap **{_fmt(e4.get('raw_gap'))}**, mAP **{_fmt(e4.get('raw_map'))}**; "
                f"after PCA-CenterScale mAP **{_fmt(e4.get('post_pca_centerscale_map'))}**."
            )

    lines += [
        "",
        "### Morphology vs embedding gap (E11)",
        "",
        f"- **OpenPhenom:** pixel unusualness vs embedding gap Spearman ρ ≈ **{_fmt(op.get('E11_phenotype_vs_gap', {}).get('spearman_pixel_unusualness_vs_gap'))}** (weak).",
    ]

    for m in mitospace_models:
        n = all_nums.get(m)
        if n and "E11_phenotype_vs_gap" in n and "spearman_mitotnt_unusualness_vs_gap" in n["E11_phenotype_vs_gap"]:
            e11 = n["E11_phenotype_vs_gap"]
            lines.append(
                f"- **{names[m]}:** MitoTNT unusualness vs gap ρ = **{_fmt(e11['spearman_mitotnt_unusualness_vs_gap'])}** "
                f"(p={_fmt(e11.get('p_value'))})."
            )

    lines += [
        "",
        "## Per-model reports",
        "",
        "| Model | Report |",
        "|-------|--------|",
    ]
    for m in mitospace_models:
        p = PROBE_ROOT / "results" / m / "findings.md"
        if p.is_file():
            lines.append(f"| {names[m]} | [results/{m}/findings.md](results/{m}/findings.md) |")
        else:
            lines.append(f"| {names[m]} | *pending* |")

    lines += [
        "",
        "## Practical takeaways",
        "",
        "1. **MitoSpace (4D/3D/2D)** on this mitochondrial drug panel shows **much wider** embedding geometry and "
        "**stronger drug separation** than OpenPhenom on RxRx3-core in the reference probe.",
        "2. **Random-init MitoSpace4D** does *not* preserve ranking structure (unlike OpenPhenom’s ViT prior) — "
        "training is doing substantial work.",
        "3. **PCA-CenterScale** (control-calibrated) remains useful for replicate retrieval on MitoSpace, "
        "analogous to OpenPhenom’s post-processing recipe.",
        "4. **Temporal (4D) vs volumetric (3D) vs projection (2D):** adding time helps post-processing and "
        "MitoTNT alignment; 3D alone is nearly as discriminative in raw cosine space; 2D is the most compressed baseline.",
        "",
    ]

    path = PROBE_ROOT / "results" / "findings.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def render(model_id: str | None = None) -> Path:
    if model_id:
        return render_model(model_id)
    return render_comparative()
