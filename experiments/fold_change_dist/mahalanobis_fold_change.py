import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis


def calculate_mahalanobis_distances(reference_data, target_data):
    """
    Calculates Mahalanobis distances of all data points relative to the
    distribution of the reference_data (Control).
    """
    # 1. Fit the distribution to the Control group
    mu = np.mean(reference_data, axis=0)
    cov = np.cov(reference_data, rowvar=False)

    # Use pseudo-inverse to handle potential singularity
    inv_cov = np.linalg.pinv(cov)

    # 2. Calculate distances
    d_control = [mahalanobis(p, mu, inv_cov) for p in reference_data]
    d_experiment = [mahalanobis(p, mu, inv_cov) for p in target_data]

    return np.array(d_control), np.array(d_experiment)


def plot_mahalanobis_comparison(control_features, exp_features,
                                control_label='DMSO', exp_label='Treatment'):
    # --- 1. Calculations ---
    raw_d_ctrl, raw_d_exp = calculate_mahalanobis_distances(control_features, exp_features)

    # Normalize to Fold-Change (Control Mean = 1.0)
    norm_factor = np.mean(raw_d_ctrl)
    fc_ctrl = raw_d_ctrl / norm_factor
    fc_exp = raw_d_exp / norm_factor

    # Statistics
    mu_c, sigma_c = np.mean(fc_ctrl), np.std(fc_ctrl, ddof=1)
    mu_e, sigma_e = np.mean(fc_exp), np.std(fc_exp, ddof=1)

    # Z-factor: 1 - (3*sigma_pos + 3*sigma_neg) / |mu_pos - mu_neg|
    z_factor = 1 - (3 * (sigma_c + sigma_e) / abs(mu_c - mu_e))
    cv_percent = (sigma_c / mu_c) * 100

    # T-test
    t_stat, p_val = stats.ttest_ind(fc_ctrl, fc_exp)
    significance_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    # --- 2. Plotting ---

    # Slightly taller figure to accommodate bottom text
    fig, ax1 = plt.subplots(figsize=(4, 6), dpi=150)

    # MANUALLY ADJUST PADDING HERE
    # bottom=0.2 leaves 20% of the figure height empty at the bottom for your text
    # right=0.85 ensures the right y-axis label isn't cut off
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.22)

    # Data setup
    means = [mu_c, mu_e]
    errors = [sigma_c, sigma_e]
    labels = [control_label, exp_label]
    colors = ['#EAE9E9', '#FDE725']  # Grey and Gold

    # Bar Plot
    bars = ax1.bar(labels, means, yerr=errors, capsize=5,
                   color=colors, width=0.6, edgecolor='none', zorder=1)

    # Strip Plot (Jitter)
    np.random.seed(42)
    jitter_c = np.random.normal(0, 0.04, size=len(fc_ctrl))
    jitter_e = np.random.normal(0, 0.04, size=len(fc_exp))

    ax1.scatter(np.zeros_like(fc_ctrl) + jitter_c, fc_ctrl,
                color='grey', alpha=0.6, edgecolor='black', linewidth=0.5, s=30, zorder=2)
    ax1.scatter(np.ones_like(fc_exp) + jitter_e, fc_exp,
                color='grey', alpha=0.6, edgecolor='black', linewidth=0.5, s=30, zorder=2)

    # --- 3. Styling ---

    # Left Axis
    ax1.set_ylabel('Mahalanobis\nFold-change Distance', fontsize=12)

    # Set Y-limit dynamically but with headroom for significance bar
    y_max_data = max(max(fc_ctrl + sigma_c), max(fc_exp + sigma_e))
    ax1.set_ylim(0, y_max_data * 1.3)

    # Right Axis
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel('Fold-change vs DMSO', fontsize=12)

    # Significance Bracket
    x1, x2 = 0, 1
    y_line = y_max_data * 1.15
    y_text = y_line + (y_max_data * 0.02)

    ax1.plot([x1, x1, x2, x2], [y_line * 0.98, y_line, y_line, y_line * 0.98], lw=1, c='k')
    ax1.text((x1 + x2) * .5, y_text, significance_text, ha='center', va='bottom', fontsize=12)

    # Bottom Stats Text
    # Placed at y=0.05 (inside the 0.2 bottom margin we reserved)
    stats_text = f"CV% {control_label}: {cv_percent:.1f}%\nZ-factor: {z_factor:.3f}"
    plt.figtext(0.5, 0.05, stats_text, ha='center', fontsize=11)

    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_4D-embeddings_2024v2-model"
    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_3D-embeddings_Kinetics3D-model"
    target_label = 36  # The specific experimental label you want to analyze

    try:
        # Load Data
        embeddings = np.load(f"{embeddings_dir}/embeddings_raw.npy")
        embeddings = np.mean(embeddings, axis=1)
        labels = np.load(f"{embeddings_dir}/labels.npy")

        # Extract Groups
        control_idxs = np.where(labels == 0)[0]
        control_group = embeddings[control_idxs]

        experiment_idxs = np.where(labels == target_label)[0]

        if len(experiment_idxs) == 0:
            raise ValueError(f"No samples found for label {target_label}")
        experiment_group = embeddings[experiment_idxs]
        print(f"Control samples: {len(control_group)}, Experiment samples: {len(experiment_group)}")

        # Plot Comparison
        plot_mahalanobis_comparison(control_group, experiment_group,
                                    control_label='DMSO', exp_label=f' {target_label}')
    except Exception as e:
        print(f"Error: {e}")