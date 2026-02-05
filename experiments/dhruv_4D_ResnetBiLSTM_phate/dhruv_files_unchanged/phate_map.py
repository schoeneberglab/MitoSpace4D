import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import phate
import umap
from scipy.stats import spearmanr
import os

# ===========================================================
#                  LOAD NEW DATAFRAME
# ===========================================================

# data_path = "/home/dhruvagarwal/projects/MitoSpace4D/temporal_experiments/expanded_embeddings_with_time.parquet"
data_path = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilistm_encoder_consistent_temporal/embeddings_kinetics_data_60frames/expanded_embeddings_with_time.parquet'
print("Loading exploded DataFrame...")
df = pd.read_parquet(data_path, engine="pyarrow")  # parquet is faster than CSV

# Convert embeddings column to numpy array
X = np.stack(df['embedding'].values)
print("Embeddings shape:", X.shape)

# Ensure 'drug' and 'time' exist
if 'drug' not in df.columns or 'time' not in df.columns:
    raise ValueError("DataFrame must have 'drug' and 'time' columns")

meta = df[['drug', 'time']].copy()

# ===========================================================
#            LOAD COLORS FROM colors.txt
# ===========================================================

colors_path = "/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/colors.txt"

color_dict = {}
with open(colors_path, 'r') as f:
    for line in f:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        # columns: date, drug_name, idx, R, G, B
        drug_name = parts[1]
        r, g, b = map(int, parts[3:6])
        color_dict[drug_name] = (r/255, g/255, b/255)  # matplotlib expects 0-1

# Filter to drugs in meta
drug_colors = {drug: color_dict[drug] for drug in meta['drug'].unique() if drug in color_dict}

# ===========================================================
#               COMPUTE PHATE & UMAP
# ===========================================================

print("\nRunning PHATE...")
ph = phate.PHATE(n_components=2, knn=30, decay=40, t='auto', random_state=0)
X_phate = ph.fit_transform(X)
meta["PHATE1"] = X_phate[:,0]
meta["PHATE2"] = X_phate[:,1]
print("PHATE complete.")

print("\nRunning UMAP...")
um = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0)
X_umap = um.fit_transform(X)
meta["UMAP1"] = X_umap[:,0]
meta["UMAP2"] = X_umap[:,1]
print("UMAP complete.")

# ===========================================================
#               PLOTTING FUNCTIONS
# ===========================================================

def plot_embedding_with_colors(meta, x, y, color_dict, title, fname):
    plt.figure(figsize=(8,8))
    for drug_name, color in color_dict.items():
        subset = meta[meta['drug'] == drug_name]
        plt.scatter(subset[x], subset[y], color=color, s=5, label=drug_name)
    plt.xlabel(x); plt.ylabel(y)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved:", fname)

# PHATE colored by drug
plot_embedding_with_colors(meta, "PHATE1", "PHATE2", drug_colors,
                           "PHATE colored by drug", "phate_drugs.png")

# UMAP colored by drug
plot_embedding_with_colors(meta, "UMAP1", "UMAP2", drug_colors,
                           "UMAP colored by drug", "umap_drugs.png")

# ===========================================================
#         TRAJECTORIES (CENTROIDS PER DRUG & TIME)
# ===========================================================

print("\nPlotting drug trajectories...")

for embx, emby, prefix in [("PHATE1", "PHATE2", "phate"),
                           ("UMAP1", "UMAP2", "umap")]:

    plt.figure(figsize=(10,8))

    for drug_name, base_color in drug_colors.items():
        subset = meta[meta["drug"] == drug_name]

        # compute centroids per time
        centroids = subset.groupby("time")[[embx, emby]].mean().sort_index()
        xs = centroids[embx].to_numpy()
        ys = centroids[emby].to_numpy()
        time_vals = centroids.index.to_numpy()

        # ----- Region-wise shade mapping (lighter → darker) -----
        num_t = len(time_vals)
        if num_t == 1:
            shades = [base_color]
        else:
            shades = []
            base = np.array(base_color)

            for i in range(num_t):
                # 0 → lightest, last → darkest
                shade_factor = 0.35 + 0.65 * (i / (num_t - 1))
                shade = tuple(base * shade_factor)
                shades.append(shade)

        # ----- Plot shaded segments -----
        for i in range(num_t - 1):
            plt.plot(
                xs[i:i+2], ys[i:i+2],
                color=shades[i], linewidth=3
            )

            # Direction arrow
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]

            plt.quiver(
                xs[i], ys[i],
                dx, dy,
                angles='xy', scale_units='xy', scale=1,
                color=shades[i],
                width=0.004, headwidth=3, headlength=4,
                alpha=0.9
            )

        # Mark points
        plt.scatter(xs, ys, c=shades, s=25, label=drug_name)

    plt.xlabel(embx)
    plt.ylabel(emby)
    plt.title(f"{prefix.upper()} Trajectories with Arrows + Shaded Time")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{prefix}_trajectories_kinetics_arrows_shaded.png", dpi=300)
    plt.close()
    print("Saved:", f"{prefix}_trajectories_kinetics_arrows_shaded.png")

# ===========================================================
#         TEMPORAL ORDER METRICS (SPEARMAN RHO)
# ===========================================================

print("\nComputing temporal preservation metrics...")

results = []
for drug_name in meta['drug'].unique():
    g = meta[meta["drug"] == drug_name]

    rho_ph, _ = spearmanr(g["time"], g["PHATE1"])
    rho_um, _ = spearmanr(g["time"], g["UMAP1"])

    results.append({"drug": drug_name,
                    "rho_PHATE_time": rho_ph,
                    "rho_UMAP_time": rho_um})

df_results = pd.DataFrame(results)
df_results.to_csv("temporal_kinetics_correlation.csv", index=False)
print("\nSaved: temporal_correlation_kinetics.csv")
print(df_results)

print("\nDONE. Generated files:")
print("""
phate_drugs_kinetics.png
umap_drugs_kinetics.png
phate_trajectories_kinetics.png
umap_trajectories_kinetics.png
temporal_correlation_kinetics.csv
""")