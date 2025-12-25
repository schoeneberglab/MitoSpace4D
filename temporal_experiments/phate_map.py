import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import phate
import umap
from scipy.stats import spearmanr
import ast  # for converting string embeddings in CSV to np.array if needed
import os

# ===========================================================
#                  LOAD NEW DATAFRAME
# ===========================================================

data_path = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/df.csv"
print("Loading exploded DataFrame...")
df = pd.read_csv(data_path)

# If embeddings are stored as strings in CSV, convert to np.array
if isinstance(df.loc[0, 'embedding'], str):
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

# Convert embeddings to numpy array of shape (n_cells, emb_dim)
X = np.stack(df['embedding'].values)
print("Embeddings shape:", X.shape)

# Ensure 'labels' and 'time' exist
if 'labels' not in df.columns or 'time' not in df.columns:
    raise ValueError("DataFrame must have 'labels' and 'time' columns")

meta = df[['labels', 'time']].copy()
meta.rename(columns={'labels':'drug'}, inplace=True)  # align naming with previous code

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

def plot_embedding(meta, x, y, color, title, fname):
    plt.figure(figsize=(7,7))
    sc = plt.scatter(meta[x], meta[y], c=meta[color], cmap='viridis', s=3)
    plt.xlabel(x); plt.ylabel(y)
    plt.title(title)
    plt.colorbar(sc, label=color)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved:", fname)

# PHATE colored by time
plot_embedding(meta, "PHATE1", "PHATE2", "time",
               "PHATE colored by time", "phate_time.png")

# UMAP colored by time
plot_embedding(meta, "UMAP1", "UMAP2", "time",
               "UMAP colored by time", "umap_time.png")

# ===========================================================
#         TRAJECTORIES (CENTROIDS PER DRUG & TIME)
# ===========================================================

print("\nPlotting drug trajectories...")

for embx, emby, prefix in [("PHATE1", "PHATE2", "phate"),
                           ("UMAP1", "UMAP2", "umap")]:
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=meta, x=embx, y=emby,
                    hue="time", palette="viridis", s=5, legend=False)

    for drug in meta['drug'].unique():
        subset = meta[meta["drug"] == drug]
        centroids = subset.groupby("time")[[embx, emby]].mean()

        x = centroids[embx].to_numpy()
        y = centroids[emby].to_numpy()
        plt.plot(x, y, '-o', label=f"drug {drug}")

    plt.legend()
    plt.title(f"{prefix.upper()} Trajectories (Centroids per time)")
    plt.tight_layout()
    plt.savefig(f"{prefix}_trajectories.png", dpi=300)
    plt.close()
    print("Saved:", f"{prefix}_trajectories.png")

# ===========================================================
#         TEMPORAL ORDER METRICS (SPEARMAN RHO)
# ===========================================================

print("\nComputing temporal preservation metrics...")

results = []
for drug in meta['drug'].unique():
    g = meta[meta["drug"] == drug]

    rho_ph, _ = spearmanr(g["time"], g["PHATE1"])
    rho_um, _ = spearmanr(g["time"], g["UMAP1"])

    results.append({"drug": drug,
                    "rho_PHATE_time": rho_ph,
                    "rho_UMAP_time": rho_um})

df_results = pd.DataFrame(results)
df_results.to_csv("temporal_correlation.csv", index=False)
print("\nSaved: temporal_correlation.csv")
print(df_results)

print("\nDONE. Generated files:")
print("""
phate_time.png
umap_time.png
phate_trajectories.png
umap_trajectories.png
temporal_correlation.csv
""")
