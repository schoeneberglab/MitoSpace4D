import pandas as pd

dst_file = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_random_init/embeddings+metadata_vis_joined.parquet"
src_file = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_252eps/embeddings+metadata_vis_joined.parquet"

dst = pd.read_parquet(dst_file)
src = pd.read_parquet(src_file)

feature_cols = [c for c in src.columns if c not in dst.columns]
print(f"Copying {len(feature_cols)} classical feature columns from src -> dst")

src_subset = src[["image_paths"] + feature_cols]
merged = dst.merge(src_subset, on="image_paths", how="left")

assert len(merged) == len(dst), f"row count changed: {len(dst)} -> {len(merged)}"
assert merged["embeddings"].equals(dst["embeddings"]), "embeddings changed"
assert merged["embeddings_umap"].equals(dst["embeddings_umap"]), "embeddings_umap changed"

merged.to_parquet(dst_file)
print(f"Wrote {len(merged)} rows with {len(merged.columns)} columns to {dst_file}")
