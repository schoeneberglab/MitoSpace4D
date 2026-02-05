import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_PATH = "dummy_kinetics_data.parquet"
EMBEDDING_DIM = 128
NUM_TIMEPOINTS = 10  # t=0 to t=9
SIGMA = 0.1  # Noise level for random walk (lower = smoother trajectory)

# Subset of drugs from your color list
DRUGS = [
    "control", "p110", "myls22", "mfi8", "tbhp", "h2o2",
    "mitoq", "resveratrol", "lonidamine", "oligomycin",
    "dnp", "valinomycin", "cccp"
]

# ==========================================
# GENERATE DATA
# ==========================================
data_rows = []

for drug in DRUGS:
    # 1. Start with a random base vector for this drug (t=0)
    #    This ensures different drugs start in different places in 128D space
    current_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

    # 2. Generate sequential timepoints
    for t in range(NUM_TIMEPOINTS):
        # Add the row
        data_rows.append({
            "drug": drug,
            "time": float(t),
            "embedding": current_embedding.copy()  # Store copy of current state
        })

        # 3. Update embedding for next timepoint (Random Walk)
        #    New = Old + small random noise
        step = np.random.normal(0, SIGMA, EMBEDDING_DIM)
        current_embedding += step

# ==========================================
# SAVE
# ==========================================
df = pd.DataFrame(data_rows)

# Ensure embedding column is object type to hold arrays (parquet requirement)
# Sometimes pandas infers this automatically, but being explicit helps.
print(f"Generated {len(df)} rows.")
print(f"Columns: {df.columns.tolist()}")

# Save to Parquet
df.to_parquet(OUTPUT_PATH, engine="pyarrow")

print(f"✅ Saved to: {os.path.abspath(OUTPUT_PATH)}")
print("You can now point DATA_PATH in your main script to this file.")