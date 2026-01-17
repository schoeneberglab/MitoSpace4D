import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

infile = "/home/earkfeld/Projects/MitoSpace4D/runs/embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220/w2_distance_matrix.npy"

conditions = ["NTC", "PTEN_C4", "PTEN_C5", "NTC_cisplatin", "PTEN_C4_cisplatin", "PTEN_C5_cisplatin"]

w2_matrix = np.load(infile)

# Plot the W2 distance matrix
sns.heatmap(w2_matrix, annot=True, cmap="Blues", fmt=".4f", xticklabels=conditions, yticklabels=conditions)
plt.title("Wasserstein Distance Matrix")
plt.tight_layout()
plt.show()

