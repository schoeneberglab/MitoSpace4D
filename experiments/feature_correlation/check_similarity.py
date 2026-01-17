import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

emb_file = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-all-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm/embeddings_raw.npy"
label_file = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_kinetics-all-60frames_embeddings_resnet3d-kinetics-300eps_ablated-tmrm/labels.npy"

embeddings = np.load(emb_file)
labels = np.load(label_file)

pick_labels = 14

label_indices = np.where(labels == pick_labels)[0]
embeddings = embeddings[label_indices]

# sample = embeddings[0]
sample = embeddings[np.random.randint(0, embeddings.shape[0])]
dist = sample @ sample.T # calculate the dot product (cosine similarity)

# Plot the heatmap of the cosine similarity matrix
sns.heatmap(dist, vmax=1, vmin=0.8)
plt.title(f"Cosine Similarity of Sample {pick_labels}")
plt.show()
