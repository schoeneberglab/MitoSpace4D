import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    embeddings = np.load(
        '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings_static_oligo.npy')
    labels = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/labels_static_oligo.npy')

    label_of_interest = 0

    embeddings = embeddings[labels == label_of_interest]

    temporal_dist = np.zeros((embeddings.shape[0], 20, 20))
    for idx, embedding in enumerate(embeddings):
        cos_dists = np.dot(embedding, embedding.T)
        # l2_dists = np.linalg.norm(embedding[:, None] - embedding, axis=2)
        temporal_dist[idx] = cos_dists

    temporal_dist = np.mean(temporal_dist, axis=0)

    sns.set(style="white")
    sns.set_context("notebook", font_scale=0.5)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(temporal_dist, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title(f"Cosine Similarity Heatmap for Label Static Oligomycin")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()