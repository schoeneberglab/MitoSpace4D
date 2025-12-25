import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    embeddings = np.load(
        '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings_static_h2o2.npy')
    labels = np.load('/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/labels_static_h2o2.npy')

    label_of_interest_oligo = 5
    label_of_interest_oligo_static = 27
    embeddings_oligo = embeddings[labels == label_of_interest_oligo]
    embeddings_oligo_static = embeddings[labels == label_of_interest_oligo_static]

    # find the distances between every timestep for the same sample
    diff = np.linalg.norm(embeddings_oligo - embeddings_oligo_static, axis=2)
    diff = np.mean(diff, axis=0)

    # plot the difference
    plt.figure(figsize=(10, 8))
    sns.set(style="white")
    sns.set_context("notebook", font_scale=0.5)
    ax = sns.heatmap(diff, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title(f"Cosine Similarity Heatmap for Label Static Oligomycin")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()