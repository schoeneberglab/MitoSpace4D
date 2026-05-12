import os.path as osp
import numpy as np
import pandas as pd

if __name__ == '__main__':
    emb_root = '/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace'
    emb = np.load(osp.join(emb_root, 'OrganoAgeSpace.npy'))  # Shape: (N, 2048)
    identifiers = np.load(osp.join(emb_root, 'identifiers.npy'))
    labels = np.load(osp.join(emb_root, 'labels.npy'))

    label_map = {0: 0, 1: 1, 2: 2, 15: 3, 16: 4, 17: 5, 20: 6, 67: 7, 68: 8, 131: 9, 132: 10}
    labels_mapped = np.array([label_map[label] for label in labels])

    mitotnt_feats = pd.read_csv('/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/df_all_bins_20250303.csv')
    mitotnt_feats = mitotnt_feats.rename(columns={'Unnamed: 0': 'Identifier'})

    # Create a DataFrame for embeddings
    emb_cols = [f'Emb_{i}' for i in range(2048)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, 'Identifier', identifiers)  # Insert image names as the first column

    # add the labels column
    emb_df.insert(1, 'Label', labels)

    # add the mapped labels column
    emb_df.insert(2, 'Mapped Label', labels_mapped)

    # Merge efficiently instead of looping
    mitotnt_feats = mitotnt_feats.merge(emb_df, on='Identifier', how='left')

    # Save the updated dataframe
    mitotnt_feats.to_csv('/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/df_all_bins_20250303_embeddings.csv', index=False)
