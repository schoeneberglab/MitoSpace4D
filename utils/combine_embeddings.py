import numpy as np
import pandas as pd
import os.path as osp
import os


def combine_embeddings(embeddings_dirs, save_dir):
    """ Combines embeddings from multiple directories into a single directory.
    Notes: 
    - Assumes that the embeddings, image paths, label names, and labels are stored in the same format in each directory.
    - Preserves the order of the embeddings per directory and are stacked in the order that they are provided.
    """

    os.makedirs(save_dir, exist_ok=False), "Save directory already exists. Please choose a different directory."

    all_embeddings = []
    all_image_paths = []
    all_label_names = []
    all_labels = []

    for embeddings_dir in embeddings_dirs:
        embeddings = np.load(osp.join(embeddings_dir, 'embeddings_raw.npy'))
        image_paths = np.loadtxt(osp.join(embeddings_dir, 'image_paths.csv'), dtype=str).tolist()
        label_names = np.load(osp.join(embeddings_dir, 'label_names.npy'))
        labels = np.load(osp.join(embeddings_dir, 'labels.npy'))

        all_embeddings.append(embeddings)
        all_label_names.append(label_names)
        all_labels.append(labels)
        all_image_paths.extend(image_paths)
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_label_names = np.concatenate(all_label_names, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_image_paths = np.array(all_image_paths)

    # Save combined data
    np.save(osp.join(save_dir, 'embeddings_raw.npy'), all_embeddings)
    pd.DataFrame(all_image_paths).to_csv(osp.join(save_dir, 'image_paths.csv'), header=False, index=False)
    np.save(osp.join(save_dir, 'label_names.npy'), all_label_names)
    np.save(osp.join(save_dir, 'labels.npy'), all_labels)
    print(f"Combined embeddings from {embeddings_dirs}")
    print(f"Saved to {save_dir}")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='Combine Embeddings')
    # parser.add_argument('--embeddings_dirs', nargs='+', required=True, help='List of directories containing embeddings to combine')
    # parser.add_argument('--save_dir', required=True, help='Directory to save combined embeddings')
    # args = parser.parse_args()

    # assert len(args.embeddings_dirs) > 1, "Please provide at least two directories to combine."

    # combine_embeddings(args.embeddings_dirs, args.save_dir)

    embedding_dirs = [
        "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_20250811",
        "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_20250828",
    ]

    save_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_combined_r20250905"
    combine_embeddings(embedding_dirs, save_dir)
