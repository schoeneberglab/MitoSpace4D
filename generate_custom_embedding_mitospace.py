import os
import os.path as osp
import argparse
import numpy as np
import umap

from utils.vis import make_mitospace_minimal


def load_folder_label_maps(drugs_to_labels_path):
    folder_to_label = {}
    label_to_drug = {}
    with open(drugs_to_labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            folder, drug, label = parts
            label_int = int(label)
            folder_to_label[folder] = label_int
            label_to_drug[label_int] = drug
    # Build label_names array (sorted by label)
    if label_to_drug:
        max_label = max(label_to_drug.keys())
        label_names = np.array([label_to_drug.get(i, f"label_{i}") for i in range(max_label + 1)], dtype=object)
    else:
        label_names = np.array([], dtype=object)
    return folder_to_label, label_names


def load_colors(colors_file_path):
    colors = {}
    if not osp.exists(colors_file_path):
        return None
    with open(colors_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:
                _, _, index, r, g, b = parts
                r_f, g_f, b_f = float(r), float(g), float(b)
                if r_f >= 1 or g_f >= 1 or b_f >= 1:
                    colors[int(index)] = [r_f / 255, g_f / 255, b_f / 255]
                else:
                    colors[int(index)] = [r_f, g_f, b_f]
    return colors if colors else None


def maybe_build_umap_embeddings(embeddings_dir, folder_to_label, label_names):
    emb_umap_path = osp.join(embeddings_dir, 'embeddings_umap.npy')
    labels_path = osp.join(embeddings_dir, 'labels.npy')
    label_names_path = osp.join(embeddings_dir, 'label_names.npy')

    if osp.exists(emb_umap_path) and osp.exists(labels_path) and osp.exists(label_names_path):
        return  # Nothing to do

    files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy') and f.startswith('embeddings_20')])
    if not files:
        raise FileNotFoundError(f"No per-sample embeddings found in {embeddings_dir}")

    all_embeddings = []
    all_labels = []
    for fname in files:
        fpath = osp.join(embeddings_dir, fname)
        emb = np.load(fpath)
        emb = emb.reshape(1, -1)
        all_embeddings.append(emb)
        # Infer folder key from filename: embeddings_<folder>_*.npy
        parts = osp.basename(fname).split('_')
        folder_key = parts[1] if len(parts) > 2 else parts[1] if len(parts) > 1 else None
        label = folder_to_label.get(folder_key, -1)
        all_labels.append(label)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels, dtype=int)

    reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
    embeddings_umap = reducer.fit_transform(embeddings)

    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(emb_umap_path, embeddings_umap)
    np.save(labels_path, labels)
    if not osp.exists(label_names_path) and label_names.size > 0:
        np.save(label_names_path, label_names)


def main():
    parser = argparse.ArgumentParser(description='Visualize existing embeddings as MitoSpace')
    parser.add_argument('--checkpoint_dir', default='checkpoint_combined_drugs', type=str,
                        help='Path to checkpoint directory containing embeddings/')
    parser.add_argument('--colors_file', default='extraction_utils/colors_phenotypic.txt',
                        type=str, help='Path to colors file for label palette')
    parser.add_argument('--drugs_to_labels', default='extraction_utils/drugs_to_labels.txt',
                        type=str, help='Path to folder->drug->label mapping file')
    parser.add_argument('--pick_labels', nargs='*', type=int, default=None,
                        help='Subset of labels to visualize')
    args = parser.parse_args()

    embeddings_dir = osp.join(args.checkpoint_dir, 'embeddings_30')
    folder_to_label, label_names = load_folder_label_maps(args.drugs_to_labels)
    colors = load_colors(args.colors_file)

    maybe_build_umap_embeddings(embeddings_dir, folder_to_label, label_names)

    make_mitospace_minimal(embedding_dir=embeddings_dir,
                           pick_labels=args.pick_labels,
                           color_palette=colors)


if __name__ == '__main__':
    main()


