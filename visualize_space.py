import os
import os.path as osp
import argparse
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vis_data import add_to_viewer
import napari
from validation_zslices import compute_confusion_matrix_and_entropy_from_embeddings_folder
from tqdm import tqdm
from utils.utils import topKfrequent
import json

parser = argparse.ArgumentParser(description='MitoSpace Visualization')
parser.add_argument('--checkpoint_dir', default='checkpoint_combined_drugs', type=str,
                    help='Path to checkpoint directory containing embeddings/')
parser.add_argument('--colors_file', default='extraction_utils/colors_eric.txt',
                    type=str, help='Path to colors file for label palette')
parser.add_argument('--drugs_to_labels', default='extraction_utils/drugs_to_labels.txt',
                    type=str, help='Path to folder->drug->label mapping file')
parser.add_argument('--embedding_folder', default='embeddings', type=str,
                    help='Path to embedding folder relative to checkpoint_dir')
parser.add_argument('--pick_labels', nargs='*', type=str, default=None,
                    help='Subset of label names to visualize (e.g., control nocodazole)')
parser.add_argument('--data_base_paths', nargs='*', type=str, 
                    default=[
                        "/run/user/1004/gvfs/afp-volume:host=JSLab-Server1.local,volume=SSD_Processing/Others/MitoSpace4D/2024_summer_new/",
                        "/run/user/1004/gvfs/afp-volume:host=JSLab-Server1.local,user=JSLab_FileShare,volume=SSD_Processing/Others/MitoSpace4D/2024_summer_new/"
                    ],
                    help='Base paths to search for image files when clicking on embeddings')
parser.add_argument('--umap_n_neighbors', default=25, type=int,
                    help='Number of neighbors for UMAP')
parser.add_argument('--umap_min_dist', default=0.01, type=float,
                    help='Minimum distance for UMAP')
parser.add_argument('--umap_metric', default='cosine', type=str,
                    help='Distance metric for UMAP')
parser.add_argument('--umap_n_components', default=3, type=int,
                    help='Number of components for UMAP (2 or 3)')
parser.add_argument('--interactive', action='store_true',
                    help='Enable interactive visualization (click to view images)')
parser.add_argument('--compute_entropy', action='store_true',
                    help='Compute and save entropy metrics')
parser.add_argument('--evaluate_knn', action='store_true',
                    help='Evaluate k-NN accuracy using cosine distance')
parser.add_argument('--k_neighbors', nargs='*', type=int, default=[100],
                    help='Number of neighbors for k-NN evaluation')
parser.add_argument('--top_ns', nargs='*', type=int, default=[1, 3, 5],
                    help='Top-N accuracies to compute')
parser.add_argument('--train_split', default=0.9, type=float,
                    help='Fraction of data to use for training (rest for evaluation)')
parser.add_argument('--dist_metric', default='cosine', type=str, choices=['cosine', 'l2'],
                    help='Distance metric for evaluation')
parser.add_argument('--temperature', default=1.0, type=float,
                    help='Temperature for weighted cosine distance')


def load_folder_label_maps(drugs_to_labels_path):
    """
    Load folder to label and drug mappings from a text file.
    
    Args:
        drugs_to_labels_path (str): Path to file with format: folder drug label
        
    Returns:
        tuple: (folder_to_label dict, label_names array, folder_to_drug dict)
    """
    folder_to_label = {}
    label_to_drug = {}
    folder_to_drug = {}
    
    with open(drugs_to_labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            folder, drug, label = parts
            label_int = int(label)
            folder_to_label[folder] = label_int
            label_to_drug[label_int] = drug
            folder_to_drug[folder] = drug
    
    # Build label_names array (sorted by label)
    if label_to_drug:
        max_label = max(label_to_drug.keys())
        label_names = np.array([label_to_drug.get(i, f"label_{i}") for i in range(max_label + 1)], dtype=object)
    else:
        label_names = np.array([], dtype=object)
    
    return folder_to_label, label_names, folder_to_drug


def load_colors(colors_file_path):
    """
    Load color palette from a text file.
    
    Args:
        colors_file_path (str): Path to colors file
        
    Returns:
        dict or None: Dictionary mapping label index to RGB color tuple, or None if file doesn't exist
    """
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


def build_umap_embeddings(embeddings_dir, folder_to_label, label_names, 
                          n_neighbors=25, min_dist=0.01, metric='cosine', 
                          n_components=3, force_rebuild=False):
    """
    Build UMAP embeddings from per-sample embedding files.
    
    Args:
        embeddings_dir (str): Directory containing per-sample embedding .npy files
        folder_to_label (dict): Mapping from folder name to label
        label_names (np.ndarray): Array of label names
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        metric (str): Distance metric for UMAP
        n_components (int): Number of UMAP components (2 or 3)
        force_rebuild (bool): If True, rebuild even if files exist
        
    Returns:
        tuple: (embeddings_umap, labels, files) where:
            - embeddings_umap: np.ndarray of shape [N, n_components]
            - labels: np.ndarray of shape [N]
            - files: list of filenames
    """
    emb_umap_path = osp.join(embeddings_dir, 'embeddings_umap.npy')
    labels_path = osp.join(embeddings_dir, 'labels.npy')
    label_names_path = osp.join(embeddings_dir, 'label_names.npy')
    
    # Check if already exists
    if not force_rebuild and osp.exists(emb_umap_path) and osp.exists(labels_path) and osp.exists(label_names_path):
        print(f"✅ Loading existing UMAP embeddings from {embeddings_dir}")
        embeddings_umap = np.load(emb_umap_path)
        labels = np.load(labels_path)
        files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy') and f.startswith('embeddings_20')])
        return embeddings_umap, labels, files
    
    print(f"🔹 Building UMAP embeddings from {embeddings_dir}")
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
        if "-" in folder_key:
            folder_key = folder_key.split("-")[0]
        label = folder_to_label.get(folder_key, -1)
        all_labels.append(label)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels, dtype=int)
    
    print(f"🔹 Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    reducer = umap.UMAP(verbose=True, n_components=n_components, n_neighbors=n_neighbors, 
                       min_dist=min_dist, metric=metric)
    embeddings_umap = reducer.fit_transform(embeddings)
    
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(emb_umap_path, embeddings_umap)
    np.save(labels_path, labels)
    if not osp.exists(label_names_path) and label_names.size > 0:
        np.save(label_names_path, label_names)
    
    print(f"✅ Saved UMAP embeddings to {emb_umap_path}")
    return embeddings_umap, labels, files


def plot_embedding_space(embeddings_umap, labels=None, label_names=None, 
                        colors_palette=None, pick_names=None, show=True):
    """
    Plot embeddings in 2D or 3D space.
    
    Args:
        embeddings_umap (np.ndarray): [N, 2] or [N, 3] array of UMAP embeddings
        labels (np.ndarray): [N] array of labels
        label_names (np.ndarray): Array of label names
        colors_palette (dict or list): Color palette for labels
        pick_names (list): List of label names to filter
        show (bool): If True, show the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if len(embeddings_umap.shape) != 2:
        raise ValueError("embeddings_umap should be 2D array")
    
    n_components = embeddings_umap.shape[1]
    if n_components not in [2, 3]:
        raise ValueError("embeddings_umap should have 2 or 3 components")
    
    # Filter by pick_names if provided
    mask = None
    if pick_names is not None and labels is not None and label_names is not None:
        mask = np.isin([str(label_names[l]) for l in labels], pick_names)
        embeddings_umap = embeddings_umap[mask]
        labels = labels[mask]
    
    # Set colors
    scatter_colors = None
    if labels is not None and colors_palette is not None:
        if isinstance(colors_palette, dict):
            scatter_colors = np.array([colors_palette.get(int(l), (0.6, 0.6, 0.6)) for l in labels])
        else:
            scatter_colors = np.array([colors_palette[int(l)] if (int(l) < len(colors_palette) and l >= 0) else (0.6, 0.6, 0.6) for l in labels])
    
    # Create plot
    fig = plt.figure(figsize=(18, 14))
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = embeddings_umap[:, 0], embeddings_umap[:, 1], embeddings_umap[:, 2]
        if scatter_colors is not None and len(scatter_colors) == len(x):
            pts = ax.scatter(x, y, z, s=20, alpha=0.8, c=scatter_colors)
        else:
            pts = ax.scatter(x, y, z, s=20, alpha=0.8)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
    else:
        ax = fig.add_subplot(111)
        x, y = embeddings_umap[:, 0], embeddings_umap[:, 1]
        if scatter_colors is not None and len(scatter_colors) == len(x):
            pts = ax.scatter(x, y, s=20, alpha=0.8, c=scatter_colors)
        else:
            pts = ax.scatter(x, y, s=20, alpha=0.8)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    
    plt.title("MitoSpace Embedding Visualization")
    
    # Add legend
    if label_names is not None and colors_palette is not None:
        legend_patches = []
        unique_labels = np.unique(labels) if labels is not None else []
        for label_idx in unique_labels:
            if isinstance(colors_palette, dict):
                color = colors_palette.get(int(label_idx), (0.6, 0.6, 0.6))
            else:
                color = colors_palette[int(label_idx)] if int(label_idx) < len(colors_palette) else (0.6, 0.6, 0.6)
            label_name = str(label_names[label_idx]) if label_idx < len(label_names) else f"label_{label_idx}"
            legend_patches.append(mpatches.Patch(color=color, label=label_name))
        if legend_patches:
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def select_and_plot_embedding(embeddings_dir, embeddings_umap=None, embeddings=None, 
                              files=None, labels=None, label_names=None, 
                              show=True, colors_palette=None, pick_names=None,
                              data_base_paths=None):
    """
    Interactive visualization: click on a point to view its image.
    
    Args:
        embeddings_dir (str): Path to embeddings directory
        embeddings_umap (np.ndarray): [N, 3] UMAP embeddings
        embeddings (np.ndarray): [N, F] original embeddings (optional)
        files (list): List of filenames
        labels (np.ndarray): [N] array of labels
        label_names (np.ndarray): Array of label names
        show (bool): If True, show the plot
        colors_palette: Color palette for labels
        pick_names (list): List of label names to filter
        data_base_paths (list): Base paths to search for image files
    """
    if data_base_paths is None:
        data_base_paths = [
            "/run/user/1004/gvfs/afp-volume:host=JSLab-Server1.local,volume=SSD_Processing/Others/MitoSpace4D/2024_summer_new/",
        ]
    
    # Load UMAP and files if not provided
    if embeddings_umap is None:
        embeddings_umap = np.load(osp.join(embeddings_dir, "embeddings_umap.npy"))
    if files is None:
        files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy') and f.startswith('embeddings_20')])
    if labels is None:
        labels_path = osp.join(embeddings_dir, "labels.npy")
        if osp.exists(labels_path):
            labels = np.load(labels_path)
    if label_names is None:
        label_names_path = osp.join(embeddings_dir, "label_names.npy")
        if osp.exists(label_names_path):
            label_names = np.load(label_names_path, allow_pickle=True)
    
    if len(embeddings_umap.shape) == 2 and embeddings_umap.shape[1] == 3:
        x, y, z = embeddings_umap[:, 0], embeddings_umap[:, 1], embeddings_umap[:, 2]
    else:
        raise ValueError("embeddings_umap should be shape [N, 3]")
    
    # Filter by pick_names
    if pick_names is None and label_names is not None:
        pick_names = list(np.unique(label_names))
    
    # Set colors
    scatter_colors = None
    if labels is not None and colors_palette is not None:
        if isinstance(colors_palette, dict):
            scatter_colors = np.array([colors_palette.get(int(l), (0.6, 0.6, 0.6)) for l in labels])
        else:
            scatter_colors = np.array([colors_palette[int(l)] if (int(l) < len(colors_palette) and l >= 0) else (0.6, 0.6, 0.6) for l in labels])
    
    # Apply mask
    mask = None
    if label_names is not None and labels is not None and pick_names is not None:
        mask = np.isin([str(label_names[l]) for l in labels], pick_names)
        x, y, z = x[mask], y[mask], z[mask]
        if scatter_colors is not None and len(scatter_colors) == len(labels):
            scatter_colors = scatter_colors[mask]
        labels = labels[mask]
        files = [f for f, keep in zip(files, mask) if keep]
    
    # Create plot
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.09, right=0.94, top=0.93, bottom=0.08)
    if scatter_colors is not None and len(scatter_colors) == len(x):
        pts = ax.scatter(x, y, z, s=20, alpha=0.8, c=scatter_colors)
    else:
        pts = ax.scatter(x, y, z, s=20, alpha=0.8)
    plt.title("Click on a point to show its image\n(Use mouse wheel to zoom!)")
    
    # Add legend
    if label_names is not None and colors_palette is not None:
        legend_patches = []
        for i, label_name in enumerate(label_names):
            if isinstance(colors_palette, dict):
                color = colors_palette.get(i, (0.6, 0.6, 0.6))
            else:
                color = colors_palette[i] if i < len(colors_palette) else (0.6, 0.6, 0.6)
            legend_patches.append(mpatches.Patch(color=color, label=str(label_name)))
        if legend_patches:
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    selected_idx = [None]
    
    def onpick(event):
        if hasattr(event, 'ind'):
            ind = event.ind[0]
            selected_idx[0] = ind
            highlight_and_show(ind)
    
    def highlight_and_show(idx):
        ax.scatter([x[idx]], [y[idx]], [z[idx]], s=100, c="red", marker="*", alpha=1.0)
        fig.canvas.draw_idle()
        fname = files[idx]
        print(f"Selected idx: {idx}, file: {fname}")
        
        # Infer folder/image from fname
        parts = fname.split("_")
        if len(parts) >= 3:
            folder = parts[1]
            img_basename = "_".join(parts[2:]).replace(".npy", "")
        else:
            folder = "unknown"
            img_basename = fname.replace(".npy", "")
        print(f"Folder: {folder}; Image base name: {img_basename}")
        
        # Search for image file
        found_path = None
        for ext in ['.npy']:
            for d in data_base_paths:
                fullpath = osp.join(d, folder, img_basename + ext)
                if osp.exists(fullpath):
                    found_path = fullpath
                    print(f"Found path: {fullpath}")
                    break
            if found_path:
                break
        
        if found_path:
            viewer = napari.Viewer(ndisplay=3)
            add_to_viewer(viewer, found_path, translate=(0, 0), channel=0)
            add_to_viewer(viewer, found_path, translate=(0, 256 + 10), channel=1)
            napari.run()
        else:
            print(f"⚠️ Could not find image file for {fname}")
    
    # Zoom support
    def zoom_factory(ax, base_scale=1.2):
        def zoom_fun(event):
            if event.inaxes != ax:
                return
            scale_factor = 1
            if event.button == 'up':
                scale_factor = 1/base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return
            
            xlim = ax.get_xlim3d()
            ylim = ax.get_ylim3d()
            zlim = ax.get_zlim3d()
            
            xmean = np.mean(xlim)
            ymean = np.mean(ylim)
            zmean = np.mean(zlim)
            
            x_range = (xlim[1] - xlim[0]) * scale_factor
            y_range = (ylim[1] - ylim[0]) * scale_factor
            z_range = (zlim[1] - zlim[0]) * scale_factor
            
            ax.set_xlim3d([xmean - x_range/2, xmean + x_range/2])
            ax.set_ylim3d([ymean - y_range/2, ymean + y_range/2])
            ax.set_zlim3d([zmean - z_range/2, zmean + z_range/2])
            fig.canvas.draw_idle()
        return zoom_fun
    
    # Connect events
    fig.canvas.mpl_connect('pick_event', onpick)
    pts.set_picker(True)
    fig.canvas.mpl_connect('scroll_event', zoom_factory(ax, base_scale=1.2))
    
    if show:
        plt.show()


def cosine_distance(eval_embeddings, train_embeddings, weighted=False, temperature=1.0):
    """
    Compute cosine distance matrix between evaluation and training embeddings.
    
    Args:
        eval_embeddings (np.ndarray): [N_eval, D] evaluation embeddings
        train_embeddings (np.ndarray): [N_train, D] training embeddings
        weighted (bool): If True, apply temperature weighting
        temperature (float): Temperature for weighted distance
        
    Returns:
        tuple: (dist_matrix_sorted, dist_matrix_idxs) where:
            - dist_matrix_sorted: [N_eval, N_train] sorted distance matrix
            - dist_matrix_idxs: [N_eval, N_train] indices sorted by distance
    """
    dist_matrix = eval_embeddings @ train_embeddings.T
    if weighted:
        dist_matrix = dist_matrix / temperature
        dist_matrix = np.exp(dist_matrix)
    
    dist_matrix_idxs = (-1 * dist_matrix).argsort(1)  # Sort in descending order (higher = more similar)
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    
    return dist_matrix_sorted, dist_matrix_idxs


def l2_distance(eval_embeddings, train_embeddings):
    """
    Compute L2 distance matrix between evaluation and training embeddings.
    
    Args:
        eval_embeddings (np.ndarray): [N_eval, D] evaluation embeddings
        train_embeddings (np.ndarray): [N_train, D] training embeddings
        
    Returns:
        np.ndarray: [N_eval, N_train] distance matrix sorted by distance
    """
    dist_matrix = np.linalg.norm(eval_embeddings[:, None] - train_embeddings[None, :], axis=-1)
    dist_matrix_idxs = dist_matrix.argsort(1)  # Sort in ascending order (lower = closer)
    dist_matrix_sorted = np.take_along_axis(dist_matrix, dist_matrix_idxs, axis=1)
    
    return dist_matrix_sorted, dist_matrix_idxs


def nearest_neighbor_evaluation(eval_labels, train_labels, top_ns, dist_matrix, dist_matrix_idxs,
                                num_neighbors=[100], verbose=True):
    """
    Evaluate k-NN accuracy using distance matrix.
    
    Args:
        eval_labels (np.ndarray): [N_eval] ground truth labels for evaluation set
        train_labels (np.ndarray): [N_train] labels for training set
        top_ns (list): List of top-N values to evaluate (e.g., [1, 3, 5])
        dist_matrix (np.ndarray): [N_eval, N_train] sorted distance matrix
        dist_matrix_idxs (np.ndarray): [N_eval, N_train] indices sorted by distance
        num_neighbors (list): List of k values for k-NN
        verbose (bool): If True, print progress and results
        
    Returns:
        tuple: (preds, correct_preds_idxs, incorrect_preds_idxs) where:
            - preds: dict mapping top_n to list of predictions
            - correct_preds_idxs: dict mapping top_n to list of correct prediction indices
            - incorrect_preds_idxs: dict mapping top_n to list of incorrect prediction indices
    """
    preds = None
    
    # Normalize the distance matrix (for cosine similarity, convert to [0, 1])
    dist_matrix = (dist_matrix + 1) / 2
    
    results = {}
    
    for k in num_neighbors:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation for {k} Neighbors")
            print(f"{'='*60}")
        
        correct_preds = {top_n: 0 for top_n in top_ns}
        correct_preds_per_class = {top_n: {lbl: 0 for lbl in np.unique(train_labels)} for top_n in top_ns}
        preds = {top_n: [] for top_n in top_ns}
        
        # Save the indices of the correct and incorrect predictions
        correct_preds_idxs = {top_n: [] for top_n in top_ns}
        incorrect_preds_idxs = {top_n: [] for top_n in top_ns}
        
        pbar = tqdm(total=len(dist_matrix), desc=f"Evaluating k={k}") if verbose else None
        for i in range(len(dist_matrix)):
            eval_lbl = eval_labels[i]  # ground truth label
            k_nearest_nbs = train_labels[dist_matrix_idxs[i][:k]]
            k_nearest_dist = dist_matrix[i][:k]
            
            for top_n in top_ns:
                top_most_freq_lbls = topKfrequent(k_nearest_nbs, k_nearest_dist, top_n, weighted=True)
                if eval_lbl in top_most_freq_lbls:
                    correct_preds[top_n] += 1
                    correct_preds_per_class[top_n][eval_lbl] += 1
                    preds[top_n].append(eval_lbl)
                    correct_preds_idxs[top_n].append(i)
                else:
                    preds[top_n].append(top_most_freq_lbls[0])
                    incorrect_preds_idxs[top_n].append(i)
            
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        # Print results
        for top_n in top_ns:
            correct = correct_preds[top_n]
            if verbose:
                print(f"\n{'-'*60}")
                print(f"Top-{top_n} Accuracy")
                print(f"{'-'*60}")
                print(f"Correct: {correct}; Total: {len(dist_matrix)}")
                acc = correct * 100. / len(eval_labels)
                print(f"Accuracy: {acc:.2f}%")
                
                # Print per class accuracy
                print(f"\nPer-class accuracy:")
                for lbl in np.unique(train_labels):
                    total = np.sum(eval_labels == lbl)
                    correct_class = correct_preds_per_class[top_n][lbl]
                    if total > 0:
                        acc_class = correct_class * 100. / total
                        print(f"  Class {lbl}: {correct_class}/{total} ({acc_class:.2f}%)")
        
        results[k] = {
            'preds': preds,
            'correct_preds_idxs': correct_preds_idxs,
            'incorrect_preds_idxs': incorrect_preds_idxs,
            'accuracies': {top_n: correct_preds[top_n] * 100. / len(eval_labels) for top_n in top_ns}
        }
    
    return results


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Load mappings
    embeddings_dir = osp.join(args.checkpoint_dir, args.embedding_folder)
    folder_to_label, label_names, folder_to_drug = load_folder_label_maps(args.drugs_to_labels)
    colors = load_colors(args.colors_file)
    
    print(f"🔹 Loading embeddings from {embeddings_dir}")
    print(f"🔹 Found {len(folder_to_label)} folder mappings")
    print(f"🔹 Found {len(label_names)} label names")
    
    # Build UMAP embeddings
    embeddings_umap, labels, files = build_umap_embeddings(
        embeddings_dir, 
        folder_to_label, 
        label_names,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_components=args.umap_n_components
    )
    
    print(f"✅ Built UMAP embeddings: shape {embeddings_umap.shape}")
    
    # Interactive visualization
    if args.interactive:
        print("🔹 Starting interactive visualization...")
        select_and_plot_embedding(
            embeddings_dir=embeddings_dir,
            embeddings_umap=embeddings_umap,
            labels=labels,
            label_names=label_names,
            files=files,
            colors_palette=colors,
            pick_names=args.pick_labels,
            data_base_paths=args.data_base_paths
        )
    else:
        # Static plot
        print("🔹 Creating static plot...")
        plot_embedding_space(
            embeddings_umap=embeddings_umap,
            labels=labels,
            label_names=label_names,
            colors_palette=colors,
            pick_names=args.pick_labels
        )
    
    # Evaluate k-NN accuracy if requested
    if args.evaluate_knn:
        print("\n" + "="*60)
        print("🔹 Evaluating k-NN Accuracy")
        print("="*60)
        
        # Load original embeddings (not UMAP)
        print("🔹 Loading original embeddings...")
        all_embeddings = []
        for fname in files:
            fpath = osp.join(embeddings_dir, fname)
            emb = np.load(fpath)
            emb = emb.reshape(1, -1)
            all_embeddings.append(emb)
        
        embeddings_original = np.concatenate(all_embeddings, axis=0)
        print(f"✅ Loaded embeddings: shape {embeddings_original.shape}")
        
        # Normalize embeddings
        embeddings_original = embeddings_original / np.linalg.norm(embeddings_original, axis=1, keepdims=True)
        
        # Split into train and eval
        n_total = len(embeddings_original)
        n_train = int(n_total * args.train_split)
        n_eval = n_total - n_train
        
        train_embeddings = embeddings_original[:n_train]
        eval_embeddings = embeddings_original[n_train:]
        train_labels = labels[:n_train]
        eval_labels = labels[n_train:]
        
        print(f"🔹 Split: {n_train} train, {n_eval} eval")
        
        # Compute distance matrix
        if args.dist_metric == 'cosine':
            print(f"🔹 Computing cosine distance matrix...")
            dist_matrix, dist_matrix_idxs = cosine_distance(
                eval_embeddings, 
                train_embeddings, 
                weighted=False,
                temperature=args.temperature
            )
        elif args.dist_metric == 'l2':
            print(f"🔹 Computing L2 distance matrix...")
            dist_matrix, dist_matrix_idxs = l2_distance(eval_embeddings, train_embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {args.dist_metric}")
        
        print(f"✅ Distance matrix shape: {dist_matrix.shape}")
        
        # Evaluate
        results = nearest_neighbor_evaluation(
            eval_labels=eval_labels,
            train_labels=train_labels,
            top_ns=args.top_ns,
            dist_matrix=dist_matrix,
            dist_matrix_idxs=dist_matrix_idxs,
            num_neighbors=args.k_neighbors,
            verbose=True
        )
        
        # Save results

        results_path = osp.join(args.checkpoint_dir, "knn_evaluation_results.json")
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for k, result in results.items():
            results_serializable[k] = {
                'accuracies': result['accuracies'],
                'num_correct_preds': {top_n: len(result['correct_preds_idxs'][top_n]) for top_n in args.top_ns},
                'num_incorrect_preds': {top_n: len(result['incorrect_preds_idxs'][top_n]) for top_n in args.top_ns}
            }
        
        with open(results_path, "w") as f:
            json.dump({
                'config': {
                    'k_neighbors': args.k_neighbors,
                    'top_ns': args.top_ns,
                    'dist_metric': args.dist_metric,
                    'train_split': args.train_split,
                    'n_train': n_train,
                    'n_eval': n_eval
                },
                'results': results_serializable
            }, f, indent=2)
        print(f"\n✅ Saved evaluation results to {results_path}")
    
    # Compute entropy metrics if requested
    if args.compute_entropy:
        print("\n🔹 Computing entropy metrics...")
        metrics = compute_confusion_matrix_and_entropy_from_embeddings_folder(
            embeddings_dir, 
            folder_to_drug, 
            folder_to_label
        )
        print(metrics)
        import json
        metrics_path = osp.join(args.checkpoint_dir, "entropy_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Saved entropy metrics to {metrics_path}")

