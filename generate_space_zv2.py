import os
import os.path as osp
import argparse
import numpy as np
import umap

from utils.vis import make_mitospace_minimal
from image_viewer import view_4d_image_with_sliders
import matplotlib.patches as mpatches
from vis_data import add_to_viewer
import napari
from validation_zslices import compute_confusion_matrix_and_entropy_from_embeddings_folder

def load_folder_label_maps(drugs_to_labels_path):
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
    return folder_to_label, label_names, folder_to_drug, label_to_drug


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
        if "-" in folder_key:
            
            folder_key = folder_key.split("-")[0]
            print(folder_key)
        label = folder_to_label.get(folder_key, -1)
        print(label)
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

import matplotlib.pyplot as plt

def select_and_plot_embedding(embeddings_dir, embeddings_umap=None, embeddings=None, files=None, show=True, colors_palette=None):
    """
    Enables user to click/select a point in the UMAP embedding space, retrieves the corresponding .npy embedding,
    infers the folder and image name, and attempts to display the central z-slice of the associated image (if possible).

    Supports mouse wheel zooming - use your mouse wheel or touchpad scrolling gesture to zoom the view.
    
    Args:
        embeddings_dir (str): Path to embeddings directory containing per-image embedding npy files.
        embeddings_umap (np.ndarray): [N, 3] array of UMAP embeddings. If None, loads from embeddings_umap.npy.
        embeddings (np.ndarray): [N, F] original high-dim embeddings. Optional, not required for plotting.
        files (list of str): List of per-image npy filenames (sorted). If None, discovers as in maybe_build_umap_embeddings.
        show (bool): If True, show the plot.
        colors_palette: Palette for labeling.
    """
    import numpy as np
    import os
    import os.path as osp

    # Load UMAP and files if not provided
    if embeddings_umap is None:
        embeddings_umap = np.load(osp.join(embeddings_dir, "embeddings_umap.npy"))
    if files is None:
        files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy') and f.startswith('embeddings_20')])
    if len(embeddings_umap.shape) == 2 and embeddings_umap.shape[1] == 3:
        x, y, z = embeddings_umap[:, 0], embeddings_umap[:, 1], embeddings_umap[:, 2]
    else:
        raise ValueError("embeddings_umap should be shape [N, 3]")

    # Load labels and label names if available
    labels_path = os.path.join(embeddings_dir, "labels.npy")
    label_names_path = os.path.join(embeddings_dir, "label_names.npy")
    labels = None
    label_names = None

    try:
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        if os.path.exists(label_names_path):
            # allow_pickle for label_names which may be strings/objects
            label_names = np.load(label_names_path, allow_pickle=True)
    except Exception as e:
        print("Could not load labels or label_names:", e)
    
    # Allow interactive filtering: Pick only some labels/classes

    # pick_names = ['control', 'nocodazole', 'valinomycin', 'nigericin', 'h2o2', 'mitomycinc',  'cisplatin']#'h2o2', 'mitomycinC', 'p110', 'cisplatin']
    # pick_names = []
    pick_names = list(np.unique(label_names))
    # print(pick_names)
    # After loading label_names and labels, filter only those in pick_names
        # Set color values per point if palette and labels are available (with fallback)
    if labels is not None and colors_palette is not None:
        # The palette may be dict or list/array
        if isinstance(colors_palette, dict):
            scatter_colors = np.array([colors_palette.get(int(l), (0.6, 0.6, 0.6)) for l in labels])
        else:
            scatter_colors = np.array([colors_palette[int(l)] if (int(l) < len(colors_palette) and l >= 0) else (0.6, 0.6, 0.6) for l in labels])
    else:
        scatter_colors = None

    mask = None
    if 'label_names' in locals() and label_names is not None and labels is not None:
        mask = np.isin([str(label_names[l]) for l in labels], pick_names)
        x, y, z = x[mask], y[mask], z[mask]
        if scatter_colors is not None and len(scatter_colors) == len(labels):
            scatter_colors = scatter_colors[mask]

        labels = labels[mask]
        files = [f for f, keep in zip(files, mask) if keep]
        # Now the resulting arrays contain only selected embeddings


    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.09, right=0.94, top=0.93, bottom=0.08)
    if scatter_colors is not None and len(scatter_colors) == len(x):
        pts = ax.scatter(x, y, z, s=20, alpha=0.8, c=scatter_colors)
    else:
        pts = ax.scatter(x, y, z, s=20, alpha=0.8)
    plt.title("Click on a point to show its image\n(Use mouse wheel to zoom!)")

    # Add a legend for the labels, if available (one dot per class with color)
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

    # store [ind] in the plot object; workaround for old matplotlib
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
        # Infer folder/image from fname: e.g. embeddings_<folder>_<imgname>.npy
        parts = fname.split("_")
        if len(parts) >= 3:
            folder = parts[1]
            img_basename = "_".join(parts[2:]).replace(".npy", "")
        else:
            folder = "unknown"
            img_basename = fname.replace(".npy", "")
        print(f"Folder: {folder}; Image base name: {img_basename}")

        # Try to locate the image file (example: search in embeddings_dir/../<folder>/<img_basename>.npy or .tif)
        possible_dirs = [
            # osp.join(embeddings_dir, '..', folder),
            # osp.join(embeddings_dir, folder),
            # osp.abspath(osp.join(embeddings_dir, '..', folder))
            # "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/",
            "/run/user/1004/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/2024_summer_new/"
            # "/run/user/1004/gvfs/afp-volume:host=JSLab-Server1.local,volume=SSD_Processing/Others/MitoSpace4D/2024_summer_new/",
            # "/run/user/1004/gvfs/afp-volume:host=JSLab-Server1.local,user=JSLab_FileShare,volume=SSD_Processing/Others/MitoSpace4D/2024_summer_new/"

        ]
        found_path = None
        selected_paths = []
        for ext in ['.npy']:
            for d in possible_dirs:
                fullpath = osp.join(d, folder, img_basename + ext)
                
                print(f"Found path: {fullpath}")
                if osp.exists(fullpath):
                    found_path = fullpath
                    selected_paths.append(fullpath)
                    break
            if found_path:
                break
        # Now, candidate_paths contains all checked paths in order.

        # view_4d_image_with_sliders(found_path, position = idx)
        viewer = napari.Viewer(ndisplay=3)
        add_to_viewer(viewer, found_path, translate=(0, 0), channel=0)
        add_to_viewer(viewer, found_path, translate=(0, 256 + 10), channel=1)
        napari.run()

    # --- Zooming interaction support for 3D plot ---
    # https://stackoverflow.com/questions/24177974/matplotlib-3d-plot-zooming-with-scroll-wheel
    def zoom_factory(ax, base_scale = 1.2):
        def zoom_fun(event):
            # Only act on scroll event in axes and if it's our axes
            if event.inaxes != ax:
                return

            # For 3D axes, limit to zooming along all 3 axes equally
            scale_factor = 1
            if event.button == 'up':
                # zoom in
                scale_factor = 1/base_scale
            elif event.button == 'down':
                # zoom out
                scale_factor = base_scale
            else:
                # unknown event, ignore
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

    # Connect click events to picking
    fig.canvas.mpl_connect('pick_event', onpick)
    pts.set_picker(True)
    
    # Connect scroll for zooming
    fig.canvas.mpl_connect('scroll_event', zoom_factory(ax, base_scale=1.2))

    if show:
        plt.show()

# Example usage:
# 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize existing embeddings as MitoSpace')
    parser.add_argument('--checkpoint_dir', default='checkpoint_contrastive_nocodazole_colchicine', type=str,
                        help='Path to checkpoint directory containing embeddings/')
    parser.add_argument('--colors_file', default='extraction_utils/colors_eric.txt',
                        type=str, help='Path to colors file for label palette')
    parser.add_argument('--drugs_to_labels', default='extraction_utils/drugs_to_labels.txt',
                        type=str, help='Path to folder->drug->label mapping file')
    parser.add_argument('--pick_labels', nargs='*', type=int, default=None,
                        help='Subset of labels to visualize')
    parser.add_argument("--visualize", type=bool, default=True, help="Whether to visualize the embeddings")
    parser.add_argument('--embedding_folder', help='Path to embedding folder', default="embeddings")
    args = parser.parse_args()

    embeddings_dir = osp.join(args.checkpoint_dir, args.embedding_folder)
    folder_to_label, label_names, folder_to_drug, label_to_drug_dict = load_folder_label_maps(args.drugs_to_labels)
    colors = load_colors(args.colors_file)
    maybe_build_umap_embeddings(embeddings_dir, folder_to_label, label_names)

    if args.visualize:
        select_and_plot_embedding(embeddings_dir=embeddings_dir, colors_palette=colors)
 
    
    # make_mitospace_minimal(embedding_dir=embeddings_dir,
    #                        pick_labels=args.pick_labels,
    #                        color_palette=colors)
    
    print("Computing confusion matrix and entropy from embeddings folder")
    metrics = compute_confusion_matrix_and_entropy_from_embeddings_folder(embeddings_dir, folder_to_drug, folder_to_label, label_drug_dict=label_to_drug_dict)
    print(metrics)
    import json
    with open(osp.join(args.checkpoint_dir, "entropy_metrics.json"), "w") as f:
        json.dump(metrics, f)


