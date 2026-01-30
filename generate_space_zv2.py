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

import plotly.graph_objects as go

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

    # pick_names = ['control', 'mdivi1']#'nocodazole', 'valinomycin', 'nigericin', 'h2o2', 'mitomycinc',  'cisplatin']#'h2o2', 'mitomycinC', 'p110', 'cisplatin']
    pick_names = []
    if not pick_names and label_names is not None:
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

    def highlight_and_show(idx):
        """Handle point selection and display image"""
        fname = files[idx]
        print(f"Selected idx: {idx}, file: {fname}")
        # Infer folder/image from fname: e.g. embeddings_<folder>_<imgname>.npy
        parts = fname.split("_")
        if len(parts) >= 3:
            folder = parts[1]
            if "-1" not in folder:
                folder = folder + "-1"
            img_basename = "_".join(parts[2:]).replace(".npy", "")
            if "-0" not in img_basename:
                img_basename = img_basename + "-0"
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
            # "/run/user/1004/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/2024v2_data/processed_data/20240830-1/",
            '/run/user/1004/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/2024v2_data/processed_data/'
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

        view_4d_image_with_sliders(found_path, position = idx)
        # viewer = napari.Viewer(ndisplay=3)
        # add_to_viewer(viewer, found_path, translate=(0, 0), channel=0)
        # add_to_viewer(viewer, found_path, translate=(0, 256 + 10), channel=1)
        # napari.run()

    # Prepare colors for Plotly (convert RGB tuples to hex strings)
    def rgb_to_hex(rgb):
        """Convert RGB tuple to hex string"""
        if isinstance(rgb, (list, tuple, np.ndarray)):
            if len(rgb) >= 3:
                r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                return f'#{r:02x}{g:02x}{b:02x}'
        return '#999999'  # default gray

    # Convert scatter_colors to hex if available
    if scatter_colors is not None and len(scatter_colors) == len(x):
        colors_hex = [rgb_to_hex(c) for c in scatter_colors]
    else:
        colors_hex = None

    # Create hover text with file names
    hover_texts = [f"Index: {i}<br>File: {f}" for i, f in enumerate(files)]

    # Create the 3D scatter plot
    fig = go.Figure()

    # If we have colors, we might want to group by label for legend
    if label_names is not None and colors_palette is not None and labels is not None:
        # Group points by label for better legend
        unique_labels = np.unique(labels)
        for label_val in unique_labels:
            mask_label = labels == label_val
            x_subset = x[mask_label]
            y_subset = y[mask_label]
            z_subset = z[mask_label]
            hover_subset = [hover_texts[i] for i in range(len(hover_texts)) if mask_label[i]]
            
            # Get color for this label
            if isinstance(colors_palette, dict):
                color = colors_palette.get(int(label_val), (0.6, 0.6, 0.6))
            else:
                color = colors_palette[int(label_val)] if (int(label_val) < len(colors_palette) and label_val >= 0) else (0.6, 0.6, 0.6)
            color_hex = rgb_to_hex(color)
            
            # Get label name
            label_name = str(label_names[label_val]) if label_val < len(label_names) else f"Label {label_val}"
            
            fig.add_trace(go.Scatter3d(
                x=x_subset,
                y=y_subset,
                z=z_subset,
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_hex,
                    opacity=0.8,
                ),
                name=label_name,
                text=hover_subset,
                hovertemplate='%{text}<extra></extra>',
                customdata=np.where(mask_label)[0],  # Store original indices
            ))
    else:
        # Single trace with all points
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=colors_hex if colors_hex else None,
                opacity=0.8,
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            customdata=list(range(len(x))),  # Store original indices
        ))

    # Update layout
    fig.update_layout(
        title="Click on a point to show its image<br>(Use mouse wheel to zoom!)",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        width=1200,
        height=900,
        hovermode='closest',
        legend=dict(
            title="Labels",
            x=-0.07,      # Move legend to the left out of plot area
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=10),
            itemsizing='constant',
            itemwidth=30
        ),
        margin=dict(l=150, r=0, t=50, b=0),  # Add left margin for legend
    )

    # Use Dash for reliable click event handling in standalone scripts
    if show:
        try:
            from dash import Dash, dcc, html, Input, Output, State
            
            app = Dash(__name__)
            
            # Store image data globally for the callback
            image_data_store = {'data': None, 'path': None, 'idx': None}
            
            def load_image_data_from_file(fname):
                """Load image data from the embedding filename"""
                print(f"Loading image for file: {fname}")
                
                # Extract folder and image basename from embedding filename
                # Format: embeddings_<folder>_<imgname>.npy
                parts = fname.replace(".npy", "").split("_")
                if len(parts) >= 3:
                    folder = parts[1]
                    if "-1" not in folder:
                        folder = folder + "-1"
                    img_basename = "_".join(parts[2:])
                    if "-0" not in img_basename:
                        img_basename = img_basename + "-0"
                else:
                    folder = "unknown"
                    img_basename = fname.replace(".npy", "")
                
                # Try to locate the image file
                possible_dirs = [
                    '/run/user/1004/gvfs/smb-share:server=jslab-server1.local,share=ssd_processing/Others/MitoSpace4D/2024v2_data/processed_data/'
                ]
                
                print(f"Searching for image - Folder: {folder}, Image basename: {img_basename}")
                found_path = None
                for ext in ['.npy']:
                    for d in possible_dirs:
                        fullpath = osp.join(d, folder, img_basename + ext)
                        print(f"Checking path: {fullpath}")
                        print(f"  Path exists: {osp.exists(fullpath)}")
                        if osp.exists(fullpath):
                            found_path = fullpath
                            print(f"✓ Found path: {found_path}")
                            break
                    if found_path:
                        break
                
                if not found_path:
                    print(f"✗ Image not found for folder: {folder}, basename: {img_basename}")
                
                if found_path:
                    try:
                        image_data = np.load(found_path)
                        # Handle shape: (time_points, z_slices, y_dim, x_dim, channels) or (channels, time_points, z_slices, y_dim, x_dim)
                        if image_data.ndim == 5:
                            if image_data.shape[0] == 2:
                                # Shape is (channels, time_points, z_slices, y_dim, x_dim)
                                image_data = image_data.transpose(1, 0, 2, 3, 4)
                            # Now shape is (time_points, channels, z_slices, y_dim, x_dim)
                            image_data_store['data'] = image_data
                            image_data_store['path'] = found_path
                            image_data_store['fname'] = fname
                            return True
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        return False
                return False
            
            app.layout = html.Div([
                html.Div([
                    dcc.Graph(
                        id='umap-plot',
                        figure=fig,
                        config={'displayModeBar': True},
                        style={'width': '60%', 'display': 'inline-block'}
                    ),
                    html.Div([
                        html.Div(id='image-info', style={'margin': '10px'}),
                        dcc.Graph(id='channel1-plot', style={'width': '100%'}),
                        dcc.Graph(id='channel2-plot', style={'width': '100%'}),
                        html.Div([
                            html.Label('Time:', style={'margin-right': '10px'}),
                            dcc.Slider(
                                id='time-slider',
                                min=0,
                                max=1,
                                value=0,
                                step=1,
                                marks={},
                                disabled=True
                            ),
                        ], style={'width': '100%', 'margin': '20px 0'}),
                        html.Div([
                            html.Label('Z-Slice:', style={'margin-right': '10px'}),
                            dcc.Slider(
                                id='z-slider',
                                min=0,
                                max=1,
                                value=0,
                                step=1,
                                marks={},
                                disabled=True
                            ),
                        ], style={'width': '100%', 'margin': '20px 0'}),
                    ], style={'width': '38%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
                ], style={'display': 'flex'}),
                html.Div(id='click-output', style={'display': 'none'})
            ])
            
            @app.callback(
                [Output('click-output', 'children'),
                 Output('image-info', 'children'),
                 Output('time-slider', 'max'),
                 Output('time-slider', 'value'),
                 Output('time-slider', 'disabled'),
                 Output('time-slider', 'marks'),
                 Output('z-slider', 'max'),
                 Output('z-slider', 'value'),
                 Output('z-slider', 'disabled'),
                 Output('z-slider', 'marks')],
                Input('umap-plot', 'clickData')
            )
            def handle_click(clickData):
                print(f"handle_click called with clickData: {clickData}")
                if clickData and 'points' in clickData and len(clickData['points']) > 0:
                    point = clickData['points'][0]
                    # Extract filename from the text field in click data
                    # Format: "Index: 2796<br>File: embeddings_20240805-1_001009-0.npy"
                    text = point.get('text', '')
                    fname = None
                    if 'File:' in text:
                        # Extract filename from text
                        file_part = text.split('File:')[1].strip()
                        # Remove any HTML tags if present
                        fname = file_part.replace('<br>', '').replace('</br>', '').strip()
                        print(f"Extracted filename from click data: {fname}")
                    
                    if fname:
                        if load_image_data_from_file(fname):
                            data = image_data_store['data']
                            time_points, num_channels, z_slices, y_dim, x_dim = data.shape
                            
                            folder, basename = osp.split(image_data_store['path'])
                            folder = osp.basename(folder)
                            
                            info_text = f"File: {basename}<br>Folder: {folder}<br>Embedding: {image_data_store.get('fname', 'N/A')}<br>Shape: {data.shape}"
                            
                            return (
                                '',  # click-output
                                html.Div([html.P(info_text, style={'white-space': 'pre-wrap'})]),  # image-info
                                time_points - 1, 0, False, {},  # time slider
                                z_slices - 1, z_slices // 2, False, {}  # z slider
                            )
                        else:
                            return (
                                '', html.Div([html.P("Image not found")]),
                                1, 0, True, {}, 1, 0, True, {}
                            )
                    else:
                        print("Could not extract filename from click data")
                        return (
                            '', html.Div([html.P("Could not extract filename from click data")]),
                            1, 0, True, {}, 1, 0, True, {}
                        )
                
                return ('', html.Div([]), 1, 0, True, {}, 1, 0, True, {})
            
            @app.callback(
                [Output('channel1-plot', 'figure'),
                 Output('channel2-plot', 'figure')],
                [Input('time-slider', 'value'),
                 Input('z-slider', 'value')]
            )
            def update_images(time_val, z_val):
                if image_data_store['data'] is None:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="Click on a point to load image", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    return empty_fig, empty_fig
                
                data = image_data_store['data']
                time_points, num_channels, z_slices, y_dim, x_dim = data.shape
                
                time_idx = int(time_val) if time_val is not None else 0
                z_idx = int(z_val) if z_val is not None else z_slices // 2
                
                # Ensure indices are valid
                time_idx = max(0, min(time_idx, time_points - 1))
                z_idx = max(0, min(z_idx, z_slices - 1))
                
                # Get slices for both channels
                channel1_slice = data[time_idx, 0, z_idx, :, :]
                channel2_slice = data[time_idx, 1, z_idx, :, :] if num_channels >= 2 else np.zeros_like(channel1_slice)
                
                # Create Plotly figures
                fig1 = go.Figure(data=go.Heatmap(
                    z=channel1_slice,
                    colorscale='gray',
                    showscale=True
                ))
                fig1.update_layout(
                    title=f"Channel 1 (Time: {time_idx}, Z: {z_idx})",
                    width=400,
                    height=400,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(autorange='reversed')
                )
                
                fig2 = go.Figure(data=go.Heatmap(
                    z=channel2_slice,
                    colorscale='gray',
                    showscale=True
                ))
                fig2.update_layout(
                    title=f"Channel 2 (Time: {time_idx}, Z: {z_idx})",
                    width=400,
                    height=400,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(autorange='reversed')
                )
                
                return fig1, fig2
            
            # Run Dash app
            print("\n" + "="*60)
            print("Starting interactive visualization server...")
            print("Click on any point in the plot to view its image.")
            print("Server running at http://127.0.0.1:1050")
            print("Press Ctrl+C to stop.")
            print("="*60 + "\n")
            
            app.run(debug=False, port=1050)
            
        except ImportError as e:
            print(f"Warning: Required packages not available: {e}")
            print("Install with: pip install dash pillow")
            print("Falling back to basic Plotly display.")
            fig.show()
    else:
        return fig

    return fig

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
    parser.add_argument("--visualize", type=bool, default=False, help="Whether to visualize the embeddings")
    parser.add_argument('--embedding_folder', help='Path to embedding folder', default="embeddings")
    parser.add_argument('--output_name', help='Name of the output file', default="entropy_metrics")
    parser.add_argument('--evaluate', type=bool, default=True, help="Whether to evaluate the embeddings")
    args = parser.parse_args()

    embeddings_dir = osp.join(args.checkpoint_dir, args.embedding_folder)
    folder_to_label, label_names, folder_to_drug, label_to_drug_dict = load_folder_label_maps(args.drugs_to_labels)
    colors = load_colors(args.colors_file)
    maybe_build_umap_embeddings(embeddings_dir, folder_to_label, label_names)

    if args.visualize:
        select_and_plot_embedding(embeddings_dir=embeddings_dir, colors_palette=colors, )
 
    
    # make_mitospace_minimal(embedding_dir=embeddings_dir,
    #                        pick_labels=args.pick_labels,
    #                        color_palette=colors)
    if args.evaluate:
        print("Computing confusion matrix and entropy from embeddings folder")
        metrics = compute_confusion_matrix_and_entropy_from_embeddings_folder(embeddings_dir, folder_to_drug, folder_to_label, label_drug_dict=label_to_drug_dict)
        print(metrics)
        import json
        with open(osp.join(args.checkpoint_dir, "entropy_metrics_"+args.output_name+".json"), "w") as f:
            json.dump(metrics, f)


