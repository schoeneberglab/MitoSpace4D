import os
import numpy as np
import umap
import plotly.express as px
from dataclasses import dataclass
import plotly.graph_objects as go

# Load drug to label mappings
drug_to_labels_dict = {}
folder_to_label = {}
folder_to_drug = {}
with open(f"/home/mayunagupta/experiments/MitoSpace4D/extraction_utils/drugs_to_labels.txt", 'r') as f:
    for line in f:
        folder, drug, label = line.split()
        drug_to_labels_dict[drug] = int(label)
        folder_to_label[folder] = int(label)
        folder_to_drug[folder] = drug

@dataclass
class Config_embeddings:
    save_paths: list  # List of checkpoint paths
    base_paths: list  # List of corresponding base paths
    embeddings_filepaths: list  # Will be populated for each save_path
    batch_size: int = 200  # Number of embeddings to load at a time

def plot_embeddings(cfg):
    # Step 1: Collect all file paths and metadata first
    all_file_info = []
    drug_names = []
    drug_labels = []
    volume_id_counter = 0
    
    for checkpoint_idx, save_path in enumerate(cfg.save_paths):
        embeddings_dir = f"{save_path}/embeddings"
        if not os.path.exists(embeddings_dir):
            print(f"Warning: Directory {embeddings_dir} does not exist. Skipping...")
            continue
            
        embeddings_files = sorted(os.listdir(embeddings_dir))  # Sort for consistency
        # drug_label = save_path.split("_")[1]
        
       
        
        for filepath in embeddings_files:
            drug_folder = os.path.basename(filepath).split("_")[1]
            if drug_folder not in drug_names:
                drug_names.append(folder_to_drug[drug_folder])
                drug_labels.append(folder_to_label[drug_folder])
            all_file_info.append({
                'filepath': f"{embeddings_dir}/{filepath}",
                'drug_label': folder_to_label[drug_folder],
                'checkpoint_idx': checkpoint_idx,
                'volume_id': volume_id_counter
            })
            volume_id_counter += 1
    
    print(f"Total embeddings to load: {len(all_file_info)}")
    
    # Step 2: Load embeddings in batches
    all_embeddings_batches = []
    all_labels = []
    
    for i in range(0, len(all_file_info), cfg.batch_size):
        batch = all_file_info[i:i + cfg.batch_size]
        batch_num = i // cfg.batch_size + 1
        total_batches = (len(all_file_info) + cfg.batch_size - 1) // cfg.batch_size
        print(f"Loading batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        batch_embeddings = []
        
        for file_info in batch:
            try:
                embedding = np.load(file_info['filepath']).reshape(1, -1)
                batch_embeddings.append(embedding)
                all_labels.append(file_info['drug_label'])
            except Exception as e:
                print(f"Error loading {file_info['filepath']}: {e}")
                continue
        
        if batch_embeddings:
            batch_array = np.concatenate(batch_embeddings, axis=0)
            all_embeddings_batches.append(batch_array)
    
    # Step 3: Combine all batches into final array
    print("Combining all embedding batches...")
    combined_embeddings = np.concatenate(all_embeddings_batches, axis=0)
    
    # Fit UMAP on combined embeddings
    umap_embeddings = umap.UMAP(
        n_neighbors=10,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    ).fit_transform(combined_embeddings)

    # Create scatter plot using Plotly Graph Objects for more control
    fig = go.Figure()

    # Add traces for each drug
    for drug_label, drug_name in zip(drug_labels, drug_names):
        mask = [label == drug_label for label in all_labels]
        drug_points = umap_embeddings[mask]
        try:
            fig.add_trace(go.Scatter(
                x=drug_points[:, 0],
                y=drug_points[:, 1],
                mode='markers',
                name=f"{drug_name} ({folder_to_drug[drug_name]})",
                marker=dict(
                    size=8,
                    opacity=0.7
                )
            ))
        except Exception as e:
            fig.add_trace(go.Scatter(
                x=drug_points[:, 0],
                y=drug_points[:, 1],
                mode='markers',
                name=f"{drug_name} (combined)",
                marker=dict(
                    size=8,
                    opacity=0.7
                )
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="UMAP of Combined Drug Embeddings",
            x=0.5,
            y=0.95,
            xanchor='center',
            font=dict(size=24)
        ),
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend=dict(
            title="Drugs",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        plot_bgcolor='white',
        width=1200,
        height=800
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.show()

if __name__ == "__main__":
    cfg = Config_embeddings(
        save_paths=[
            # "checkpoint_20240729",
            # "checkpoint_20240826",
            # "checkpoint_20240830"
            "checkpoint_combined_drugs"
        ],
        base_paths=[
            "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/",
            "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240826/",
            "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240830/"
        ],
        embeddings_filepaths=[],  # Will be populated automatically for each save_path
        batch_size=200  # Load 200 embeddings at a time to manage memory
    )
    plot_embeddings(cfg)