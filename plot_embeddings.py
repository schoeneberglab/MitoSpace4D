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

def plot_embeddings(cfg):
    all_embeddings = []
    all_labels = []
    drug_names = []  # To store drug names for the legend
    
    # Load embeddings from all checkpoint directories
    for save_path in cfg.save_paths:
        embeddings_dir = f"{save_path}/embeddings"
        if not os.path.exists(embeddings_dir):
            print(f"Warning: Directory {embeddings_dir} does not exist. Skipping...")
            continue
            
        embeddings_files = os.listdir(embeddings_dir)
        
        for filepath in embeddings_files:
            embedding = np.load(f"{embeddings_dir}/{filepath}").reshape(1, -1)
            label = save_path.split("_")[1]
            
            all_embeddings.append(embedding)
            all_labels.append(label)
            if label not in drug_names:
                drug_names.append(label)

    # Combine all embeddings
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    
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
    for drug_name in drug_names:
        mask = [label == drug_name for label in all_labels]
        drug_points = umap_embeddings[mask]
        
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
            "checkpoint_20240729",
            "checkpoint_20240826",
            "checkpoint_20240830"
        ],
        base_paths=[
            "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/",
            "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240826/",
            "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240830/"
        ],
        embeddings_filepaths=[]  # Will be populated automatically for each save_path
    )
    plot_embeddings(cfg)