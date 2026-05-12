import numpy as np
import pandas as pd
import os
import open3d as o3d
import matplotlib.pyplot as plt
import os.path as osp
from utils.vis import pick_points


def generate_color_intensities(feature_values, colormap_name='gnuplot2'):
    """
    Generate RGB color intensities for a list of feature values using a colormap.

    Args:
        feature_values (list or np.ndarray): List of feature values.
        colormap_name (str): Name of the colormap to use. Default is 'gnuplot2'.

    Returns:
        list: List of RGB tuples representing color intensities.
    """
    # Normalize feature values to the range [0, 1]
    feature_values = np.array(feature_values)

    top_1_percentile = np.percentile(feature_values, 99)
    bottom_1_percentile = np.percentile(feature_values, 1)

    feature_values = np.clip(feature_values, bottom_1_percentile, top_1_percentile)

    # find nan values and replace them with the mean of the rest of the values
    nan_idxs = np.where(np.isnan(feature_values))[0]
    feature_values[nan_idxs] = np.nanmean(feature_values)

    # Load the specified colormap
    colormap = plt.get_cmap(colormap_name)

    # normalize the features such that the full range of the colormap is used
    normalized_values = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))

    plt.hist(normalized_values, bins=20)
    plt.show()

    # Map normalized values to colors
    colors = colormap(normalized_values)

    # Extract RGB intensities and convert them to a list of tuples
    rgb_colors = [tuple(color[:3]) for color in colors]  # Ignore alpha channel

    return np.array(rgb_colors)


def get_feats_and_embs(mitotnt_features, umap_emb, feature, identifier, data_path_root, folder_to_label):
    # Filter out NaN values from the feature column
    feature_values = mitotnt_features[feature].dropna()

    # Keep only the rows without NaNs
    mitotnt_features = mitotnt_features.loc[feature_values.index]
    drug_folder = mitotnt_features['Date'].astype(str)
    fname = mitotnt_features['Cell ID'].apply(lambda x: str(x).zfill(6))

    # Create a mapping for faster lookups
    identifier_dict = {tuple(id_pair): idx for idx, id_pair in enumerate(identifier)}

    # Find embeddings for valid rows
    embeddings = []
    not_found = []
    image_paths = []
    labels = []

    for i in range(len(drug_folder)):
        key = (drug_folder.iloc[i], fname.iloc[i])
        if key in identifier_dict:
            embeddings.append(umap_emb[identifier_dict[key]])
            image_paths.append(os.path.join(data_path_root, drug_folder.iloc[i], f'{fname.iloc[i]}.npy'))
            labels.append(folder_to_label[drug_folder.iloc[i]])
        else:
            not_found.append(key)

    if not_found:
        print(f'Not found: {len(not_found)} items. Example: {not_found[:5]}')

    return feature_values.values, embeddings, labels, image_paths


def get_identifiers(img_file_paths):
    identifiers = []
    for img_file_path in img_file_paths:
        fname, drug_folder = img_file_path.split('/')[-1].split('.')[0], img_file_path.split('/')[-2]
        identifiers.append([drug_folder, fname])

    return identifiers


if __name__ == '__main__':
    umap_emb = np.load(
        '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/embeddings_umap.npy')
    img_file_paths = np.load(
        '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/image_file_paths.npy')
    labels = np.load(
        '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings/labels.npy')
    data_path_root = '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/'
    embeddings_root = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings'

    pick_labels = [0, 13]

    mask = np.isin(labels, pick_labels)
    umap_emb = umap_emb[mask]
    labels = labels[mask]
    img_file_paths = img_file_paths[mask]

    folder_to_label = {}
    with open(f"/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            folder_to_label[folder] = int(label)

    embeddings = np.array(umap_emb)

    feature_values = [int(x.split('/')[-1].split('.')[0]) for x in img_file_paths]
    feature_values = np.array(feature_values)

    # Generate RGB color intensities for the feature values
    colors = generate_color_intensities(feature_values)

    # Create a point cloud from the embeddings
    # embeddings[:, [0, 1]] = np.repeat((embeddings[:, [0]] + embeddings[:, [1]])/2, 2, 1)
    # embeddings[:, [2]] = 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(embeddings)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # add the coordinate frame in the point cloud
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd, mesh_frame])

    # draw the point cloud in a zoomed out view
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
    vis.run()

    LABEL_NAME_PATH = osp.join(embeddings_root, 'label_names.npy')
    label_names = np.load(LABEL_NAME_PATH)

    while True:
        pick_points(pcd, labels, label_names, img_file_paths)

