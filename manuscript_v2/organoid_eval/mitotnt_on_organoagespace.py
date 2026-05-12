import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import open3d as o3d

from interpretability.mitotnt_on_mitospace import generate_color_intensities


def get_feats(features, identifiers, embeddings, feature_to_plot):
    # check there are no nans
    assert features[feature_to_plot].isna().sum() == 0
    identifiers_csv = features['Unnamed: 0']
    identifiers_csv = [str(id) for id in identifiers_csv]

    features_values_to_plot = []
    embeddings_filtered = []
    for embd_idx, id in enumerate(identifiers):
        if id in identifiers_csv:
            idx = identifiers_csv.index(id)
            features_values_to_plot.append(features[feature_to_plot][idx])
            embeddings_filtered.append(embeddings[embd_idx])
        else:
            print(f"Identifier {id} not found in the features csv file.")

    embeddings_filtered = np.array(embeddings_filtered)
    return features_values_to_plot, embeddings_filtered


def preprocess(feature_to_plot):
    feature_to_plot = np.array(feature_to_plot)
    feature_to_plot = (feature_to_plot - feature_to_plot.min()) / (feature_to_plot.max() - feature_to_plot.min())
    feature_to_plot = feature_to_plot**0.4

    # plot the distribution of the feature
    plt.hist(feature_to_plot, bins=50)
    plt.show()

    return feature_to_plot


def get_umap(embeddings):
    reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=25, min_dist=0.01, metric='cosine')
    embeddings = reducer.fit_transform(embeddings)
    return embeddings


if __name__ == "__main__":
    root_path = '/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/'
    embeddings = np.load(osp.join(root_path, 'OrganoAgeSpace_umap.npy'))
    identifiers = np.load(osp.join(root_path, 'identifiers.npy'))
    labels = np.load(osp.join(root_path, 'labels.npy'))
    mitotnt_features = pd.read_csv(osp.join(root_path, 'df_all_bins_20250303.csv'))

    feature_to_plot = 'ave mito to membrane'

    feature_to_plot, embeddings = get_feats(mitotnt_features, identifiers, embeddings, feature_to_plot)

    feature_to_plot = preprocess(feature_to_plot)

    colors = generate_color_intensities(feature_to_plot)

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

    opt = vis.get_render_option()
    opt.point_size = 3

    # vis.add_geometry(mesh_frame)
    vis.run()


