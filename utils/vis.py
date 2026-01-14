import copy
import open3d as o3d
import matplotlib.patches as mpatches
import os.path as osp
import random
import colorsys

import skimage
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import napari

cmap = LinearSegmentedColormap.from_list('blackgreen', ["k", "lime"], N=256)


def generate_distinct_colors(n):
    colors = []
    for i in range(int(n)):
        hue = (i * 1.0 / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append((r, g, b))
    return colors


def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    pcd_data = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(pcd_data.paths[0])
    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def add_to_viewer(viewer, img, translate, channel=0, label=""):
    # visualising mito channel, change the index to 0 for tmrm channel
    viewer.add_image(img[:, channel], name=f"{label}", translate=translate, colormap='cyan')


def pick_points(pcd, labels, label_names, image_paths=None):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)

    vis.run()  # user picks points
    idxs = vis.get_picked_points()

    napari_viewer = napari.Viewer()

    drug_names = []
    picked_image_paths = []
    imgs = []
    for i, idx in enumerate(idxs):
        drug_names.append(label_names[(labels[idx]%27)])
        picked_image_paths.append(image_paths[idx])
        if image_paths is not None:
            img_4d = np.load(image_paths[idx])

            # skimage.io.imsave("/home/dhruvagarwal/Desktop/p110.tiff", img_4d[0, 0])

            add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 0), channel=0, label=label_names[(labels[idx]%27)])
            add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 256 + 10), channel=1, label=label_names[(labels[idx]%27)])

            img_4d = img_4d.astype(np.float32)
            img_4d[:, 0] = np.clip(img_4d[:, 0], 0., 25000) / 25000
            img_4d[:, 1] = np.clip(img_4d[:, 1], 0., 10000) / 10000

            imgs.append((img_4d[0, :, :].max(axis=1)))

    napari_viewer.window.add_plugin_dock_widget(
        plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
    )

    napari.run()

    print(drug_names)
    print(picked_image_paths)

    if len(idxs) == 0:
        return

    colors = np.array(pcd.colors)[idxs]

    f, axarr = plt.subplots(2, len(imgs), figsize=(20, 20))
    f.suptitle("MIP", fontsize=50)
    vals = []
    for i in range(len(imgs)):
        mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
        tmrm_idx = 0
        axarr[0, i].imshow(imgs[i][tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
        axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)

        vals.append(np.mean(imgs[i][:, :, 0]))
        axarr[0, i].set_xticks([])
        # for minor ticks
        axarr[0, i].set_yticks([])

        axarr[1, i].set_xticks([])
        # for minor ticks
        axarr[1, i].set_yticks([])

    print(vals)
    plt.xticks([]), plt.yticks([])
    for idx in idxs:
        print(idx)
        print(labels[idx])
        print(label_names[int(labels[idx]%27)])
        print("")
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=label_names[int(labels[idxs[i]]%27)])) for i in
               range(len(idxs))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    print("")
    return vis.get_picked_points()

def make_mitospace(embedding_dir, pick_labels=None, color_palette=None, image_paths=None):
    EMBEDDING_PATH = osp.join(embedding_dir, 'embeddings_umap.npy')
    LABEL_PATH = osp.join(embedding_dir, 'labels.npy')
    LABEL_NAME_PATH = osp.join(embedding_dir, 'label_names.npy')

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)

    # labels = labels.repeat(20)
    # print(labels, labels.shape, labels.min(), labels.max())
    # pick labels present in the pick_labels list
    if pick_labels is not None:
        mask = np.isin(labels, pick_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]
        if image_paths is not None:
            image_paths = [image_paths[i] for i in range(len(image_paths)) if mask[i]]

        # to visualise the temporal progression in the embeddings
        # for temp, label in enumerate(labels):
        #     labels[temp] = (int(temp) % 20)

    label_names = np.load(LABEL_NAME_PATH)

    max_label = labels.max()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(embeddings)

    # color_palette = generate_distinct_colors(max_label+1)
    # print(color_palette)
    colors = np.array([color_palette[int(label)] for label in labels])
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = np.array([0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # save the pcd
    o3d.io.write_point_cloud('/home/mayunagupta/experiments/MitoSpace4D/4d_mitospace.pcd', pcd)

    opt = vis.get_render_option()
    opt.point_size = 10

    # vis.add_geometry(mesh_frame)
    vis.run()

    # Create legend
    legend_patches = [mpatches.Patch(color=color_palette[i], label=label_names[i]) for i in range(len(label_names))]
    plt.figure(figsize=(10, 10))
    plt.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, 0.5))
    plt.axis('off')
    plt.show()

    while True:
        pick_points(pcd, labels, label_names, image_paths)

def pick_points_flybrain(pcd, labels, label_names, image_paths=None, chunk_idxs=None):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)

    vis.run()  # user picks points
    idxs = vis.get_picked_points()

    if len(idxs) == 0:
        print("No points selected.")
        return []

    napari_viewer = napari.Viewer()

    drug_names = []
    picked_image_paths = []
    picked_chunk_idxs = []
    mip_images = []
    chunk_stacks = []
    
    for i, idx in enumerate(idxs):
        label_idx = int(labels[idx])
        if label_idx < len(label_names):
            drug_name = label_names[label_idx]
        else:
            drug_name = f"Label_{label_idx}"
        
        drug_names.append(drug_name)
        
        if image_paths is not None and idx < len(image_paths):
            chunk_file_path = image_paths[idx]
            picked_image_paths.append(chunk_file_path)
            
            # Load the chunk file (contains multiple chunks)
            chunks_data = np.load(chunk_file_path)  # Shape: (162, 1, 2, 114, 256, 256) or similar
            
            # Get the specific chunk index
            if chunk_idxs is not None and idx < len(chunk_idxs):
                chunk_idx = chunk_idxs[idx]
            else:
                chunk_idx = 0  # Default to first chunk if not provided
            
            picked_chunk_idxs.append(chunk_idx)
            
            # Extract the specific chunk
            if chunk_idx < len(chunks_data):
                selected_chunk = chunks_data[chunk_idx]  # Shape: (1, 2, 114, 256, 256)
                selected_chunk = selected_chunk.squeeze(0)  # Remove batch dim: (2, 114, 256, 256)
                
                # Normalize the data
                chunk_normalized = selected_chunk.astype(np.float32)
                chunk_normalized[0] = np.clip(chunk_normalized[0], 0., chunk_normalized[0].max()) / (chunk_normalized[0].max() + 1e-6)
                chunk_normalized[1] = np.clip(chunk_normalized[1], 0., chunk_normalized[1].max()) / (chunk_normalized[1].max() + 1e-6)
                
                # Compute Maximum Intensity Projections (MIP) for each channel
                mip_channel_0 = np.max(chunk_normalized[0], axis=0)  # Shape: (256, 256)
                mip_channel_1 = np.max(chunk_normalized[1], axis=0)  # Shape: (256, 256)
                
                mip_images.append({
                    'channel_0': mip_channel_0,
                    'channel_1': mip_channel_1,
                    'label': drug_name
                })
                
                chunk_stacks.append({
                    'channel_0': chunk_normalized[0],  # Shape: (114, 256, 256)
                    'channel_1': chunk_normalized[1],  # Shape: (114, 256, 256)
                    'label': drug_name
                })
                
                # Add MIPs to napari viewer, positioned side by side
                translate_x = i * 270  # Space images horizontally
                napari_viewer.add_image(
                    mip_channel_0, 
                    name=f'{drug_name} - Ch0 MIP (Chunk {chunk_idx})',
                    translate=(translate_x, 0),
                    colormap='cyan',
                    blending='additive'
                )
                napari_viewer.add_image(
                    mip_channel_1,
                    name=f'{drug_name} - Ch1 MIP (Chunk {chunk_idx})',
                    translate=(translate_x, 260),
                    colormap='magenta',
                    blending='additive'
                )
                
                # Add full 3D stacks (hidden by default for cleaner view)
                napari_viewer.add_image(
                    chunk_normalized[0],
                    name=f'{drug_name} - Ch0 Stack (Chunk {chunk_idx})',
                    translate=(translate_x, 0),
                    colormap='cyan',
                    blending='additive',
                    visible=False
                )
                napari_viewer.add_image(
                    chunk_normalized[1],
                    name=f'{drug_name} - Ch1 Stack (Chunk {chunk_idx})',
                    translate=(translate_x, 0),
                    colormap='magenta',
                    blending='additive',
                    visible=False
                )
            else:
                print(f"Warning: Chunk index {chunk_idx} out of range for file {chunk_file_path}")
        else:
            print(f"Warning: Index {idx} out of range for image_paths")

    napari_viewer.window.add_plugin_dock_widget(
        plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
    )

    print("\nSelected points:")
    print(f"Drug names: {drug_names}")
    print(f"Image paths: {picked_image_paths}")
    print(f"Chunk indices: {picked_chunk_idxs}")
    print("\nOpening napari viewer...")
    
    napari.run()

    # Create matplotlib plots for the selected points
    if len(mip_images) > 0:
        colors = np.array(pcd.colors)[idxs]
        
        num_selected = len(mip_images)
        f, axarr = plt.subplots(2, num_selected, figsize=(4*num_selected, 8))
        if num_selected == 1:
            axarr = axarr.reshape(-1, 1)
        
        f.suptitle("Maximum Intensity Projections (MIP)", fontsize=16)
        
        for i in range(num_selected):
            mip_data = mip_images[i]
            
            # Channel 0 (TMRM-like) - top row
            im0 = axarr[0, i].imshow(mip_data['channel_0'], vmin=0., vmax=1., cmap=plt.cm.hot)
            axarr[0, i].set_title(f"Ch0 - {mip_data['label']}\n(Chunk {picked_chunk_idxs[i]})", fontsize=10)
            axarr[0, i].set_xticks([])
            axarr[0, i].set_yticks([])
            plt.colorbar(im0, ax=axarr[0, i], fraction=0.046)
            
            # Channel 1 (Mitochondria) - bottom row
            im1 = axarr[1, i].imshow(mip_data['channel_1'], vmin=0., vmax=1., cmap=plt.cm.viridis)
            axarr[1, i].set_title(f"Ch1 - {mip_data['label']}\n(Chunk {picked_chunk_idxs[i]})", fontsize=10)
            axarr[1, i].set_xticks([])
            axarr[1, i].set_yticks([])
            plt.colorbar(im1, ax=axarr[1, i], fraction=0.046)
        
        # Create legend with colors matching the point cloud
        patches = [mpatches.Patch(color=colors[i], label=f"{drug_names[i]} (idx: {idxs[i]})") 
                   for i in range(len(idxs))]
        f.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information
        print("\nDetailed information:")
        for i, idx in enumerate(idxs):
            print(f"\nPoint {i+1}:")
            print(f"  Embedding index: {idx}")
            print(f"  Label: {labels[idx]} ({drug_names[i]})")
            print(f"  Chunk file: {picked_image_paths[i]}")
            print(f"  Chunk index: {picked_chunk_idxs[i]}")
            print(f"  Color (RGB): {colors[i]}")
    
    print("")
    return idxs


def make_mitospace_flybrain(embedding_dir, pick_labels=None, color_palette=None, image_paths=None, chunk_idxs=None):
    EMBEDDING_PATH = osp.join(embedding_dir, 'embeddings_umap.npy')
    LABEL_PATH = osp.join(embedding_dir, 'labels.npy')
    LABEL_NAME_PATH = osp.join(embedding_dir, 'label_names.npy')

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)

    # labels = labels.repeat(20)
    # print(labels, labels.shape, labels.min(), labels.max())
    # pick labels present in the pick_labels list
    # if pick_labels is not None:
    #     mask = np.isin(labels, pick_labels)
    #     embeddings = embeddings[mask]
    #     labels = labels[mask]
    #     if image_paths is not None:
    #         image_paths = [image_paths[i] for i in range(len(image_paths)) if mask[i]]

        # to visualise the temporal progression in the embeddings
        # for temp, label in enumerate(labels):
        #     labels[temp] = (int(temp) % 20)

    label_names = np.load(LABEL_NAME_PATH)

    max_label = labels.max()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(embeddings)

    color_palette = generate_distinct_colors(max_label+1)
    print(color_palette)
    colors = np.array([color_palette[int(label)] for label in labels])
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = np.array([0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # save the pcd
    o3d.io.write_point_cloud('/home/mayunagupta/experiments/MitoSpace4D/4d_mitospace.pcd', pcd)

    opt = vis.get_render_option()
    opt.point_size = 10

    # vis.add_geometry(mesh_frame)
    vis.run()

    # Create legend
    legend_patches = [mpatches.Patch(color=color_palette[i+1], label=label_names[i]) for i in range(len(label_names))]
    plt.figure(figsize=(10, 10))
    plt.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, 0.5))
    plt.axis('off')
    plt.show()

    while True:
        print("pick_points_flybrain")
        # print(image_paths)
        pick_points_flybrain(pcd, labels, label_names, image_paths, chunk_idxs)


def plot_confusion_matrix(cm,
                          label_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          k=100):
    cm_unnorm = cm.copy()

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(20, 20))

    tickmarks = np.arange(cm.shape[0])
    plt.xticks(tickmarks, label_names, rotation=45)
    plt.yticks(tickmarks, label_names)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, cmap=cmap, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    thresh = cm.max() / 1.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm_unnorm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_cm(gt_labels, pred_labels, label_drug_dict, verbose=True, make_plot=True):
    cm = confusion_matrix(gt_labels, pred_labels, labels=sorted(list(label_drug_dict.keys())))

    if verbose:
        print("per class accuracy Top-1")
        for i in range(cm.shape[0]):
            print(f"{label_drug_dict[i]}: {cm[i, i] * 100. / np.sum(cm[i, :])}%")

    if make_plot:
        plot_confusion_matrix(cm,
                              list(label_drug_dict.values()),
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True,
                              k=100)

    return cm


def visualise_images_oldData(data_paths, save_dir, drug_names, num_images_per_drug=200, seed=1123):
    """deprecate this function and use the new one. Only to be used for old Cal27 and HeLa data"""

    images = [np.load(osp.join(data_path, 'train_samples.npy')) for data_path in data_paths]
    labels = [np.load(osp.join(data_path, 'train_labels.npy')) for data_path in data_paths]

    num_drugs = len(drug_names)

    label_idx_maps = [{} for _ in range(len(data_paths))]
    for i, label in enumerate(labels):
        for idx, l in enumerate(label):
            if l not in label_idx_maps[i]:
                label_idx_maps[i][l] = []
            label_idx_maps[i][l].append(idx)

    # sort the label_idx_maps according to key values
    label_idx_maps = [dict(sorted(x.items())) for x in label_idx_maps]

    for drug_label in np.unique(labels[0]):
        sep = 10
        h = 256
        w = 256
        grid = np.zeros((h * num_images_per_drug, w * len(data_paths) * 2 + sep * (len(data_paths) - 1)))

        selected_idxs = [[] for _ in range(len(data_paths))]
        for i, label_idx_map in enumerate(label_idx_maps):
            idxs = label_idx_map[drug_label].copy()
            random.seed(seed)
            random.shuffle(idxs)
            selected_idxs[i].extend(idxs[:num_images_per_drug])

        for i in range(num_images_per_drug):
            for j in range(len(data_paths)):
                img = images[j][selected_idxs[j][i]]
                grid[i * h:(i + 1) * h, j * 2 * w + j * sep: (j * 2 + 1) * w + j * sep] = img[0]
                grid[i * h:(i + 1) * h, (j * 2 + 1) * w + j * sep: (j * 2 + 2) * w + j * sep] = img[1]

        # save in grayscale
        img = Image.fromarray(np.uint8(grid * 255.))
        img.save(osp.join(save_dir, f"vis_{drug_names[drug_label]}") + "_gray.png")


def visualise_images(data_paths, save_dir, label_drug_dict, num_images_per_drug=200, seed=1123):
    """
    The function takes in the data path of all the cell lines and plots 1 random image per label for each of those cell
    lines the Grid has 2*num_cell_lines column. The first column for each cell is the TMRM and the second one is
    Mitotracker
    """

    images = []
    labels = []
    for idx, data_path in enumerate(data_paths):
        filenames = sorted(os.listdir(data_path + "/processed_data"))
        # randomly select 20000 files

        random.seed(seed)
        idxs = random.sample(range(len(filenames)), 20000)
        filenames = [filenames[i] for i in idxs]

        img_data_path = np.stack([np.load(osp.join(data_path, "processed_data", x))
                                  for x in filenames])
        images.append(img_data_path)
        labels.append(np.load(osp.join(data_path, 'labels.npy'))[idxs])

    # images = [np.load(osp.join(data_path, 'train_samples.npy')) for data_path in data_paths]
    # labels = [np.load(osp.join(data_path, 'train_labels.npy')) for data_path in data_paths]

    label_idx_maps = [{} for _ in range(len(data_paths))]
    for i, label in enumerate(labels):
        for idx, l in enumerate(label):
            if l not in label_idx_maps[i]:
                label_idx_maps[i][l] = []
            label_idx_maps[i][l].append(idx)

    # sort the label_idx_maps according to key values
    label_idx_maps = [dict(sorted(x.items())) for x in label_idx_maps]

    # find common labels from the keys of the label_idx_maps
    common_labels = [set(label_idx_maps[i].keys()) for i in range(len(data_paths))]
    common_labels = set.intersection(*common_labels)

    for drug_label in common_labels:
        sep = 10
        h = 256
        w = 256
        grid = np.zeros((h * num_images_per_drug, w * len(data_paths) * 2 + sep * (len(data_paths) - 1)))

        selected_idxs = [[] for _ in range(len(data_paths))]
        for i, label_idx_map in enumerate(label_idx_maps):
            idxs = label_idx_map[drug_label].copy()
            random.seed(seed)
            random.shuffle(idxs)
            selected_idxs[i].extend(idxs[:num_images_per_drug])

        min_num_images = min([len(x) for x in selected_idxs])
        for i in range(min(num_images_per_drug, min_num_images)):
            for j in range(len(data_paths)):
                img = images[j][selected_idxs[j][i]]
                grid[i * h:(i + 1) * h, j * 2 * w + j * sep: (j * 2 + 1) * w + j * sep] = img[0]
                grid[i * h:(i + 1) * h, (j * 2 + 1) * w + j * sep: (j * 2 + 2) * w + j * sep] = img[1]

        # save in grayscale
        # img = Image.fromarray(np.uint8(grid * 255.))
        # img.save(osp.join(save_dir, f"vis_{label_drug_dict[drug_label]}") + "_gray.png")

        # save in viridis
        cm = plt.get_cmap("viridis")
        grid = cm(grid)[:, :, :3]
        grid = np.uint8(grid * 255.)
        img = Image.fromarray(grid)
        img.save(osp.join(save_dir, f"vis_{label_drug_dict[drug_label]}") + "_virdis.png")


if __name__ == '__main__':
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace"
    data_paths = [f"{proj_dir}/data/Cal27NewHiroAndre/20240314/",
                  f"{proj_dir}/data/Cal27NewHiroAndre/20240320/"]

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    save_dir = f"{proj_dir}/dumps/"
    visualise_images(data_paths, save_dir, label_drug_dict,
                     num_images_per_drug=200, seed=1123)
