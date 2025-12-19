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
import torch
from utils.utils import get_drug_label_maps

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

def plot_arrows(pcd, arrows):
    o3d.visualization.draw_geometries([pcd] + arrows)

def get_per_frame_vals(vals, n_frames=20):
    """ Repeats the image paths by n_frames """
    frame_vals = []
    for val in vals:
        frame_vals.extend(np.repeat(val, n_frames).tolist())
    return np.array(frame_vals)

def pick_points(pcd, labels, label_names, image_paths=None, image_times=None, label_drug_dict=None, decoder=None):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")

    # get_label_name_fn = lambda idx: label_names[(labels[idx])]

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    print(f"Number of Points: {len(pcd.points)}")
    vis.get_render_option().point_size = 5.0

    vis.run()  # user picks points
    idxs = vis.get_picked_points()

    if len(idxs) == 0:
        print("No points picked, closing the visualizer.")
        vis.destroy_window()
        exit(0)

    # TODO: Fix napari visualization routines
    # napari_viewer = napari.Viewer()

    drug_names = []
    time_indices = []
    picked_image_paths = []
    imgs = []
    for i, idx in enumerate(idxs):
        # drug_names.append(label_names[(labels[idx%20]%27)])
        # drug_names.append(get_label_name_fn(idx))
        drug_names.append(label_drug_dict[labels[idx]])
        if image_times is not None:
            time_indices.append(image_times[idx])
        
        # if image_times is not None:
        #     time_index = image_times[idx]
        #     picked_image_paths.append(image_paths[time_index])
        # else:
            # picked_image_paths.append(image_paths[idx])
        
        picked_image_paths.append(image_paths[idx])
        
        if image_paths is not None:
            img_4d = np.load(image_paths[idx])

            if decoder is not None:
                img_tensor = torch.from_numpy(img_4d).unsqueeze(0).cuda()  # (1, t, c, d, h, w)
                with torch.no_grad():
                    img_tensor = decoder(img_tensor)
                img_4d = img_tensor.squeeze(0).cpu().numpy()

            # skimage.io.imsave("/home/dhruvagarwal/Desktop/p110.tiff", img_4d[0, 0])

            # add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 0), channel=0, label=label_names[(labels[idx]%27)])
            # add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 256 + 10), channel=1, label=label_names[(labels[idx]%27)])

            img_4d = img_4d.astype(np.float32)
            # img_4d[:, 0] = np.clip(img_4d[:, 0], 0., 25000) / 25000
            # img_4d[:, 1] = np.clip(img_4d[:, 1], 0., 10000) / 10000

            if image_times is not None:
                # Get the corresponding image by index
                time_index = image_times[idx]
                # imgs.append(img_4d[time_index, :, :].max(axis=1))
                imgs.append(img_4d[:, time_index, ...].max(axis=1))
            else:
                print("Image times is None")
                # imgs.append(img_4d[0, :, :].max(axis=1))
                imgs.append(img_4d[:, 0, ...].max(axis=1))

    # napari_viewer.window.add_plugin_dock_widget(
    #     plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
    # )
    # napari.run()

    print(drug_names)
    print(picked_image_paths)

    colors = np.array(pcd.colors)[idxs]

    f, axarr = plt.subplots(2, len(imgs), figsize=(20, 20))
    f.suptitle("MIP", fontsize=50)
    vals = []
    for i in range(len(imgs)):
        mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
        tmrm_idx = 0
        
        #-- EXPERIMENT
        # print("[vis.py: pick_points] EXPERIMENT: TMRM COPIED TO MITO CHANNEL FOR CANCER DATA TRIAL 3B")
        # mito_idx = 0
        # tmrm_idx = 0

        axarr[0, i].imshow(imgs[i][tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
        # axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)

        vals.append(np.mean(imgs[i][:, :, 0]))
        axarr[0, i].set_xticks([])
        # for minor ticks
        axarr[0, i].set_yticks([])

        # axarr[1, i].set_xticks([])
        # for minor ticks
        # axarr[1, i].set_yticks([])

    print(vals)
    plt.xticks([]), plt.yticks([])
    for idx in idxs:
        print(idx)
        print(labels[idx])
        # print(label_names[int(labels[idx]%27)])
        # print(get_label_name_fn(idx))
        print(label_drug_dict[labels[idx]])
        print(image_paths[idx])
        if image_times is not None:
            print(f"Time index: {image_times[idx]}")
        print("")

    # patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=label_names[int(labels[idxs[i]]%27)])) for i in
    #            range(len(idxs))]
    # patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=get_label_name_fn(idxs[i]))) for i in range(len(idxs))]
    #
    # patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=label_drug_dict[labels[idx]])) for i in range(len(idxs))]

    patches = []
    for i, idx in enumerate(idxs):
        l = label_drug_dict[labels[idx]]
        ds_id = image_paths[idx].split("/")[-2]
        sample_id = image_paths[idx].split("/")[-1].split(".npy")[0]
        sample_caption = f"{ds_id}/{sample_id}"
        patches.append(mpatches.Patch(color=colors[i], label="{l} ({sc})".format(l=l, sc=sample_caption)))

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    print("")

    return vis.get_picked_points()

# def pick_points(pcd, labels, label_names, image_paths=None, image_times=None, label_drug_dict=None, decoder=None, visualizer='napari'):
#     print("")
#     print(
#         "1) Please pick at least three correspondences using [shift + left click]"
#     )
#     print("   Press [shift + right click] to undo point picking")
#     print("2) After picking points, press 'Q' to close the window")

#     # get_label_name_fn = lambda idx: label_names[(labels[idx])]

#     vis = o3d.visualization.VisualizerWithEditing()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     print(f"Number of Points: {len(pcd.points)}")
#     vis.get_render_option().point_size = 5.0

#     vis.run()  # user picks points
#     idxs = vis.get_picked_points()

#     if len(idxs) == 0:
#         print("No points picked, closing the visualizer.")
#         vis.destroy_window()
#         exit(0)

#     drug_names = []
#     time_indices = []
#     picked_image_paths = []
#     imgs = []
#     imgs_4d = []
#     for i, idx in enumerate(idxs):
#         # drug_names.append(label_names[(labels[idx%20]%27)])
#         # drug_names.append(get_label_name_fn(idx))
#         drug_names.append(label_drug_dict[labels[idx]])
#         if image_times is not None:
#             time_indices.append(image_times[idx])
        
#         # if image_times is not None:
#         #     time_index = image_times[idx]
#         #     picked_image_paths.append(image_paths[time_index])
#         # else:
#             # picked_image_paths.append(image_paths[idx])
        
#         picked_image_paths.append(image_paths[idx])
        
#         if image_paths is not None:
#             img_4d = np.load(image_paths[idx])

#             if decoder is not None:
#                 img_tensor = torch.from_numpy(img_4d).unsqueeze(0).cuda()  # (1, t, c, d, h, w)
#                 with torch.no_grad():
#                     img_tensor = decoder(img_tensor)
#                 img_4d = img_tensor.squeeze(0).cpu().numpy()

#             # skimage.io.imsave("/home/dhruvagarwal/Desktop/p110.tiff", img_4d[0, 0])

#             # add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 0), channel=0, label=label_names[(labels[idx]%27)])
#             # add_to_viewer(napari_viewer, img_4d, translate=(i*256 + 10, 256 + 10), channel=1, label=label_names[(labels[idx]%27)])

#             img_4d = img_4d.astype(np.float32)
#             # img_4d[:, 0] = np.clip(img_4d[:, 0], 0., 25000) / 25000
#             # img_4d[:, 1] = np.clip(img_4d[:, 1], 0., 10000) / 10000

#             imgs_4d.append(img_4d)

#             if image_times is not None:
#                 # Get the corresponding image by index
#                 time_index = image_times[idx]
#                 # imgs.append(img_4d[time_index, :, :].max(axis=1))
#                 imgs.append(img_4d[:, time_index, ...].max(axis=1))
#             else:
#                 print("Image times is None")
#                 # imgs.append(img_4d[0, :, :].max(axis=1))
#                 imgs.append(img_4d[:, 0, ...].max(axis=1))
    
#     print(drug_names)
#     print(picked_image_paths)

#     if visualizer == 'napari':
#         viewer = napari.Viewer(ndisplay=3)
#         # napari_viewer.window.add_plugin_dock_widget(
#         # plugin_name="napari-matplotlib", widget_name="FeaturesHistogram"
#         # )

#         for i, idx in enumerate(idxs):
#             img_4d = np.load(image_paths[idx])
#             img_4d = img_4d.astype(np.float32)

#             viewer.add_image(img_4d[:, 0], name=f"[0] {drug_names[i]}", translate=(i*256 + 10, 0, 0), colormap='hot') # TMRM
#             viewer.add_image(img_4d[:, 1], name=f"[1] {drug_names[i]}", translate=(i*256 + 10, 256 + 10, 0), colormap='viridis') # Morphology

#             napari.run()

#             # add_to_viewer(viewer, imgs_4d[i], translate=(i*256 + 10, 0), channel=0, label=label_drug_dict[labels[idx]])
#             # add_to_viewer(viewer, imgs_4d[i], translate=(i*256 + 10, 256 + 10), channel=1, label=label_drug_dict[labels[idx]])

#     elif visualizer == 'matplotlib':
        
#         colors = np.array(pcd.colors)[idxs]

#         f, axarr = plt.subplots(2, len(imgs), figsize=(20, 20))
#         f.suptitle("MIP", fontsize=50)
#         vals = []
#         for i in range(len(imgs)):
#             mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
#             tmrm_idx = 0
            
#             #-- EXPERIMENT
#             # print("[vis.py: pick_points] EXPERIMENT: TMRM COPIED TO MITO CHANNEL FOR CANCER DATA TRIAL 3B")
#             # mito_idx = 0
#             # tmrm_idx = 0

#             axarr[0, i].imshow(imgs[i][tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
#             axarr[1, i].imshow(imgs[i][mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)

#             vals.append(np.mean(imgs[i][:, :, 0]))
#             axarr[0, i].set_xticks([])
#             # for minor ticks
#             axarr[0, i].set_yticks([])

#             axarr[1, i].set_xticks([])
#             # for minor ticks
#             axarr[1, i].set_yticks([])

#         print(vals)
#         plt.xticks([]), plt.yticks([])
#         for idx in idxs:
#             print(idx)
#             print(labels[idx])
#             # print(label_names[int(labels[idx]%27)])
#             # print(get_label_name_fn(idx))
#             print(label_drug_dict[labels[idx]])
#             print(image_paths[idx])
#             if image_times is not None:
#                 print(f"Time index: {image_times[idx]}")
#             print("")

#         patches = []
#         for i, idx in enumerate(idxs):
#             l = label_drug_dict[labels[idx]]
#             ds_id = image_paths[idx].split("/")[-2]
#             sample_id = image_paths[idx].split("/")[-1].split(".npy")[0]
#             sample_caption = f"{ds_id}/{sample_id}"
#             patches.append(mpatches.Patch(color=colors[i], label="{l} ({sc})".format(l=l, sc=sample_caption)))

#         plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#         plt.show()
#         print("")

#     return vis.get_picked_points()

def make_mitospace(embedding_dir, pick_labels=None, color_palette=None, image_paths=None, single_frames=False, save_pcd=None, label_drug_dict=None, datasets=None, decoder=None):
    EMBEDDING_PATH = osp.join(embedding_dir, 'embeddings_umap.npy')
    LABEL_PATH = osp.join(embedding_dir, 'labels.npy')
    LABEL_NAME_PATH = osp.join(embedding_dir, 'label_names.npy')
    IMAGE_PATHS = osp.join(embedding_dir, 'image_paths.csv')
    IMAGE_TIME_PATH = osp.join(embedding_dir, 'image_times.npy')

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)
    label_names = np.load(LABEL_NAME_PATH)
    image_paths = np.loadtxt(IMAGE_PATHS, dtype=str).tolist()
    image_paths = np.array(image_paths)
    colors = color_palette
    
    if single_frames:
        if osp.exists(IMAGE_TIME_PATH):
            print(f"Loading image times from {IMAGE_TIME_PATH}")
            image_times = np.load(IMAGE_TIME_PATH) if single_frames else None

            if labels.shape != image_times.shape:
                labels = get_per_frame_vals(labels)
                image_paths = get_per_frame_vals(image_paths)
        else:
            # Create a dummy array of 0-19 repeated for each sample
            print(f"Image times file {IMAGE_TIME_PATH} does not exist, creating a dummy array")
            image_times = np.arange(20)
            # Repeat the image times for each sample
            image_times = np.tile(image_times, int(len(labels)))
            # Save to the IMAGE_TIME_PATH
            # np.save(IMAGE_TIME_PATH, image_times)
            print(f"Saved dummy image times to {IMAGE_TIME_PATH}")
            # colors = get_per_frame_vals(color_palette)
            labels = get_per_frame_vals(labels)
            image_paths = get_per_frame_vals(image_paths)
    else:
        image_times = None

    # pick labels present in the pick_labels list
    if pick_labels is not None:
        mask = np.isin(labels, pick_labels)
        
        if datasets is not None:
            img_datasets = [image_paths[i].split("/")[-2] for i in range(len(image_paths))]
            img_datasets = np.array(img_datasets)
            dataset_mask = np.isin(img_datasets, datasets)
            mask = np.logical_and(mask, dataset_mask)
        
        embeddings = embeddings[mask]
        labels = labels[mask]
        image_paths = image_paths[mask]
        if single_frames:
            image_times = image_times[mask]

        # temporal/region colormaps
        if isinstance(color_palette, np.ndarray):
            if single_frames:
                colors = get_per_frame_vals(color_palette)
            colors = colors[mask]

        # Label Color map
        if isinstance(color_palette, dict):
            colors = np.array([color_palette[int(label)] for label in labels])
            colors[labels < 0] = 0
            colors = colors[:, :3]
            # colors = colors[:3]
            # colors = colors[mask]

        # image_paths = [image_paths[i] for i in range(len(image_paths)) if mask[i]]
        # image_paths = image_paths.to(list)

        # to visualise the temporal progression in the embeddings
        # for temp, label in enumerate(labels):
        #     labels[temp] = (int(temp) % 20)
    else:
        if not isinstance(color_palette, np.ndarray):
            colors = np.array([color_palette[int(label)] for label in labels])
            colors[labels < 0] = 0
            colors = colors[:, :3]

    # max_label = labels.max()
    # color_palette = generate_distinct_colors(max_label + 1)
    
    # Set up the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(embeddings)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the pcd
    if save_pcd is not None:
        if not save_pcd.endswith('.pcd'):
            save_pcd += '.pcd'
        print(f"Saving the pcd to {save_pcd}")
        o3d.io.write_point_cloud(save_pcd, pcd)
    
    while True:
        pick_points(pcd, labels, label_names, image_paths, image_times, label_drug_dict, decoder)

def plot_confusion_matrix(cm,
                          label_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          k=100,
                          vmin=None,
                          vmax=None):
    cm_unnorm = cm.copy()

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(20, 20))
    # plt.figure(figsize=(3, 3), dpi=300)

    tickmarks = np.arange(cm.shape[0])
    plt.xticks(tickmarks, label_names, rotation=90)
    plt.yticks(tickmarks, label_names)
    
    # Set no ticks nor labels
    # plt.xticks([])
    # plt.yticks([])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = plt.imshow(cm, cmap=cmap, interpolation='nearest',
                    vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im)

    thresh = cm.max() / 1.5 if cm.max() > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm_unnorm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_cm(gt_labels, pred_labels, label_drug_dict,
            verbose=True, make_plot=True, vmin=None, vmax=None):
    cm = confusion_matrix(gt_labels, pred_labels,
                          labels=sorted(list(label_drug_dict.keys())))

    if verbose:
        print("per class accuracy Top-1")
        for i in range(cm.shape[0]):
            acc = cm[i, i] * 100.0 / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0.0
            print(f"{label_drug_dict[i]}: {acc:.2f}%")

    if make_plot:
        plot_confusion_matrix(
            cm,
            list(label_drug_dict.values()),
            title='Confusion matrix',
            cmap=None,
            normalize=True,
            k=100,
            vmin=vmin,
            vmax=vmax
        )
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
