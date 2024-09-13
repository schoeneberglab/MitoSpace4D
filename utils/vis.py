import copy
import open3d as o3d
import matplotlib.patches as mpatches
import os.path as osp
import random
import colorsys

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

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


def pick_points(pcd, images, labels, label_names):
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
    imgs = []
    for idx in idxs:
        imgs.append(np.array(images[idx]))

    if len(idxs) == 0:
        return

    colors = np.array(pcd.colors)[idxs]

    f, axarr = plt.subplots(2, len(imgs), figsize=(20, 20))
    vals = []
    for i in range(len(imgs)):
        mito_idx = 1 if imgs[i].shape[-1] > 1 else 0
        tmrm_idx = 0
        axarr[0, i].imshow(imgs[i][:, :, tmrm_idx], vmin=0., vmax=1., cmap=plt.cm.hot)
        axarr[1, i].imshow(imgs[i][:, :, mito_idx], vmin=0., vmax=1., cmap=plt.cm.viridis)

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
        print(labels[idx] % 27)
        print(label_names[int(labels[idx]) % 27])
        print("")
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=label_names[int(labels[idxs[i]]) % 27])) for i in
               range(len(idxs))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    print("")
    return vis.get_picked_points()


def make_mitospace(embedding_dir, pick_labels=None):
    EMBEDDING_PATH = osp.join(embedding_dir, 'embeddings.npy')
    LABEL_PATH = osp.join(embedding_dir, 'labels.npy')
    # IMAGE_PATH = osp.join(embedding_dir, 'images.npy')
    LABEL_NAME_PATH = osp.join(embedding_dir, 'label_names.npy')

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)
    label_names = np.load(LABEL_NAME_PATH)
    # images = np.load(IMAGE_PATH)

    # print(embeddings.shape, labels.shape, images.shape)

    print(np.unique(labels))

    # images = images.transpose(0, 2, 3, 1)

    max_label = labels.max()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(embeddings)
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # print(max_label)

    colors = generate_distinct_colors(max_label + 1)

    # assign colors to the all the points by indexing the values as the labels
    colors = np.array([colors[int(label)] for label in labels])

    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = np.array([0, 0, 0])

    # Get min and max points of the AABB
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    # Calculate diameter along each axis
    diameter_x = np.abs(max_bound[0] - min_bound[0])
    diameter_y = np.abs(max_bound[1] - min_bound[1])
    diameter_z = np.abs(max_bound[2] - min_bound[2])

    print("Diameter along X-axis:", diameter_x)
    print("Diameter along Y-axis:", diameter_y)
    print("Diameter along Z-axis:", diameter_z)

    o3d.visualization.draw_geometries([pcd, aabb])

    # if pick_labels is not None:
    #     idxs = np.concatenate([np.where(labels == x)[0] for x in pick_labels], axis=0)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(embeddings[idxs])
    #     pcd.colors = o3d.utility.Vector3dVector(colors[idxs, :3])
    #     images = images[idxs]
    #     labels = labels[idxs]

    while True:
        pick_points(pcd, images, labels, label_names)


def plot_confusion_matrix(cm,
                          label_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          k=100):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(20, 20))
    if label_names is not None:
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()


def plot_cm(gt_labels, pred_labels, label_drug_dict, verbose=True, make_plot=True):
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(label_drug_dict.keys()))

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
