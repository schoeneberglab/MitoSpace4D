import colorsys
import random
from random import shuffle
import yaml
import cv2
import numpy as np
import torch
from typing import List, Union
import matplotlib.pyplot as plt
import os
import os.path as osp
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster

from skimage.restoration import denoise_tv_bregman
from sklearn.preprocessing import MinMaxScaler


def get_valid_palette(num_colors, max_attempts=50):
    """Generates a color palette and ensures all colors are visible (not too light)."""
    for _ in range(max_attempts):  # Retry if light colors exist
        # base_palette = sns.color_palette("husl", num_colors)  # Generate colors
        base_palette = [(0.698, 0.875, 0.859), (0.820, 0.769, 0.914), (1.000, 0.800, 0.737),
                         (0.702, 0.898, 0.988), (1.000, 0.976, 0.769), (1.0, 0.0, 1.0)]

        # display the colors
        plt.figure(figsize=(12, 3))
        for i, color in enumerate(base_palette):
            plt.scatter(i, 0, color=color, s=200)
        plt.axis("off")
        plt.show()

        valid_palette = []

        for color in base_palette:
            h, l, s = colorsys.rgb_to_hls(*color)  # Convert to HLS to check lightness
            # if l < 0.85:  # Avoid very light colors
            #     valid_palette.append(color)
            valid_palette.append(color)

        # valid_palette = [(0.698, 0.875, 0.859), (0.820, 0.769, 0.914), (1.000, 0.800, 0.737),
        #                  (0.702, 0.898, 0.988), (1.000, 0.976, 0.769), (0.973, 0.733, 0.816)]

        if len(valid_palette) == num_colors:
            return valid_palette  # Return only if all colors are valid

    raise ValueError("Could not generate a valid palette without light colors.")  # Fail-safe


def get_phenotypic_colors(similarity_matrix, num_clusters):
    # Load the similarity matrix (Replace with actual data)
    np.fill_diagonal(similarity_matrix, 1)  # Set diagonal to 1 for perfect self-similarity

    # Normalize the similarity matrix
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(similarity_matrix)

    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(similarity_matrix, annot=True, ax=ax)
    plt.show()

    # Perform hierarchical clustering
    linkage_matrix = linkage(normalized_matrix, method='ward')
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # Generate base colors for each cluster
    # base_palette = sns.color_palette("tab20", num_clusters)  # Base colors
    base_palette = get_valid_palette(num_clusters, max_attempts=10)

    # Function to generate shades of a base color
    def get_shaded_colors(base_color, num_shades, min_lightness=0.25, max_lightness=0.45):
        """Generates `num_shades` variations of `base_color`, ensuring they are not too light."""
        h, l, s = colorsys.rgb_to_hls(*base_color)  # Convert to HLS

        # Generate shades by varying lightness within a safe range
        lightness_values = np.linspace(min_lightness, max_lightness, num_shades)
        shaded_colors = []

        for lightness in lightness_values:
            shaded_rgb = colorsys.hls_to_rgb(h, lightness, s)  # Convert back to RGB
            shaded_colors.append(shaded_rgb)

        return shaded_colors

    # Assign shades to classes
    class_colors = {}
    cluster_counts = {i: 0 for i in range(1, num_clusters + 1)}  # Track how many classes per cluster
    for i, cluster_id in enumerate(clusters):
        num_in_cluster = sum(clusters == cluster_id)  # Count classes in the same cluster
        shade_index = cluster_counts[cluster_id]  # Assign a unique shade index
        shaded_colors = get_shaded_colors(base_palette[cluster_id - 1], num_in_cluster)
        # plot the colors
        # plt.figure(figsize=(12, 3))
        # for i, color in enumerate(shaded_colors):
        #     plt.scatter(i, 0, color=color, s=200)
        # plt.axis("off")
        # plt.show()
        class_colors[f"Class {i}"] = shaded_colors[shade_index]  # Assign a color
        cluster_counts[cluster_id] += 1  # Increment count for next shade

    ################################# assign same color to same cluster #########################################
    # class_colors = {}
    # cluster_counts = {i: 0 for i in range(1, num_clusters + 1)}  # Track how many classes per cluster
    # for i, cluster_id in enumerate(clusters):
    #     shade_index = cluster_counts[cluster_id]  # Assign a unique shade index
    #     shaded_colors = get_shaded_colors(base_palette[cluster_id - 1], 1)
    #     # plot the colors
    #     # plt.figure(figsize=(12, 3))
    #     # for i, color in enumerate(shaded_colors):
    #     #     plt.scatter(i, 0, color=color, s=200)
    #     # plt.axis("off")
    #     # plt.show()
    #     class_colors[f"Class {i}"] = shaded_colors[shade_index]  # Assign a color

    ###################################################### comment the code block to have shades ################

    # Print assigned colors
    to_print_dict = {0: "20240729 control 0", 1: "20240730 p110 1", 2: "20240731 myls22 2",
                     3: "20240801 mfi8 3", 4: "20240731 tbhp 4", 5: "20240805 h2o2 5",
                     6: "20240806 mitoq 6", 7: "20240807 resveratrol 7", 8: "20240808 lonidmaine 8",
                     9: "20240809 oligomycin 9", 10: "20240813 dnp 10", 11: "20240814 valinomycin 11",
                     12: "20240815 cccp 12", 13: "20240816 mitomycinc 13", 14: "20240820 cytochalasind 14",
                     15: "20240821 lantrunculinb 15", 16: "20240823 mdivi1 16", 17: "20240826 nocodazole 17",
                     18: "20240830 colchicine 18", 19: "20240903 antimycina 19", 20: "20240904 tiron 20",
                     21: "20240905 cisplatin 21", 22: "20240910 rotenone 22", 23: "20240911 nigericin 23",
                     24: "20240912 azide 24", 25: "20240913 paraquat 25", 26: "20240917 metformin 26"}

    for cls, color in class_colors.items():
        str_color = str(color).replace('(', '').replace(')', '').replace(',', '')
        print(f"{to_print_dict[int(cls.split(' ')[-1])]} {str_color}")

    # Optional: Visualize the color assignments
    plt.figure(figsize=(12, 3))
    for i, (cls, color) in enumerate(class_colors.items()):
        plt.scatter(i, 0, color=color, s=200, label=cls)
    plt.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.2))
    plt.axis("off")

    plt.show()


def get_drug_labels(fpath):
    drug_labels_dict = {}
    label_drug_dict = {}
    with open(fpath, 'r') as f:
        for line in f:
            drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    return drug_labels_dict, label_drug_dict


def denoise(img, weight=3):
    """

    Args:
        weight: filter strength, lower removes more noise, but also removes details
        img: grayscale image

    Returns:
        denoised image
    """
    denoised_img = denoise_tv_bregman(img, weight=weight)
    return denoised_img


def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def minus_one_to_one_normalization(x: Union[List[torch.Tensor], torch.Tensor]) -> Union[
    List[torch.Tensor], torch.Tensor]:
    if isinstance(x, list):
        return [2 * var - 1 for var in x]
    return 2 * x - 1


def idxs_to_keep(x: Union[List[np.ndarray], np.ndarray], idxs: List[int] = None) -> Union[List[np.ndarray], np.ndarray]:
    if idxs is None:
        return x

    if isinstance(x, list):
        if len(idxs) == 1:
            return [var[idxs[0]: idxs[0] + 1] for var in x]

        else:
            return [var[idxs] for var in x]
    else:
        if len(idxs) == 1:
            return x[idxs[0]: idxs[0] + 1]
        else:
            return x[idxs]


def topKfrequent(nums, weights, k, weighted=False):
    d = dict()

    for i, n in enumerate(nums):
        if weighted:
            d[n] = d.setdefault(n, 0) + weights[i]
        else:
            d[n] = d.setdefault(n, 0) + 1

    sortedNumsKeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)
    return sortedNumsKeys[:k]


def agressive_sigmoid(x, alpha):
    return -1 + 2 * (1 / (1 + np.exp(-alpha * x)))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          checkpoint_path=None,
                          top_n=None,
                          k=100):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            # plt.text(j, i, "{:0.4f}".format(cm[i, j]),
            plt.text(j, i, "",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            # plt.text(j, i, "{:,}".format(cm[i, j]),
            plt.text(j, i, "",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(f"{checkpoint_path}/confusion_matrix_top_{top_n}_test_func_1_1", dpi=400)


def get_patches_around_nucleus(image, patch_size=128, contour_area=100, only_center=True):
    """calculates the centroids of the nuclei and returns the patches around them as a list of
    (x_min, y_min, x_max, y_max).
    Expected image shape: (h, w) and the pixel values should be in the range [0, 1]

    only_center: if True, only the patches around the center of the nuclei are returned
    otherwise, nearby areas around the nuclei are also returned
    """

    image = image * 255.
    h, w = image.shape

    blurred = cv2.GaussianBlur(image, (5, 5), 0).astype('uint8')
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour_img = cv2.drawContours(image=blurred, contours=contours, contourIdx=-1, color=(255, 255, 255),
    #                                thickness=2, lineType=cv2.LINE_AA)

    # contour_areas = np.array([cv2.contourArea(contour) for contour in contours])

    centroids = []

    for contour in contours:
        if cv2.contourArea(contour) > contour_area:
            # plot the selected contour

            contour_img = cv2.drawContours(image=blurred, contours=[contour], contourIdx=-1, color=(255, 255, 255),
                                           thickness=2, lineType=cv2.LINE_AA)
            # plt.imshow(contour_img)
            # plt.show()

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))

    # get patches around the centroids
    patches = []
    dx, dy = patch_size // 8, patch_size // 8

    if only_center:
        delta_x = [0]
        delta_y = [0]

    else:
        delta_x = [0, -dx, -2 * dx, dx, 2 * dx]
        delta_y = [0, -dy, -2 * dy, dy, 2 * dy]

    deltas = []
    for i in delta_x:
        for j in delta_y:
            deltas.append((i, j))

    for centroid in centroids:
        x, y = centroid

        for delta in deltas:
            x_min = x + delta[0] - patch_size // 2
            x_max = x + delta[0] + patch_size // 2
            y_min = y + delta[1] - patch_size // 2
            y_max = y + delta[1] + patch_size // 2

            if x_min < 0 or x_max >= w or y_min < 0 or y_max >= h:
                continue

            patches.append((x_min, y_min, x_max, y_max))

    return patches


def get_all_patches(image, patch_size=1024, stride=512):
    """
    returns a list of patches from the image
    :param image: input image
    :param patch_size: size of the patch
    :param stride: stride between the patches
    :return: list of patches
    """

    min_pixel_value = image.min()

    # get the dimensions of the image
    h, w = image.shape

    # get the number of patches in the x and y direction
    num_patches_x = (w - patch_size) // stride + 1
    num_patches_y = (h - patch_size) // stride + 1

    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x_min = i * stride
            x_max = x_min + patch_size
            y_min = j * stride
            y_max = y_min + patch_size

            # if the patch is at the edge of the image, then skip it
            if x_max >= w or y_max >= h or x_min < 0 or y_min < 0:
                continue

            # if most of the area of the patch is empty, then skip it (80% empty is the threshold here)
            patch = image[y_min:y_max, x_min:x_max]
            num_empty_pixels = (patch <= min_pixel_value).sum()

            if num_empty_pixels >= 0.8 * patch.size:
                continue

            patch_tuple = (x_min, y_min, x_max, y_max)
            patches.append(patch_tuple)

    return patches


def normalize(img):
    """normalize image between 0-1; channel-wise"""
    min_ = np.min(img, axis=(0, 1))[None, None]
    max_ = np.max(img, axis=(0, 1))[None, None]
    return (img - min_) / (max_ - min_)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def increase_contrast(gray_images):
    """
    Increase the contrast of an image using histogram equalization.

    Parameters:
    - image: NumPy array representing the image

    Returns:
    - enhanced_image: NumPy array representing the enhanced image
    """

    if not isinstance(gray_images, torch.Tensor):
        image = (gray_images - gray_images.min()) / (gray_images.max() - gray_images.min())
        equalized_image = cv2.equalizeHist((image * 255.).astype(np.uint8))
        return equalized_image

    # Apply histogram equalization to enhance contrast
    images = gray_images.numpy()
    images = [((x - x.min()) / (x.max() - x.min())) for x in images]

    equalized_image = np.stack([cv2.equalizeHist((x * 255.).astype(np.uint8)) for x in images])
    equalized_image = torch.from_numpy(equalized_image).float()
    equalized_image /= 255.
    equalized_image = equalized_image * 2 - 1

    return equalized_image


def get_fpaths(root_dir, seed=1123):
    drug_labels = {}
    with open('/u/earkfeld/MitoSpace4D/extraction_utils/drugs_to_labels.txt', 'r') as f:
        drugs_to_labels = f.readlines()
        for line in drugs_to_labels:
            folder, drug, label = line.split()
            drug_labels[folder] = {'drug': drug, 'label': int(label)}

    drug_folders = sorted([file for file in os.listdir(osp.join(root_dir, 'processed_data'))])

    all_filenames = []

    for drug_folder in drug_folders:
        filenames = sorted([file for file in os.listdir(osp.join(root_dir, 'processed_data', drug_folder))])
        filenames = [osp.join(root_dir, 'processed_data', drug_folder, file) for file in filenames]

        all_filenames.extend(filenames)

    random.seed(seed)
    shuffle(all_filenames)

    return all_filenames
