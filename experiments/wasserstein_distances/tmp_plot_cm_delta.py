import numpy as np
import matplotlib.pyplot as plt


pre = np.load("/home/earkfeld/Projects/MitoSpace4D/lstm_top1_cm.npy")
post = np.load("/home/earkfeld/Projects/MitoSpace4D/resnet_top1_cm.npy")
label_names = np.load("/home/earkfeld/Projects/MitoSpace4D/label_names_cm.npy")

delta = post - pre

def plot_confusion_matrix(cm,
                          label_names,
                          title='Delta Matrix (Top1; PostLSTM - PreLSTM)',
                          cmap=None,
                          normalize=False,
                          vmin=None,
                          vmax=None):
    cm_unnorm = cm.copy()

    min_val = np.min(cm)
    max_val = np.max(cm)
    abs_max = max(abs(min_val), abs(max_val))
    vmin = -abs_max
    vmax = abs_max

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    # plt.figure(figsize=(20, 20))
    plt.figure(figsize=(10, 10), constrained_layout=True)
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

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(delta, label_names=label_names, cmap="coolwarm")