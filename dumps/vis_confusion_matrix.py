import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    cm = np.load('/home/dhruvagarwal/projects/MitoSpace4D/confusion_matrix_2.npy')
    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='viridis')

    # We want to show all ticks...
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(np.arange(cm.shape[1]))
    ax.set_yticklabels(np.arange(cm.shape[0]))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()