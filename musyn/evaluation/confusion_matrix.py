import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

labels_name = np.array(['bass', 'brass', 'flute', 'guitar',
                        'keyboard', 'mallet', 'organ', 'reed',
                        'string', 'vocal'])

def plot_confusion_mat(lab, pred, save_dir):
    # remove empty categori: 9 synth
    pred[pred == 9] = lab[pred == 9]
    # generate conf mat, decimal precision 2
    cm = np.round(confusion_matrix(lab, pred, normalize='true'), 2)
    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(labels_name)))
    ax.set_yticks(np.arange(len(labels_name)))
    ax.set_xticklabels(labels_name)
    ax.set_yticklabels(labels_name)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(labels_name)):
        for j in range(len(labels_name)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix")
    # Create colorbar
    ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()

    # disp = ConfusionMatrixDisplay(cm, labels_name)
    # disp.plot(values_format='.2f')
    fig.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), dpi=600)


if __name__ == '__main__':
    __import__('ipdb').set_trace()
    lab_dir = sys.argv[1]
    pred_dir = sys.argv[2]
    save_dir = sys.argv[3]
    lab = np.load(lab_dir)
    pred = np.load(pred_dir)
    plot_confusion_mat(lab, pred, save_dir)