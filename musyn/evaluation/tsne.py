import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import plot_det_curve
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def plot_TSNE(feat, labels, save_dir, prefix=None, writer=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, learning_rate=50.0, n_iter=300)
    feat_embedded = tsne.fit_transform(feat)
    # __import__('ipdb').set_trace()
    for label, pre in zip(labels, prefix):

        font = {'size': 25}
        matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(1,1,1)
        
        # label encode
        # __import__('ipdb').set_trace()
        labels_encoder = LabelEncoder()
        labels_encoder.fit(label)
        label = labels_encoder.transform(label).reshape((-1, 1))
        
        scatter = ax.scatter(x=feat_embedded[:,0],
                             y=feat_embedded[:,1],
                             c=label,
                             cmap='rainbow')   # others: hsv        
        ax.legend(*scatter.legend_elements(), title=pre)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 't-SNE-{}.pdf'.format(pre)), dpi=600)
        if writer is not None:
            writer.add_embedding(feat, label, global_step=0, tag=pre)



def plot_DET(x, y, eer, save_dir):
    fig = plt.figure()
    plt.title('DET')
    plt.xlabel('False Positives')
    plt.ylabel('True Positive rate')
    plt.plot(x, y, label = 'EER: ' + str(eer))
    fig.savefig(os.path.join(save_dir, 'DET.png'), dpi=600)


if __name__ == '__main__':
    feat = np.load(sys.argv[1])
    # __import__('ipdb').set_trace()
    labels = glob.glob(os.path.join(sys.argv[2], 'labels_*.npy'))
    labels = [np.load(item) for item in labels]
    save_dir = sys.argv[3]
    # prefix = ['instr_fml', 'instr', 'MIDI pitch number', 'MIDI velocity value', 'instr_src']
    prefix = ['dynamics', 'instr', 'style']
    assert len(labels) == len(prefix)
    plot_TSNE(feat, labels, save_dir, prefix)