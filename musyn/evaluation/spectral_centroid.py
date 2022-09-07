import os
import sys
import glob
import json
import librosa
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from collections import OrderedDict
from scipy.sparse import base
from sklearn.manifold import TSNE
from sklearn.metrics import plot_det_curve
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm


INSTR_FML = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
MARKER = ["s" , "P" , "D" , "o" , "+" , "<", ">", "h", "2", "3", "4"]

def plot_TSNE(feat, labels, save_dir, legend_dict, prefix=None, writer=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, learning_rate=50.0, n_iter=300)
    feat_embedded = tsne.fit_transform(feat)
    feat_embedded = np.load(os.path.join(save_dir, 'feats_embedded.npy'))

    for label, pre in zip(labels, prefix):
        font = {'size': 25}
        matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(1,1,1)
        
        # get label
        label_set = list(set(label))
        label_set.sort()
        if len(MARKER) > len(label_set):
            marker = MARKER[:len(label_set)]
        else:
            raise ValueError
        for label_item, marker_item in zip(label_set, marker):
            index = np.where(label == label_item)
            if marker_item == 'o':
                scatter = ax.scatter(x=feat_embedded[index, 0],
                                    y=feat_embedded[index, 1],
                                    label=label_item,
                                    marker=marker_item,
                                    cmap='rainbow',
                                    facecolors = 'none',
                                    edgecolors = 'r',
                                    s=100)   # others: hsv
            else:
                scatter = ax.scatter(x=feat_embedded[index, 0],
                                    y=feat_embedded[index, 1],
                                    label=label_item,
                                    marker=marker_item,
                                    cmap='rainbow',
                                    s=100)   # others: hsv
        ax.legend(title=pre, bbox_to_anchor=(1.05, 1))

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 't-SNE-{}.pdf'.format(pre)), dpi=600)
        if writer is not None:
            writer.add_embedding(feat, label, global_step=0, tag=pre)
    return feat_embedded


def plot_nsynth():
    data_dir = 'data/nsynth/nsynth-test/audio'
    json_file_dir = 'data/nsynth/nsynth-test/examples.json'
    # feat_file_dir = 'log/nsynth_sinc_midi_aug_asm_s100_instr/feats_A_instr.npy'
    feat_file_dir = 'feats_A.npy'
    save_dir = 'spectral/nsynth_instr'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(json_file_dir, 'r') as f:
        json_file = json.load(f)
    dataset = OrderedDict(json_file)
    
    feat = np.load(feat_file_dir)
    instr_fml = []
    instr = []
    pitch = []
    instr_fml_id2str_dict = {}
    instr_id2str_dict = {}

    i = 0
    for k, v in dataset.items():
        v.update({'feat': feat[0,:]})
        i += 1
        dataset[k] = v
        instr_fml.append(v['instrument_family_str'])
        instr.append(v['instrument_str'])
        pitch.append(v['pitch'])
        instr_fml_id2str_dict[v['instrument_family']] = v['instrument_family_str']
        instr_id2str_dict[v['instrument']] = v['instrument_str']
    instr_fml = np.hstack(instr_fml)
    instr = np.hstack(instr)
    pitch = np.hstack(pitch)

    ### filter data points
    dataset_fil = OrderedDict()
    index_fil = []
    j = 0
    for k, v in dataset.items():
        source_item = v['instrument_source_str']
        pitch_item = v['pitch']
        if source_item == 'acoustic' and pitch_item > 47 and pitch_item < 58:
            dataset_fil[k] = v
            index_fil.append(j)
        j += 1
    index_fil = np.hstack(index_fil)
    feat_fil = feat[index_fil,:]
    instr_fml_fil = instr_fml[index_fil]
    instr_fil = instr[index_fil]
    pitch_fil = pitch[index_fil]

    ### plot filtered data
    kwargs = {
        'feat': feat_fil,
        'labels': [instr_fml_fil],
        'save_dir': save_dir,
        'prefix': ['Instrument Family'],
        'legend_dict': {'instr_fml': instr_fml_id2str_dict, 
                        'instr': instr_id2str_dict},
    }
    feat_embedded = plot_TSNE(**kwargs)
    np.save(os.path.join(save_dir, 'feats_embedded.npy'), feat_embedded)

    ### compute spectral
    cent_mean_list = []
    for k, v in dataset_fil.items():
        audio_file_dir = os.path.join(data_dir, k + '.wav')
        y, sr = librosa.load(audio_file_dir)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        v.update({'cent_mean': cent_mean})
        dataset_fil[k] = v
        cent_mean_list.append(cent_mean)
    cent_mean_list = np.hstack(cent_mean_list)

    np.save('spectral/cent_mean.npy', cent_mean_list)
    np.save('spectral/feat_embedded.npy', feat_embedded)

    ### interpolation and draw the spectral figure
    X = np.linspace(min(feat_embedded[:,0]), max(feat_embedded[:,0]))
    Y = np.linspace(min(feat_embedded[:,1]), max(feat_embedded[:,1]))
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(list(zip(feat_embedded[:,0], feat_embedded[:,1])), cent_mean_list)
    Z = interp(X, Y)
    fig = plt.figure(figsize=(12,10))
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(feat_embedded[:,0], feat_embedded[:,1], "ok", label="input point")
    plt.legend()
    plt.colorbar()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'spectral_centriod.pdf'), dpi=600)


def plot_rwc():
    feat_brass_dir = 'data/RWC_MDB_I/probing_data/feat_midi/brass'
    feat_strings_dir = 'data/RWC_MDB_I/probing_data/feat_midi/strings'
    feat_ww_dir = 'data/RWC_MDB_I/probing_data/feat_midi/woodwinds'
    data_dir = 'data/RWC_MDB_I/test/audio'
    save_dir = 'spectral/rwc'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feat_dir_list = []
    for item in [feat_brass_dir]:
        feat_dir_list.extend(glob.glob(os.path.join(item, '*.npy')))
    
    feat = []
    instr = []
    for item in feat_dir_list:
        basename = os.path.basename(item)[:-4]
        feat.append(np.load(item))
        instr.append(int(basename[:3]))
    feat = np.vstack(feat)
    instr = np.hstack(instr)

    ### plot filtered data
    kwargs = {
        'feat': feat,
        'labels': [instr],
        'save_dir': save_dir,
        'prefix': ['rwc_brass_instr'],
        'legend_dict': None,
    }
    feat_embedded = plot_TSNE(**kwargs)

    ### compute spectral
    cent_mean_list = []
    for item in feat_dir_list:
        basename = os.path.basename(item)[:-4]
        audio_file_dir = os.path.join(data_dir, basename + '.wav')
        y, sr = librosa.load(audio_file_dir)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_mean_list.append(cent_mean)
    cent_mean_list = np.hstack(cent_mean_list)

    np.save('spectral/rwc/brass_cent_mean.npy', cent_mean_list)
    np.save('spectral/rwc/brass_feat_embedded.npy', feat_embedded)

    ### interpolation and draw the spectral figure
    X = np.linspace(min(feat_embedded[:,0]), max(feat_embedded[:,0]))
    Y = np.linspace(min(feat_embedded[:,1]), max(feat_embedded[:,1]))
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(list(zip(feat_embedded[:,0], feat_embedded[:,1])), cent_mean_list)
    Z = interp(X, Y)
    fig = plt.figure(figsize=(12,10))
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(feat_embedded[:,0], feat_embedded[:,1], "ok", label="input point")
    plt.legend()
    plt.colorbar()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'spectral_centriod.pdf'), dpi=600)


if __name__ == '__main__':
    plot_nsynth()
    # plot_rwc()




