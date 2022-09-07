import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import torch
import librosa


def to_mel(hz):
        """Converts frequency in Hz to the mel scale.
        """
        return 2595 * np.log10(1 + hz / 700)


def to_hz(mel):
        """Converts frequency in the mel scale to Hz.
        """
        return 700 * (10 ** (mel / 2595) - 1)


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5, label=None):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='None', ecolor='k', label=label)

    return artists


def main(ckpt_dir, save_dir, filter_type):
    ### sinc
    checkpoint = torch.load(ckpt_dir, 
                            map_location=lambda storage, loc: storage)
    sinc_low_hz = checkpoint['state_dict']['trans.low_hz_']
    sinc_band_hz = checkpoint['state_dict']['trans.band_hz_'] # 80*1 /128*1

    
    ### mel
    min_low_hz = 5
    min_band_hz = 5
    sample_rate = 16000

    if filter_type == 'mel':
        out_channels = 80
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = torch.linspace(
            to_mel(min_low_hz),
            to_mel(high_hz),
            out_channels + 1,
        )
        hz = to_hz(mel)
        low_hz = hz[:-1].unsqueeze(1)
        band_hz = (hz[1:] - hz[:-1]).unsqueeze(1)

    elif filter_type == 'midi':
        out_channels = 122
        midi_critical_freq = [librosa.midi_to_hz(x) for x in range(out_channels)]
        if midi_critical_freq[-1] > sample_rate / 2:
            high_hz = midi_critical_freq[-1] + min_band_hz
        else:
            high_hz = sample_rate / 2
        midi_critical_freq = [min_low_hz] + midi_critical_freq + [high_hz]
        hz = torch.Tensor(midi_critical_freq).squeeze() - min_low_hz
        low_hz = hz[:-2].unsqueeze(1)
        band_hz = (hz[2:] - hz[:-2]).unsqueeze(1)


    def get_params(low_hz, band_hz):
        low = min_low_hz + torch.abs(low_hz)
        # Setting minimum band and minimum freq
        high = torch.clamp(
            low + min_band_hz + torch.abs(band_hz),
            min_low_hz,
            sample_rate / 2,
        )
        band = (high - low)[:, 0]
        return low, band    


    ###########################################################################
    ### Plot
    fig, ax = plt.subplots(1)
    plt.subplots_adjust(left=0.15)
    x = np.arange(out_channels)

    # Sinc
    y = np.array(sinc_low_hz.squeeze()) 
    xerr = np.vstack([np.zeros(out_channels), np.ones(out_channels)])
    yerr = np.vstack([np.zeros(out_channels), np.array(sinc_band_hz.squeeze())])
    _ = make_error_boxes(ax, x, y, xerr, yerr)

    # Filter
    y = np.array(low_hz.squeeze())
    xerr = np.vstack([np.zeros(out_channels), np.ones(out_channels)])
    yerr = np.vstack([np.zeros(out_channels), np.array(band_hz.squeeze())])
    _ = make_error_boxes(ax, x, y, xerr, yerr, facecolor='g')
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Frequency')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.legend()
    # plt.rc('font', size=30)
    plt.show()
    # fig.savefig(os.path.join(save_dir, 'midi.png'), dpi=600)


if __name__ == '__main__':
    ckpt_dir = sys.argv[1]
    save_dir = sys.argv[2]
    filter_type = sys.argv[3]
    main(ckpt_dir, save_dir, filter_type)
