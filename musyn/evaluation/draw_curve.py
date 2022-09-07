import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 15}
matplotlib.rc('font', **font)

def draw_loss_curve(tag_list, data_list, fmt_list, xlabel, ylabel, title):

    fig, ax = plt.subplots()

    x_list = [np.array(item['Step'])[:500] for item in data_list]
    y_list = [np.array(item['Value'])[:500] for item in data_list]
    for label, x, y, fmt in zip(tag_list, x_list, y_list, fmt_list):
        
        if 'midi' in label and 'scratch' in label:
            x = x[:300]
            y = y[:300]
        
        if 'mel_scratch' in label or 'mel_pretrained' in label or 'meltrans_pretrained' in label:
            x *= 2
        
        if 'miditrans_pretrained' in label:
            x *= 2
            x = x[:250]
            y = y[:250]

        line, = ax.plot(x, y, fmt, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    plt.subplots_adjust(bottom=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(title+'.pdf', dpi=600)



def draw_f1_curve(tag_list, data_list, fmt_list, xlabel, ylabel, title):

    fig, ax = plt.subplots()

    x_list = [np.array(item['Step'])[:30] for item in data_list]
    y_list = [np.array(item['Value'])[:30] for item in data_list]
    for label, x, y, fmt in zip(tag_list, x_list, y_list, fmt_list):
        line, = ax.plot(x, y, fmt, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    plt.subplots_adjust(bottom=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(title+'.pdf', dpi=600)



def main():
    # loss
    tag_list = ['mel_scratch', 'mel_pretrained',
                'meltrans_scratch', 'meltrans_pretrained',
                'miditrans_scratch', 'miditrans_pretrained']
    file_list = ['rwc_fil_mel_asm_s1000',
                 'rwc_fil_mel_asm_s1000_pretrained',
                 'rwc_fil_sinc_mel_asm_s1000',
                 'rwc_fil_sinc_mel_asm_s1000_pretrained',
                 'rwc_fil_sinc_midi_asm_s1000',
                 'rwc_fil_sinc_midi_asm_s1000_pretrained']
    fmt_list = ['-b', '-.b', '-c', '-.c', '-m', '-.m']
    file_list = [os.path.join('log/exp2_rwc_log', item, 'csv/run-.-tag-Loss_train.csv') for item in file_list]
    assert len(tag_list) == len(file_list)
    data_list = [pd.read_csv(f) for f in file_list]
    draw_loss_curve(tag_list, data_list, fmt_list, 'Step', 'Loss', 'rwc_loss')
    # draw_f1_curve(tag_list, data_list, fmt_list, 'Epoch', 'F1-Macro', 'rwc_f1_mac')



if __name__ == '__main__':
    main()