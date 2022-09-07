#!/usr/bin/env python

from matplotlib import markers, pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import sys
import os
from statsmodels.stats.contingency_tables import mcnemar
from statistical_significance import get_eer, compute_HTER_independent, reject_null_holm_bonferroni


def mk_groups(data):
    try:
        newdata = data.items()
        if '100' in data.keys():
            return
    except:
        return
            
    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos + .1, xpos], [ypos, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def label_group_bar(ax, data):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)
    __import__('ipdb').set_trace()
    # ax.barh(xticks, y, align='center')
    # ax.set_yticks(xticks)
    # ax.set_yticklabels(x)
    # ax.set_ylim(.5, ly + .5)
    # ax.xaxis.grid(True)

    scale = 1. / ly
    for pos in range(ly + 1):
        add_line(ax, -.1, pos * scale)
    
    xpos = -.2
    while groups:
        group = groups.pop()
        pos = 0.
        for label, rpos in group:
            lypos = (pos + .5 * rpos) * scale
            ax.text(xpos, lypos - 0.09, label, ha='left', rotation='vertical' , transform=ax.transAxes)
            add_line(ax, xpos, pos * scale)
            pos += rpos
        add_line(ax, pos * scale, xpos)
        xpos -= .1


def parse_data(data_struct, dataset):
    try:
        prefix = data_struct['prefix']
        dataset[prefix] = {
            'tar_data': np.load(os.path.join(
                'log', prefix, 'tar_scores.npy')),
            'nontar_data': np.load(os.path.join(
                'log', prefix, 'nontar_scores.npy')),
            # 'preds_B': np.load(os.path.join(
            #     'log', prefix, 'preds_B.npy'))
        }
        return 
    except:
        for k, v in data_struct.items():
            parse_data(v, dataset)


def generate_data(data_struct, mode='eer'):
    """
    generate the statistic significance test matrix: n*n, where n is the number of systems
    Input:
    - data_struct: hierarchy dict
    - mode: string, 'eer'/'mc'
    """

    dataset = dict()
    parse_data(data_struct, dataset)
    system_list = list(dataset.keys())
    sys_len = len(system_list)
    sta_sig_ma = np.ones((sys_len, sys_len))
    
    labels_B = np.load('labels_B.npy')
    
    for i in range(sys_len):
        for j in range(sys_len):
            if mode == 'eer':
                tar_i = dataset[system_list[i]]['tar_data']
                nontar_i = dataset[system_list[i]]['nontar_data']
                tar_j = dataset[system_list[j]]['tar_data']
                nontar_j = dataset[system_list[j]]['nontar_data']
                eer_i, _ = get_eer(tar_i, nontar_i)
                eer_j, _ = get_eer(tar_j, nontar_j)
                z_value = compute_HTER_independent(eer_i, eer_i, eer_j, eer_j, 
                                                tar_i.shape[0], nontar_i.shape[0])
                significance_level = 0.05
                test_output_mat = reject_null_holm_bonferroni(np.array([z_value]), significance_level)
                sta_sig_ma[sys_len-i-1][j] -= test_output_mat
            elif mode == 'mc':
                preds_i = dataset[system_list[i]]['preds_B']
                preds_j = dataset[system_list[j]]['preds_B']
                corr_i = np.where(preds_i==labels_B)[0]
                corr_j = np.where(preds_j==labels_B)[0]
                num_ir_jr = np.intersect1d(corr_i, corr_j).shape[0]
                num_ir_jw = np.setdiff1d(corr_i, corr_j).shape[0]
                num_iw_jr = np.setdiff1d(corr_j, corr_i).shape[0]
                num_iw_jw = labels_B.shape[0] - (num_ir_jr + num_ir_jw + num_iw_jr)
                table = [[num_ir_jr, num_ir_jw], [num_iw_jr, num_iw_jw]]
                _, pvalue = mcnemar(table, exact=False)
                significance_level = 0.05
                if pvalue > significance_level:
                    sta_sig_ma[sys_len-i-1][j] -= 1.
    
    np.save('sta_sig_eer_tab3_rwc.npy', sta_sig_ma)
    print(system_list)
    # __import__('ipdb').set_trace()
    # sta_sig_ma = np.load('sta_sig_ma_mc.npy')
    # system_list = ['nsynth_sinc_mel_s100', 'nsynth_sinc_mel_s1000', 'nsynth_sinc_mel_s10000', 'nsynth_sinc_mel_aug_s100', 'nsynth_sinc_mel_aug_s1000', 'nsynth_sinc_mel_aug_s10000', 'nsynth_sinc_mel_asm_s100', 'nsynth_sinc_mel_asm_s1000', 'nsynth_sinc_mel_asm_s10000', 'nsynth_sinc_mel_aug_asm_s100', 'nsynth_sinc_mel_aug_asm_s1000', 'nsynth_sinc_mel_aug_asm_s10000', 'nsynth_sinc_midi_s100', 'nsynth_sinc_midi_s1000', 'nsynth_sinc_midi_s10000', 'nsynth_sinc_midi_aug_s100', 'nsynth_sinc_midi_aug_s1000', 'nsynth_sinc_midi_aug_s10000', 'nsynth_sinc_midi_asm_s100', 'nsynth_sinc_midi_asm_s1000', 'nsynth_sinc_midi_asm_s10000', 'nsynth_sinc_midi_aug_asm_s100', 'nsynth_sinc_midi_aug_asm_s1000', 'nsynth_sinc_midi_aug_asm_s10000', 'nsynth_mel_s100', 'nsynth_mel_s1000', 'nsynth_mel_s10000', 'nsynth_mel_aug_s100', 'nsynth_mel_aug_s1000', 'nsynth_mel_aug_s10000', 'nsynth_mel_asm_s100', 'nsynth_mel_asm_s1000', 'nsynth_mel_asm_s10000', 'nsynth_mel_aug_asm_s100', 'nsynth_mel_aug_asm_s1000', 'nsynth_mel_aug_asm_s10000', 'nsynth_midi_s100', 'nsynth_midi_s1000', 'nsynth_midi_s10000', 'nsynth_midi_aug_s100', 'nsynth_midi_aug_s1000', 'nsynth_midi_aug_s10000', 'nsynth_midi_asm_s100', 'nsynth_midi_asm_s1000', 'nsynth_midi_asm_s10000', 'nsynth_midi_aug_asm_s100', 'nsynth_midi_aug_asm_s1000', 'nsynth_midi_aug_asm_s10000']
    # print('system list:{}, len:{}'.format(system_list, len(system_list)))
    return system_list, sta_sig_ma


def plot_mat(ax, matrix, labels):
    ax.spy(matrix, markersize=5.9)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)
    # add grid lines on the matrix
    """
    loc = plticker.MultipleLocator(base=3)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(True, axis='both')
    ax.set_axisbelow(False)
    """


def main():
    ###########################################################################
    ### Data Preparation
    
    data_struct_v1 = {
        'Transform Layer':{
            'Mel':{
                'plain':{
                    '100':{
                        'prefix': 'nsynth_sinc_mel_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_mel_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_mel_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug':{
                    '100':{
                        'prefix': 'nsynth_sinc_mel_aug_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_mel_aug_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_mel_aug_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'asm':{
                    '100':{
                        'prefix': 'nsynth_sinc_mel_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_mel_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_mel_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug-asm':{
                    '100':{
                        'prefix': 'nsynth_sinc_mel_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_mel_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_mel_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
            'MIDI':{
                'plain':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                    },
                },
                'aug':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_aug_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_aug_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_aug_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'asm':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug-asm':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
        'Fixed Spectrogram':{
            'Mel':{
                'plain':{
                    '100':{
                        'prefix': 'nsynth_mel_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_mel_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_mel_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug':{
                    '100':{
                        'prefix': 'nsynth_mel_aug_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_mel_aug_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_mel_aug_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'asm':{
                    '100':{
                        'prefix': 'nsynth_mel_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_mel_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_mel_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug-asm':{
                    '100':{
                        'prefix': 'nsynth_mel_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_mel_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_mel_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
            'MIDI':{
                'plain':{
                    '100':{
                        'prefix': 'nsynth_midi_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_midi_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_midi_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug':{
                    '100':{
                        'prefix': 'nsynth_midi_aug_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_midi_aug_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_midi_aug_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'asm':{
                    '100':{
                        'prefix': 'nsynth_midi_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_midi_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_midi_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'aug-asm':{
                    '100':{
                        'prefix': 'nsynth_midi_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_midi_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_midi_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
    }
    data_struct_tab2_nsynth = {
        'Type - Mel':{
            'Update - False':{
                'Dim - 80':{
                    '100':{
                        'prefix': 'nsynth_mel_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_mel_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_mel_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'Dim - 122':{
                    '100':{
                        'prefix': 'nsynth_mel122_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_mel122_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_mel122_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
            'Update - True':{
                'Dim - 80':{
                    '100':{
                        'prefix': 'nsynth_sinc_mel_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_mel_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_mel_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'Dim - 122':{
                    '100':{
                        'prefix': 'nsynth_sinc_mel122_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_mel122_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_mel122_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
        'Type - CQT':{
            'Update - False':{
                'Dim - 122':{
                    '100':{
                        'prefix': 'nsynth_midi_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_midi_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_midi_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
            'Update - True':{
                'Dim - 122':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
    }
    data_struct_tab2_rwc = {
        'Mel':{ # Mel
            'No Update':{
                '80':{
                    '100':{
                        'prefix': 'rwc_fil_mel_asm_s100_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_mel_asm_s1000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_mel_asm_s10000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                '122':{
                    '100':{
                        'prefix': 'rwc_fil_mel122_aug_asm_s100_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_mel122_aug_asm_s1000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_mel122_aug_asm_s10000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
            'Update':{ # Update
                '80':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_mel_asm_s100_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_mel_asm_s1000_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_mel_asm_s10000_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                '122':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_mel122_aug_asm_s100_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_mel122_aug_asm_s1000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_mel122_aug_asm_s10000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
        'CQT':{ # CQT
            'No Update':{ # Not Update
                '122':{
                    '100':{
                        'prefix': 'rwc_fil_midi_aug_asm_s100_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_midi_aug_asm_s1000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_midi_aug_asm_s10000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
            'Update':{ # Update
                '122':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s100_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s1000_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s10000_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
    }
    data_struct_tab3_nsynth = {
        'base':{
            'base':{ 
                'base':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'wo da':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_asm_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_asm_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_asm_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'wo if':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s100_instr',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s1000_instr',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_aug_asm_s10000_instr',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'wo as':{
                    '100':{
                        'prefix': 'nsynth_sinc_midi_aug_s100',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'nsynth_sinc_midi_aug_s1000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'nsynth_sinc_midi_aug_s10000',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
    }
    data_struct_tab3_rwc = {
        'base':{
            'base':{ 
                'base':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s100_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s1000_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s10000_pretrained_w_trans',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'wo da':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s100_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s1000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_midi_asm_s10000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'wo if':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_midi_aug_asm_s100_instr_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_midi_aug_asm_s1000_instr_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_midi_aug_asm_s10000_instr_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
                'wo as':{
                    '100':{
                        'prefix': 'rwc_fil_sinc_midi_aug_s100_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '1000':{
                        'prefix': 'rwc_fil_sinc_midi_aug_s1000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                    '10000':{
                        'prefix': 'rwc_fil_sinc_midi_aug_s10000_pretrained',
                        'tar_data': None,
                        'nontar_data': None,
                        'pred': None,
                    },
                },
            },
        },
    }

    data_struct = data_struct_tab3_rwc
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # system_list, sta_sig_ma_eer = generate_data(data_struct)
    system_list, sta_sig_ma = generate_data(data_struct, mode='eer')
    return
    plot_mat(ax, sta_sig_ma, system_list)
    label_group_bar(ax, data_struct)
    fig.subplots_adjust(left=0.3, bottom=0.3)
    fig.tight_layout()
    fig.savefig('statistical_significance_test_eer_tab2.pdf')


if __name__ == '__main__':
    main()
