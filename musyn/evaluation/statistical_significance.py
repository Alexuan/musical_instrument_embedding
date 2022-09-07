import os
import sys

import numpy as np
from scipy import stats

# TODO: change the path 
sys.path.append('/Users/shixuan/REPO/project-NN-Pytorch-scripts')
# sys.path.append('/home/smg/v-xuanshi/REPO/project-NN-Pytorch-scripts')
import itertools

from core_scripts.data_io import io_tools, wav_tools
from sandbox import eval_asvspoof
from tutorials.plot_tools import plot_API, plot_lib, table_API


###############################################################################
### FUNCTIONS
###############################################################################

### For plotting
def plot_mat(data_p_value, system_list, runs, cmap='RdBu'):
    ticklabels = range(len(system_list) * runs)[runs//2::runs]            
    total=len(system_list)*runs
    temp_lines = np.linspace(0, len(system_list)*runs, len(system_list)+1)
    data_list = [np.stack([np.array([0-.5, total-0.5]), np.array([x-0.5, x-0.5])], axis=1) for x in temp_lines]
    data_list = data_list + [np.stack([np.array([x-0.5, x-0.5]), np.array([0-.5, total-0.5])], axis=1) for x in temp_lines]
    data_list = data_list + [np.stack([np.array([0-0.5, total-0.5]), np.array([0-.5, total-0.5])], axis=1)]
    
    print(temp_lines)
    fig, axis = plot_API.plot_API2([data_p_value, data_list], [plot_lib.plot_imshow, plot_lib.plot_signal], [[[0,1],[0,1]], [[0,1],[0,1]]], 
                      {'figsize': (10, 10),
                       'sub': [
                       {'plot_imshow': {'cmap': cmap, 'origin': 'lower', 'aspect':'auto', 'interpolation': 'none'},
                       'yticks': ticklabels,
                       'yticklabels': {'labels': system_list, 'rotation': 0},
                       'xticks': ticklabels,
                       'xticklabels': {'labels': system_list, 'rotation': 90},
                       },
                       {'plot_signal': {'color': 'k', 'linewidth': 1.0, 'alpha':1.0}},]})
    return fig, axis


### For equal error rate computation
# From ASVspoof official tDCF https://www.asvspoof.org/resources/tDCF_python_v2.zip
def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size),
                             np.zeros(nontarget_scores.size)))

    # Sort labels based on scores                                                         
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates                                  
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size -
                            (np.arange(1, n_scores + 1) - tar_trial_sums))

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums/target_scores.size))
    # false rejection rates                                                               
    far = np.concatenate((np.atleast_1d(1),
                          nontarget_trial_sums / nontarget_scores.size))
    # false acceptance rates                                                              
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001),
                                 all_scores[indices]))
    # Thresholds are the sorted scores                                                    
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


### For significant-level correction
def reject_null_bonferroni_naive(z_values, significance_level, alternative='two-sided', 
                                 accept_value=True, reject_value=False):
    """result = reject_null_bonferroni_naive(z_values, significance_level)
    
    native bonferroni correction
    
    input
    -----
      z_values: np.array, an array of z_value
      signifiance_level: float, common choise is 0.1, 0.05, or 0.01

    output
    ------
      result: np.array, same size as z_values, if result[i] is True if z_value[i]
         is larger than the threshold
    """
    num_test = z_values.size
    new_conf_level = significance_level / num_test
    if alternative == 'less':
        x = stats.norm.ppf(new_conf_level)
    elif alternative == 'greater':
        x = stats.norm.isf(new_conf_level)
    elif alternative == 'two-sided':
        x = stats.norm.isf(new_conf_level/2)
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    idx_reject = z_values > x
    idx_accept = z_values <= x
    result = np.zeros(z_values.shape)
    result[idx_accept] = accept_value
    result[idx_reject] = reject_value
    return result


def reject_null_sidak(z_values, significance_level, alternative='two-sided', 
                      accept_value=True, reject_value=False):
    
    num_test = z_values.size
    
    FWER = 1 - (1 - significance_level) ** (1 / num_test)
    new_conf_level = 1- (1 - FWER) ** (1 / num_test)
    
    if alternative == 'less':
        x = stats.norm.ppf(new_conf_level)
    elif alternative == 'greater':
        x = stats.norm.isf(new_conf_level)
    elif alternative == 'two-sided':
        x = stats.norm.isf(new_conf_level/2)
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    idx_reject = z_values >= x
    idx_accept = z_values < x
    result = np.zeros(z_values.shape)
    result[idx_accept] = accept_value
    result[idx_reject] = reject_value
    return result


def reject_null_holm_bonferroni(z_values, significance_level, alternative='two-sided', 
                                accept_value=True, reject_value=False):
    order = np.argsort(z_values.flatten())
    index = {y:x for x,y in enumerate(order)}
    order = [index[x] for x in range(len(order))]
    sort_idx = np.reshape(order, z_values.shape)

    num_test = z_values.size
    new_conf_level = significance_level / (num_test - sort_idx)

    if alternative == 'less':
        x = stats.norm.ppf(new_conf_level)
    elif alternative == 'greater':
        x = stats.norm.isf(new_conf_level)
    elif alternative == 'two-sided':
        x = stats.norm.isf(new_conf_level/2)
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    idx_reject = z_values >= x
    idx_accept = z_values < x
    result = np.zeros(z_values.shape)
    result[idx_accept] = accept_value
    result[idx_reject] = reject_value
    return result


### For computing statistics
def compute_HTER_independent(far_a, frr_a, far_b, frr_b, NI, NC):
    """z = compute_HTER_independent(hter_a, hter_b, NI, NC)
    
    Bengio, S. & Mariéthoz, J. A statistical significance test for 
    person authentication. in Proc. Odyssey (2004). 
    
    Fig2. independent case
    
    input
    -----
      far_a: float, far of system a, which be >=0 and <=1
      frr_a: float, frr of system a, which be >=0 and <=1
      far_b: float, far of system b, which be >=0 and <=1
      frr_b: float, frr of system b, which be >=0 and <=1
      NI: int, the number of impostor accesses.
      NC: int, the number of client accesses.
      
    output
    ------
      z: float, statitics of the hypothesis test
    """
    # 
    hter_a = (far_a + frr_a)/2
    hter_b = (far_b + frr_b)/2
    denominator  = (far_a * (1 - far_a) + far_b * (1 - far_b)) / 4 / NI 
    denominator += (frr_a * (1 - frr_a) + frr_b * (1 - frr_b)) / 4 / NC 
    return np.abs(hter_a - hter_b) / np.sqrt(denominator)
    
    
def compute_HTER_dependent(far_ab, frr_ab, far_ba, frr_ba, NI, NC):
    """z = compute_HTER_independent(hter_a, hter_b, NI, NC)
    
    Bengio, S. & Mariéthoz, J. A statistical significance test for 
    person authentication. in Proc. Odyssey (2004). 
    
    Fig2. dependent case
    
    input
    -----
      far_ab: float, see paper
      frr_ab: float, see paper 
      far_ba: float, see paper 
      frr_ba: float, see paper 
      NI: int, the number of impostor accesses.
      NC: int, the number of client accesses.
      
    output
    ------
      z: float, statitics of the hypothesis test
    """
    # 
    if far_ab == far_ba and frr_ab == frr_ba:
        return 0
    else:
        denominator = np.sqrt((far_ab + far_ba) / (4 * NI) + (frr_ab + frr_ba) / (4 * NC))
        return np.abs(far_ab + frr_ab - far_ba - frr_ba) / denominator

    

def get_eer(scores_positive, scores_negative):
    """eer, threshold = get_eer(scores_positive, scores_negative)
    
    compute Equal Error Rate given input scores
    
    input
    -----
      scores_positive: np.array, scores of positive class
      scores_negative: np.array, scores of negative class
    
    output
    ------
      eer: float, equal error rate
      threshold: float, the threshold for the err
      
    """
    return compute_eer(scores_positive, scores_negative)
    
    
def get_far_frr_dependent(bona_score_a, spoof_score_a, threshold_a, 
                          bona_score_b, spoof_score_b, threshold_b, 
                          NI, NC):
    """
    """
    far_ab_idx = np.bitwise_and(spoof_score_a < threshold_a, spoof_score_b >= threshold_b)
    far_ba_idx = np.bitwise_and(spoof_score_a >= threshold_a, spoof_score_b < threshold_b)
    frr_ab_idx = np.bitwise_and(bona_score_a >= threshold_a, bona_score_b < threshold_b)
    frr_ba_idx = np.bitwise_and(bona_score_a < threshold_a, bona_score_b >= threshold_b)

    far_ab = np.sum(far_ab_idx) / NI
    far_ba = np.sum(far_ba_idx) / NI
    frr_ab = np.sum(frr_ab_idx) / NC
    frr_ba = np.sum(frr_ba_idx) / NC
    return far_ab, far_ba, frr_ab, frr_ba



def main():
    # https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts-in-matplotlib?noredirect=1&lq=1
    pass


if __name__ == '__main__':
    system_info = {
        ''
    }
    main()
