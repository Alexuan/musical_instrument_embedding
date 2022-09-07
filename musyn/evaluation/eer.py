import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import copy
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pybosaris.libperformance import compute_roc, rocch2eer, rocchdet
from musyn.evaluation.tsne import plot_DET


def compute_eer_det(feat, label, enroll_num=5, tar_num=20, nontar_num=20, save_dir=None):
    # feat: n*d
    # label: n
    random.seed(1024)
    np.random.seed(1024)
    cls_label = np.unique(label)
    tar_scores = []
    nontar_scores = []

    for i in cls_label:
        # __import__('ipdb').set_trace()
        cls_index = np.where(label==i)[0]
        enroll_index = np.random.choice(cls_index, enroll_num)
        tar_index = np.setdiff1d(cls_index, enroll_index)
        nontar_index = np.setdiff1d(np.arange(label.shape[0]), cls_index)
        for item in tar_index:
            # __import__('ipdb').set_trace()
            tar_index_sub = np.random.choice(tar_index, tar_num-1)
            tar_index_sub = np.concatenate([np.array([item]), tar_index_sub])
            nontar_index_sub = np.random.choice(nontar_index, nontar_num)
            # tar_index = np.random.choice(nontar_index, tar_num)
            # nontar_index = np.random.choice(nontar_index, nontar_num)

            # extrace feature for enrollment, target, and non-target
            enroll_feat = feat[enroll_index].mean(axis=0, keepdims=True)
            tar_feat = feat[tar_index_sub]
            nontar_feat = feat[nontar_index_sub]

            # compute cosine similarity score between enrollment and tar/non-tar
            tar_score = cosine_similarity(enroll_feat, tar_feat)
            nontar_score = cosine_similarity(enroll_feat, nontar_feat)
            tar_scores.append(tar_score.squeeze())
            nontar_scores.append(nontar_score.squeeze())

    tar_scores = np.hstack(tar_scores)
    nontar_scores = np.hstack(nontar_scores)
    if save_dir is not None:
        np.save(os.path.join(save_dir, 'tar_scores.npy'), tar_scores)
        np.save(os.path.join(save_dir, 'nontar_scores.npy'), nontar_scores)
    _, _, eer, _ = rocchdet(tar_scores, nontar_scores)
    print('EER: {}'.format(eer))
    return eer


if __name__ == '__main__':
    feat = np.load(sys.argv[1])
    label = np.load(sys.argv[2])
    save_dir = sys.argv[3]
    eer = compute_eer_det(feat, label)
    plot_DET(x, y, eer, save_dir)
    # plot_ROC(Pmiss, Pfa, save_dir)
