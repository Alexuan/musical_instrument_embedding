from numpy import zeros, ones, array, arange, argsort, cumsum, column_stack, hstack, vstack, isscalar, abs, concatenate, exp, log, inf, infty
from numpy.linalg import solve
from collections import namedtuple
from pybosaris.libmath import probit, logit, pavx, sigmoid, optimal_llr


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__credits__ = ["Niko Brummer", "Edward de Villiers", "Anthony Larcher"]
__license__ = "LGPLv3"


Box = namedtuple("Box", "left right top bottom")


def effective_prior(Ptar, cmiss, cfa):
    """This function adjusts a given prior probability of target p_targ,
    to incorporate the effects of a cost of miss,
    cmiss, and a cost of false-alarm, cfa.
    In particular note:
    EFFECTIVE_PRIOR(EFFECTIVE_PRIOR(p,cmiss,cfa),1,1)
            = EFFECTIVE_PRIOR(p,cfa,cmiss)

    The effective prior for the NIST SRE detection cost fuction,
    with p_targ = 0.01, cmiss = 10, cfa = 1 is therefore:
    EFFECTIVE_PRIOR(0.01,10,1) = 0.0917

    :param Ptar: is the probability of a target trial
    :param cmiss: is the cost of a miss
    :param cfa: is the cost of a false alarm

    :return: a prior
    """
    p = Ptar * cmiss / (Ptar * cmiss + (1 - Ptar) * cfa)
    return p


def logit_effective_prior(Ptar, cmiss, cfa):
    """This function adjusts a given prior probability of target p_targ,
    to incorporate the effects of a cost of miss,
    cmiss, and a cost of false-alarm, cfa.
    In particular note:
    EFFECTIVE_PRIOR(EFFECTIVE_PRIOR(p,cmiss,cfa),1,1)
            = EFFECTIVE_PRIOR(p,cfa,cmiss)

    The effective prior for the NIST SRE detection cost fuction,
    with p_targ = 0.01, cmiss = 10, cfa = 1 is therefore:
    EFFECTIVE_PRIOR(0.01,10,1) = 0.0917

    :param Ptar: is the probability of a target trial
    :param cmiss: is the cost of a miss
    :param cfa: is the cost of a false alarm

    :return: a prior
    """
    p = Ptar * cmiss / (Ptar * cmiss + (1 - Ptar) * cfa)
    return logit(p)


def DETsort(x, col=''):
    """DETsort Sort rows, the first in ascending, the remaining in descending
    thereby postponing the false alarms on like scores.
    based on SORTROWS

    :param x: the array to sort
    :param col: not used here

    :return: a sorted vector of scores
    """
    assert x.ndim > 1, 'x must be a 2D matrix'
    if col == '':
        list(range(1, x.shape[1]))

    ndx = arange(x.shape[0])

    # sort 2nd column ascending
    ind = argsort(x[:, 1], kind='mergesort')
    ndx = ndx[ind]

    # reverse to descending order
    ndx = ndx[::-1]

    # now sort first column ascending
    ind = argsort(x[ndx, 0], kind='mergesort')

    ndx = ndx[ind]
    sort_scores = x[ndx, :]
    return sort_scores


def compute_roc(true_scores, false_scores):
    """Computes the (observed) miss/false_alarm probabilities
    for a set of detection output scores.

    true_scores (false_scores) are detection output scores for a set of
    detection trials, given that the target hypothesis is true (false).
    (By convention, the more positive the score,
    the more likely is the target hypothesis.)

    :param true_scores: a 1D array of target scores
    :param false_scores: a 1D array of non-target scores

    :return: a tuple of two vectors, Pmiss,Pfa
    """
    num_true = true_scores.shape[0]
    num_false = false_scores.shape[0]
    assert num_true > 0, "Vector of target scores is empty"
    assert num_false > 0, "Vector of nontarget scores is empty"

    total = num_true + num_false

    Pmiss = zeros((total + 1))
    Pfa = zeros((total + 1))

    scores = zeros((total, 2))
    scores[:num_false, 0] = false_scores
    scores[:num_false, 1] = 0
    scores[num_false:, 0] = true_scores
    scores[num_false:, 1] = 1

    scores = DETsort(scores)

    sumtrue = cumsum(scores[:, 1], axis=0)
    sumfalse = num_false - (arange(1, total + 1) - sumtrue)

    Pmiss[0] = 0
    Pfa[0] = 1
    Pmiss[1:] = sumtrue / num_true
    Pfa[1:] = sumfalse / num_false
    return Pmiss, Pfa


def filter_roc(pm, pfa):
    """Removes redundant points from the sequence of points (pfa,pm) so
    that plotting an ROC or DET curve will be faster.  The output ROC
    curve will be identical to the one plotted from the input
    vectors.  All points internal to straight (horizontal or
    vertical) sections on the ROC curve are removed i.e. only the
    points at the start and end of line segments in the curve are
    retained.  Since the plotting code draws straight lines between
    points, the resulting plot will be the same as the original.

    :param pm: the vector of miss probabilities of the ROC Convex
    :param pfa: the vector of false-alarm probabilities of the ROC Convex

    :return: a tuple of two vectors, Pmiss, Pfa
    """
    out = 0
    new_pm = [pm[0]]
    new_pfa = [pfa[0]]

    for i in range(1, pm.shape[0]):
        if (pm[i] == new_pm[out]) | (pfa[i] == new_pfa[out]):
            pass
        else:
            # save previous point, because it is the last point before the
            # change.  On the next iteration, the current point will be saved.
            out += 1
            new_pm.append(pm[i - 1])
            new_pfa.append(pfa[i - 1])

    out += 1
    new_pm.append(pm[-1])
    new_pfa.append(pfa[-1])
    pm = array(new_pm)
    pfa = array(new_pfa)
    return pm, pfa


def rocch2eer(pmiss, pfa):
    """Calculates the equal error rate (eer) from pmiss and pfa vectors.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.
    Use rocch.m to convert target and non-target scores to pmiss and
    pfa values.

    :param pmiss: the vector of miss probabilities
    :param pfa: the vector of false-alarm probabilities

    :return: the equal error rate
    """
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = column_stack((xx, yy))
        dd = array([1, -1]) @ XY
        if abs(dd).min() == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = solve(XY, array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / seg.sum()

        eer = max([eer, eerseg])
    return eer


def rocch_pava(tar_scores, nontar_scores, laplace=False):
    """ROCCH: ROC Convex Hull.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.

    :param tar_scores: vector of target scores
    :param nontar_scores: vector of non-target scores

    :return: a tupple of two vectors: Pmiss, Pfa
    """
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = concatenate((tar_scores, nontar_scores))
    # Pideal is the ideal, but non-monotonic posterior
    Pideal = concatenate((ones(Nt), zeros(Nn)))

    # It is important here that scores that are the same
    # (i.e. already in order) should NOT be swapped.rb
    perturb = argsort(scores, kind='mergesort')
    #
    Pideal = Pideal[perturb]

    if laplace:
       Pideal = hstack([1,0,Pideal,1,0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
      Popt = Popt[2:len(Popt)-2]

    nbins = width.shape[0]
    pmiss = zeros(nbins + 1)
    pfa = zeros(nbins + 1)

    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = Nn
    miss = 0

    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = Pideal[:left].sum()
        fa = N - left - Pideal[left:].sum()

    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn

    return pmiss, pfa, Popt, perturb


def rocch(tar_scores, nontar_scores):
    pmiss, pfa, _, _ = rocch_pava(tar_scores, nontar_scores)
    return pmiss, pfa


def fast_actDCF(tar,non,plo,normalize=False):
    D = 1
    if not isscalar(plo):
        D = len(plo)
    T = len(tar)
    N = len(non)

    ii = argsort(hstack([-plo,tar]))
    r = zeros(T+D)
    r[ii] = arange(T+D) + 1
    r = r[:D]
    Pmiss = r - arange(start=D, step=-1, stop=0)

    ii = argsort(hstack([-plo,non]))  # -plo are thresholds
    r = zeros(N+D)
    r[ii] = arange(N+D) + 1
    r = r[:D]  # rank of thresholds
    Pfa = N - r + arange(start=D, step=-1, stop=0)

    Pmiss = Pmiss / T
    Pfa = Pfa / N

    Ptar = sigmoid(plo)
    Pnon = sigmoid(-plo)
    dcf = Ptar * Pmiss + Pnon * Pfa

    if normalize:
        dcf /= min([Ptar, Pnon])

    return dcf


def fast_minDCF(tar, non, plo, normalize=False):
    """Compute the minimum COST for given target and non-target scores
    Note that minDCF is parametrized by plo:

        minDCF(Ptar) = min_t Ptar * Pmiss(t) + (1-Ptar) * Pfa(t)

    where t is the adjustable decision threshold and:

        Ptar = sigmoid(plo) = 1./(1+exp(-plo))

    If normalize == true, then the returned value is:

        minDCF(Ptar) / min(Ptar,1-Ptar).

    Pmiss: a vector with one value for every element of plo.
    This is Pmiss(tmin), where tmin is the minimizing threshold
    for minDCF, at every value of plo. Pmiss is not altered by
    parameter 'normalize'.

    Pfa: a vector with one value for every element of plo.
    This is Pfa(tmin), where tmin is the minimizing threshold for
    minDCF, at every value of plo. Pfa is not altered by
    parameter 'normalize'.

    Note, for the un-normalized case:

        minDCF(plo) = sigmoid(plo).*Pfa(plo) + sigmoid(-plo).*Pmiss(plo)

    :param tar: vector of target scores
    :param non: vector of non-target scores
    :param plo: vector of prior-log-odds: plo = logit(Ptar) = log(Ptar) - log(1-Ptar)
    :param normalize: if true, return normalized minDCF, else un-normalized (optional, default = false)

    :return: the minDCF value
    :return: the miss probability for this point
    :return: the false-alarm probability for this point
    :return: the precision-recall break-even point: Where #FA == #miss
    :return the equal error rate
    """
    Pmiss, Pfa = rocch(tar, non)
    Nmiss = Pmiss * tar.shape[0]
    Nfa = Pfa * non.shape[0]
    prbep = rocch2eer(Nmiss, Nfa)
    eer = rocch2eer(Pmiss, Pfa)

    Ptar = sigmoid(plo)
    Pnon = sigmoid(-plo)
    cdet = vstack((Ptar, Pnon)).T @ vstack((Pmiss, Pfa))
    ii = cdet.argmin(axis=1)
    minDCF = cdet[0, ii][0]

    Pmiss = Pmiss[ii]
    Pfa = Pfa[ii]

    if normalize:
        minDCF /= min([Ptar, Pnon])

    # return minDCF, Pmiss[0], Pfa[0], prbep, eer
    return minDCF, Pmiss, Pfa, prbep, eer


def plotseg(xx, yy, box, dps):
    """Prepare the plotting of a curve.
    :param xx:
    :param yy:
    :param box:
    :param dps:
    """
    assert ((xx[1] <= xx[0]) & (yy[0] <= yy[1])), 'xx and yy should be sorted'

    XY = column_stack((xx, yy))
    dd = array([1, -1]) @  XY
    if abs(dd).min() == 0:
        eer = 0
    else:
        # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
        # when xx(i),yy(i) is on the line.
        seg = solve(XY, array([[1], [1]]))
        # candidate for EER, eer is highest candidate
        eer = 1.0 / seg.sum()

    # segment completely outside of box
    if (xx[0] < box.left) | (xx[1] > box.right) | (yy[1] < box.bottom) | (yy[0] > box.top):
        x = array([])
        y = array([])
    else:
        if xx[1] < box.left:
            xx[1] = box.left
            yy[1] = (1 - seg[0] * box.left) / seg[1]

        if xx[0] > box.right:
            xx[0] = box.right
            yy[0] = (1 - seg[0] * box.right) / seg[1]

        if yy[0] < box.bottom:
            yy[0] = box.bottom
            xx[0] = (1 - seg[1] * box.bottom) / seg[0]

        if yy[1] > box.top:
            yy[1] = box.top
            xx[1] = (1 - seg[1] * box.top) / seg[0]

        dx = xx[1] - xx[0]
        xdots = xx[0] + dx * arange(dps + 1) / dps
        ydots = (1 - seg[0] * xdots) / seg[1]
        x = probit(xdots)
        y = probit(ydots)

    return x, y, eer


def rocchdet(tar, non,
             dcfweights=array([]),
             pfa_min=5e-4,
             pfa_max=0.5,
             pmiss_min=5e-4,
             pmiss_max=0.5,
             dps=100,
             normalize=False):
    """ROCCHDET: Computes ROC Convex Hull and then maps that to the DET axes.
    The DET-curve is infinite, non-trivial limits (away from 0 and 1)
    are mandatory.

    :param tar: vector of target scores
    :param non: vector of non-target scores
    :param dcfweights: 2-vector, such that: DCF = [pmiss,pfa]*dcfweights(:)  (Optional, provide only if mindcf is
    desired, otherwise omit or use []
    :param pfa_min: limit of DET-curve rectangle. Default is 0.0005
    :param pfa_max: limit of DET-curve rectangle. Default is 0.5
    :param pmiss_min: limit of DET-curve rectangle. Default is 0.0005
    :param pmiss_max: limits of DET-curve rectangle.  Default is 0.5
    :param dps: number of returned (x,y) dots (arranged in a curve) in DET space, for every straight line-segment
    (edge) of the ROC Convex Hull. Default is 100.
    :param normalize: normalize the curve

    :return: probit(Pfa)
    :return: probit(Pmiss)
    :return: ROCCH EER = max_p mindcf(dcfweights=[p,1-p]), which is also equal to the intersection of the ROCCH
    with the line pfa = pmiss.
    :return: the mindcf: Identical to result using traditional ROC, but computed by mimimizing over the ROCCH
    vertices, rather than over all the ROC points.
    """
    assert ((pfa_min > 0) & (pfa_max < 1) & (pmiss_min > 0) & (pmiss_max < 1)), 'limits must be strictly inside (0,1)'
    assert ((pfa_min < pfa_max) & (pmiss_min < pmiss_max)), 'pfa and pmiss min and max values are not consistent'

    pmiss, pfa = rocch(tar, non)
    mindcf = 0.0

    if dcfweights.shape == (2,):
        dcf = dcfweights @ vstack((pmiss, pfa))
        mindcf = dcf.min()
        if normalize:
            mindcf = mindcf / min(dcfweights)

    # pfa is decreasing
    # pmiss is increasing
    box = Box(left=pfa_min, right=pfa_max, top=pmiss_max, bottom=pmiss_min)
    x = []
    y = []
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]
        xdots, ydots, eerseg = plotseg(xx, yy, box, dps)
        x = x + xdots.tolist()
        y = y + ydots.tolist()
        eer = max(eer, eerseg)
    return array(x), array(y), eer, mindcf


def cllr(tar_llrs, nontar_llrs):
    log2 = log(2)

    tar_posterior = sigmoid(tar_llrs)
    non_posterior = sigmoid(-nontar_llrs)
    if any(tar_posterior == 0) or any(non_posterior == 0):
        return inf

    c1 = (-log(tar_posterior)).mean() / log2
    c2 = (-log(non_posterior)).mean() / log2
    c = (c1 + c2) / 2
    return c


def min_cllr(tar_llrs, nontar_llrs, monotonicity_epsilon=1e-6):
    [tar,non] = optimal_llr(tar_llrs, nontar_llrs, laplace=False, monotonicity_epsilon=monotonicity_epsilon)
    cmin = cllr(tar, non)
    return cmin


def ece(tar, non, plo):
    if isscalar(tar):
        tar = array([tar])
    if isscalar(non):
        non = array([non])
    if isscalar(plo):
        plo = array([plo])

    ece = zeros(plo.shape)
    for i, p in enumerate(plo):
        ece[i] = sigmoid(p) * (-log(sigmoid(tar + p))).mean()
        ece[i] += sigmoid(-p) * (-log(sigmoid(-non - p))).mean()

    ece /= log(2)

    return ece

