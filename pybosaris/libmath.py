import numpy
import scipy
import copy


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__credits__ = ["Niko Brummer", "Edward de Villiers", "Anthony Larcher"]
__license__ = "LGPLv3"


def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c


def probit(p):
    """Map from [0,1] to [-inf,inf] as used to make DET out of a ROC

    :param p: the value to map

    :return: probit(input)
    """
    y = numpy.sqrt(2) * scipy.special.erfinv(2 * p - 1)
    return y


def logit(p):
    """logit function.
    This is a one-to-one mapping from probability to log-odds.
    i.e. it maps the interval (0,1) to the real line.
    The inverse function is given by SIGMOID.

    log_odds = logit(p) = log(p/(1-p))

    :param p: the input value

    :return: logit(input)
    """
    p = numpy.array(p)
    lp = numpy.zeros(p.shape)
    f0 = p == 0
    f1 = p == 1
    f = (p > 0) & (p < 1)

    if lp.shape == ():
        if f:
            lp = numpy.log(p / (1 - p))
        elif f0:
            lp = -numpy.inf
        elif f1:
            lp = numpy.inf
    else:
        lp[f] = numpy.log(p[f] / (1 - p[f]))
        lp[f0] = -numpy.inf
        lp[f1] = numpy.inf
    return lp


def sigmoid(log_odds):
    """SIGMOID: Inverse of the logit function.
    This is a one-to-one mapping from log odds to probability.
    i.e. it maps the real line to the interval (0,1).

    p = sigmoid(log_odds)

    :param log_odds: the input value

    :return: sigmoid(input)
    """
    p = 1 / (1 + numpy.exp(-log_odds))
    return p


def pavx(y):
    """PAV: Pool Adjacent Violators algorithm.
    Non-paramtetric optimization subject to monotonicity.

    ghat = pav(y)
    fits a vector ghat with nondecreasing components to the
    data vector y such that sum((y - ghat).^2) is minimal.
    (Pool-adjacent-violators algorithm).

    optional outputs:
            width: width of pav bins, from left to right
                    (the number of bins is data dependent)
            height: corresponding heights of bins (in increasing order)

    Author: This code is a simplified version of the 'IsoMeans.m' code
    made available by Lutz Duembgen at:
    http://www.imsv.unibe.ch/~duembgen/software

    :param y: input value
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Input array is empty'
    n = y.shape[0]

    index = numpy.zeros(n,dtype=int)
    length = numpy.zeros(n,dtype=int)

    ghat = numpy.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j+1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[numpy.max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = copy.deepcopy(ghat[:ci + 1])
    width = copy.deepcopy(length[:ci + 1])

    while n >= 0:
        for j in range(index[ci], n+1):
            ghat[j-1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def pavx_mapping(tar, non, monotonicity_epsilon=1e-6, large_value=1e6):
    # TODO unit test
    
    def map(scores, score_bounds, llr_bounds):
        p = (scores[:, None] >= score_bounds).sum(axis=1) - 1
        p2 = p + 1
        idx = p2 < len(score_bounds)  # failsafe for LLRs > 1e6; should not happen at all...
        
        v1 = llr_bounds[p]

        x1 = score_bounds[p[idx]]
        x2 = score_bounds[p2[idx]]
        v2 = llr_bounds[p2[idx]]

        v1[idx] += (scores[idx] - x1) / (x2 - x1) * (v2 - v1[idx])

        return v1


    scores = numpy.concatenate([[-large_value], non, tar, [large_value]])
    Pideal = numpy.concatenate([[1], numpy.zeros(len(non)), numpy.ones(len(tar)), [0]])
    perturb = numpy.argsort(scores, kind='mergesort')
    scores = scores[perturb]
    Pideal = Pideal[perturb]
    
    Popt, width, foo = pavx(Pideal)
    data_prior = (len(tar) + 1)/len(Pideal)
    llrs = logit(Popt) - logit(data_prior)
    
    # make bounds
    bnd_len = 2 * len(width)
    c = numpy.cumsum(width - 1)
    bnd_ndx = numpy.zeros(bnd_len)
    bnd_ndx[::2] = numpy.concatenate([[0], c[:-1]+1])
    bnd_ndx[1::2] = c + 1
    score_bounds = scores[bnd_ndx.astype(int)]
    llr_bounds = llrs[bnd_ndx.astype(int)]
    llr_bounds[::2] = llr_bounds[::2] - monotonicity_epsilon
    llr_bounds[1::2] = llr_bounds[1::2] + monotonicity_epsilon

    return lambda s: map(s, score_bounds=score_bounds, llr_bounds=llr_bounds)


def optimal_llr_from_Popt(Popt, perturb, Ntar, Nnon, monotonicity_epsilon=1e-6):
    posterior_log_odds = logit(Popt)
    log_prior_odds = numpy.log(Ntar/Nnon)
    llrs = posterior_log_odds - log_prior_odds
    N = Ntar + Nnon
    llrs = llrs + numpy.arange(N) * monotonicity_epsilon/N # preserve monotonicity

    idx_reverse = numpy.zeros(N, dtype=int)
    idx_reverse[perturb] = numpy.arange(N)
    llrs_reverse = llrs[idx_reverse]
    tar_llrs = llrs_reverse[:Ntar]
    nontar_llrs = llrs_reverse[Ntar:]

    return tar_llrs, nontar_llrs


def optimal_llr(tar, non, laplace=False, monotonicity_epsilon=1e-6):
    # flag Laplace: avoids infinite LLR magnitudes;
    # also, this stops DET cureves from 'curling' to the axes on sparse data (DETs stay in more populated regions)
    scores = numpy.concatenate([tar, non])
    Pideal = numpy.concatenate([numpy.ones(len(tar)), numpy.zeros(len(non))])

    perturb = numpy.argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    if laplace:
       Pideal = numpy.hstack([1,0,Pideal,1,0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
      Popt = Popt[2:len(Popt)-2]

    tar_llrs, nontar_llrs = optimal_llr_from_Popt(Popt=Popt, perturb=perturb, Ntar=len(tar), Nnon=len(non), monotonicity_epsilon=monotonicity_epsilon)
    return tar_llrs, nontar_llrs


def neglogsigmoid(log_odds):
    neg_log_p = -log_odds
    e = numpy.exp(-log_odds)
    f = numpy.argwhere(e < e+1)[0]
    neg_log_p[f] = numpy.log(1+e[f])
    return neg_log_p
