from pybosaris.calibration import LinearFuser
from pybosaris.libmath import logit
from pybosaris.calibration.objectives import CllrObjective, ReplaceHessian, ObjectiveFunction, SumOfFunctions, PenaltyFunction
import logging
import numpy
import copy


logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__license__ = "LGPLv3"
__credits__ = ["Niko Brummer", "Edward de Villiers"]


def train_binary_classifier(classifier, classf, w0, objective_function=None, prior=0.5, penalizer=None,
                            penalizer_weight=0, maxiters=100,
                            maxCG=100, optimizerState=None, quiet=True, cstepHessian=True):
    """
    from bosaris_toolkit/utility_funcs/Optimization_Toolkit/applications/fusion2class/train_binary_classifier.m
    :param classifier:
    :param classf:
    :param w0:
    :param objective_function:
    :param prior:
    :param penalizer:
    :param penalizer_weight:
    :param maxiters:
    :param maxCG:
    :param optimizerState:
    :param quiet:
    :param cstepHessian:
    :return:
    %
    %   Supervised training of a regularized fusion.
    %
    %
    % Inputs:
    %
    %   classifier: MV2DF function handle that maps parameters to llr-scores.
    %               Note: The training data is already wrapped in this handle.
    %
    %   classf: 1-by-N row of class labels:
    %                -1 for non_target,
    %                +1 for target,
    %                 0 for ignore
    %
    %   w0: initial parameters. This is NOT optional.
    %
    %   objective_function: A function handle to an Mv2DF function that
    %                       maps the output (llr-scores) of classifier, to
    %                       the to-be-minimized objective (called cxe).
    %                       optional, use [] to invoke 'cllr_obj'.
    %
    %  prior: a prior probability for target to set the 'operating point'
    %         of the objective function.
    %         optional: use [] to invoke default of 0.5
    %
    %  penalizer: MV2DF function handle that maps parameters to a positive
    %              regularization penalty.
    %
    %  lambda: a weighting for the penalizer
    %
    %  maxiters: the maximum number of Newton Trust Region optimization
    %            iterations to perform. Note, the user can make maxiters
    %            small, examine the solution and then continue training:
    %            -- see w0 and optimizerState.
    %
    %
    %
    %  optimizerState: In this implementation, it is the trust region radius.
    %                  optional:
    %                    omit or use []
    %                    If not supplied when resuming iteration,
    %                    this may cost some extra iterations.
    %                  Resume further iteration thus:
    %   [w1,...,optimizerState] = train_binary_classifier(...);
    %   ... examine solution w1  ...
    %   [w2,...,optimizerState] = train_binary_classifier(...,w1,...,optimizerState);
    %
    %
    %  quiet: if false, outputs more info during training
    %
    %
    %  Outputs:
    %    w: the solution.
    %    cxe: normalized multiclass cross-entropy of the solution.
    %         The range is 0 (good) to 1(useless).
    %
    %    optimizerState: see above, can be used to resume iteration.
    %
    """

    assert(isinstance(classifier, LinearFuser))

    if objective_function is None:
        objective_function = lambda T, weights, logit_prior: CllrObjective(T=T, weights=weights, logit_prior=logit_prior)

    logit_prior = logit(prior)

    # prior_entropy = -prior*log(prior)-(1-prior)*log(1-prior);
    prior_entropy = objective_function(
        T=numpy.array([1,-1]),
        weights=numpy.array([prior, 1-prior]),
        logit_prior=logit_prior
    ).objective(w=numpy.array([0, 0]))

    ntar = (classf > 0).sum()
    nnon = (classf < 0).sum()
    N = nnon+ntar

    weights = numpy.zeros(classf.shape)
    weights[classf > 0] = prior / (ntar * prior_entropy)
    weights[classf < 0] = (1-prior) / (nnon * prior_entropy)
    # weights remain 0, where classf==0

    w = None

    if penalizer is not None:
        assert(isinstance(penalizer, ObjectiveFunction))
        obj1 = objective_function(T=classf, weights=weights, logit_prior=logit_prior).compose_with(inner=classifier.fusion_obj)
        # obj2 = penalizer(w)  # TODO clarify
        obj2 = PenaltyFunction()
        obj = SumOfFunctions(
            weights=numpy.array([1, penalizer_weight]), 
            f=obj1, 
            g=obj2)
    else:
        obj = objective_function(T=classf, weights=weights, logit_prior=logit_prior).compose_with(inner=classifier.fusion_obj)
        obj2 = None

    if cstepHessian:
        obj = ReplaceHessian(f=obj, cstep=cstepHessian)

    w, y, optimizerState, converged = trustregion_newton_cg(
        f=obj, w=w0, maxiters=maxiters, maxCG=maxCG,
        state=optimizerState, eta=1e-4, nonnegative_f=True, quiet=quiet)

    if penalizer is not None:
        w_pen = penalizer_weight * obj2.objective(w)  # TODO clarify
    else:
        w_pen = 0

    cxe = y - w_pen
    if not quiet:
        logging.info('cxe = %g, pen = %g',cxe,w_pen)

    return w, cxe, w_pen, optimizerState, converged


def trustregion_newton_cg(f, w, maxiters, maxCG=100, state=None, eta=1e-4, nonnegative_f=False, quiet=True):
    """
    % Trust Region Newton Conjugate Gradient (TRNCG):
    %   --- Algorithm for Large-scale Unconstrained Minimization  ---
    %
    % Inputs:
    %    f: the objective function, which maps R^n to R.
    %       this function must implement the MV2DF interface in order to privde
    %       1st and 2nd order derivatives to the optimizer.
    %
    %    w: the optimization starting point in R^n
    %    maxiters: the maximum number of outer (Newton) iterations
    %    maxCG: (optional, default = 100) the maximum number of inner (conjugate gradient) iterations per
    %           outer iteration. In some cases CG eats memory. Reducing maxCG can be
    %           used to control memory consumption at the expense of possibly
    %           slower convergence. (However, in some cases, reducing maxCG may also in fact speed
    %           convergence by avoiding fruitless CG iterations.)
    %    state: (optional) if iteration is resumed this must be state as returned by previous
    %           iteration. If not given state is recomputed (although it may not be equivalent).
    %    eta:   (optional, default eta = 1e-4). Threshold for accepting iteration. Iteration is
    %                      rejected (and backtracking is done), if rho < eta.
    %                      rho is goodness of quadratic model prediction of
    %                      decrease in objective value.
    %                      Perfect prediction gives rho = 1. Increase gives
    %                      rho<0. Decrease better than model gives rho>1.
    %                      Legal values: 0 <= eta < 0.25.
    %    nonnegative_f: (optional, default = false) flag to assert that f>=0.
    %                   If true, CG iteration can be interrupted if quadratic
    %                   model prediction goes below zero.
    %    quiet: (optional, default = false) If true, does not output iteration
    %           information to screen.
    """
    assert(isinstance(f, ObjectiveFunction))

    converged = False

    if eta >= 0.25 or eta < 0:
        raise Exception('illegal eta')

    if state is not None:
        y = state.y
        g = state.g
        hess = state.hess
        delta = state.Delta
    else:
        y, deriv = f.objective_and_derivatives(w)
        g, hess, _, _ = deriv(1)
        delta = numpy.sqrt(g.T @ g)

    if not quiet:
        logging.info('TR 0 (initial state): obj = %g, Delta = %g', y, delta)

    delta_max = numpy.inf

    Hd = None
    iter = 1
    while iter <= maxiters:
        gmag = numpy.sqrt(g.T @ g)
        if gmag == 0:
            if not quiet:
                logging.info('TR %i: converged with zero gradient', iter)
            converged = True
            state = None
            break

        # epsilon = min(0.5,sqrt(gmag))*gmag; %Nocedal
        epsilon = 0.1 * gmag  # Lin

        if nonnegative_f:
            ycheck = y
        else:
            ycheck = None

        z, mz, onrim, Hd = cg_steihaug(g, hess, delta, epsilon, maxCG, ycheck, Hd, quiet)

        #  sanity check on delta_m --- looks good
        # check_m = -z'*g - 0.5*z'*hess(z);
        # [check_m,-mz]

        if z.T @ z == 0:
            if not quiet:
                logging.info('TR %i: converged with zero step', iter)
            converged = True
            state = None
            break

        w_new = w + z
        y_new, deriv_new = f.objective_and_derivatives(w_new)

        if nonnegative_f and y_new == 0:
            if not quiet:
                logging.info('TR %i: converged with zero objective', iter)
            converged = True
            w = w_new
            state = None
            break

        rho = (y_new - y) / mz

        if numpy.abs(mz) < numpy.sqrt(numpy.finfo(float).eps) and rho>0.75 and onrim == False:
            if not quiet:
                logging.info('TR %i: converged with minimal model change', iter)
            converged = True
            state = None
            break

        if rho < 0.25:
            delta = delta/4
            if not quiet:
                logging.info('contracting: Delta=%g', delta)
        elif rho > 0.75 and onrim:
            delta = min(2*delta, delta_max)
            if not quiet:
                logging.info('expanding: Delta=%g', delta)

        if rho > eta:
            if not quiet:
                logging.info('TR %i: obj=%g; rho=%g', iter, y_new, rho)
            w = w_new
            y = y_new
            g, hess, _, _ = deriv_new(1)
            Hd = None
            iter += 1
        else:
            if not quiet:
                logging.info('TR %i: obj=%g; backtracking; rho=%g', iter, y, rho)

    state = {
        'Delta': delta,
        'y': y,
        'g': g,
        'hess': hess
    }
    return w, y, state, converged


def cg_steihaug(grad, hess, Delta, epsilon, maxCG, y, Hd_back, quiet):
    # % Helper function for trustregion_newton_cg

    save_memory = True
    
    onrim = False
    backtrack = False
    if Hd_back is not None:
        if Hd_back.shape[1] > 0:
            Hd_record = Hd_back
            backtrack = True

    if not backtrack:
        if save_memory:
            Hd_record = numpy.zeros((len(grad), maxCG), dtype=numpy.float32)
        else:
            Hd_record = numpy.zeros((len(grad), maxCG))

    z = numpy.zeros(grad.shape)  # z is step, we start at origin
    mz = 0                 # 2nd order prediction for objective change at step z
    r = grad               # 2nd order residual at z:  r = H*z-grad
    d = -r                 # we're going down

    residual = numpy.sqrt(r.T @ r) / epsilon
    if residual <= 1:
        Hd_record = None
        logging.info('CG 0: as far as I can see, this should never happen')
        logging.info('CG 0: converged with zero step, residual = %g', residual)
        return

    j=0
    while True:
        if backtrack and j < Hd_back.shape[1]:
            Hd = Hd_back[:, j].astype(numpy.float)
        else:
            Hd = hess(d)
            if j < Hd_record.shape[1]:
                Hd_record[:, j] = Hd
        
        dHd = d.T @ Hd
        if dHd <= 0:  # region is non-convex in direction d
            a = d.T @ d
            b = 2 * z.T @ d
            c = z.T @ z - Delta**2
            discr = numpy.sqrt(b**2 - 4 * a * c)
            tau1 = (-b - discr) / (2*a)
            tau2 = (-b + discr) / (2*a)
            model = lambda tau: tau * grad.T @ d + 0.5 * tau**2 * dHd
            if model(tau1) < model(tau2):
                tau = tau1
            else:
                tau = tau2
            z += tau * d
            mz += tau * d.T @ grad + 0.5 * tau**2 * dHd
            onrim = True
            if not quiet:
                logging.info('CG %i: curv=%g, jump to trust region boundary', j, dHd)
            break
        
        alpha = r.T @ r / dHd
        old_z = copy.deepcopy(z)
        old_mz = copy.deepcopy(mz)
        z += alpha * d
        mz += alpha * d.T @ grad + 0.5 * alpha ** 2 * dHd
        radius = numpy.sqrt(z.T @ z) / Delta
        if radius > 1:
            a = d.T @ d
            b = 2 * z.T @ d
            c = old_z.T @ old_z - Delta**2
            discr = numpy.sqrt(b**2 - 4 * a * c)
            tau = (-b + discr) / (2*a)
            z = old_z + tau * d
            mz = old_mz + tau * d.T @ grad + 0.5 * tau**2 * dHd
            onrim = True
            if not quiet:
                logging.info('CG {}: curv={}, terminate on trust region boundary, model={}'.format(j, dHd, -mz))
            break
        
        old_r = copy.deepcopy(r)
        r += alpha * Hd
        residual = numpy.sqrt(r.T @  r) / epsilon
        if residual <= 1:
            if not quiet:
                logging.info('CG %i: curv=%G, converged inside trust region; radius = %g, residual=%g, model=%g', j, dHd, radius, residual, -mz)
            break
            
        overshot = y is not None
        if overshot:
            overshot &= ((y + mz) < 0)
        if overshot:
            if not quiet:
                logging.info('CG %i: curv=%G, overshot inside trust region; radius = %g, residual=%g, model=%g', j, dHd, radius, residual, -mz)
            break

        # stopped = backtrack && j+1 >= maxCG;
        stopped = (j+1) >= maxCG
        if stopped:
            if not quiet:
                logging.info('CG %i: curv=%G, stopped inside trust region; radius = %g, residual=%g, model=%g', j, dHd, radius, residual, -mz)
            break
            
        beta = (r.T @ r) / (old_r.T @ old_r)
        d = -r + beta * d
        if not quiet:
            logging.info('CG %i: curv=%g, radius = %g, residual=%g, model=%g', j, dHd, radius, residual, -mz)
        j = j+1
        
    if j < Hd_record.shape[1]:
        Hd_record = Hd_record[:, :j]
    
    return z, mz, onrim, Hd_record
