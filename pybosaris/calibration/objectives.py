import numpy
from pybosaris.libmath import sigmoid, logit
from abc import ABC, abstractmethod
from functools import partial


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__license__ = "LGPLv3"
__credits__ = ["Niko Brummer", "Edward de Villiers"]


def evaluate_objective(scores, classf, prior, objective_function=None):
    assert objective_function is None or isinstance(objective_function, ObjectiveFunction)

    logit_prior = logit(prior)
    if objective_function is None:
        objective_function = lambda T, weights, logit_prior: \
            CllrObjective(T=T, weights=weights, logit_prior=logit_prior)

    prior_entropy = objective_function(
        T=numpy.array([1, -1]),
        weights=numpy.array([prior, 1 - prior]),
        logit_prior=logit_prior
    ).objective(numpy.array([0, 0]))

    tar = classf > 0
    non = classf < 0
    ntar = tar.sum()
    nnon = non.sum()
    N = nnon + ntar

    weights = numpy.zeros(N)
    weights[tar] = prior / (ntar * prior_entropy)
    weights[non] = (1 - prior) / (nnon * prior_entropy)

    obj_val = objective_function(T=classf, weights=weights, logit_prior=logit_prior).objective(scores)
    return obj_val


class ObjectiveFunction(ABC):
    def compose_with(self, inner):
        assert(isinstance(inner, ObjectiveFunction))
        y = ComposeObjectiveFunction(outer=self, inner=inner)
        return y

    @abstractmethod
    def objective(self, w):
        pass

    @abstractmethod
    def objective_and_gradient(self, w):
        pass

    @abstractmethod
    def objective_and_derivatives(self, w):
        pass

    @abstractmethod
    def hessian(self, *args):
        pass

    @abstractmethod
    def hessian_and_jacobian_product(self, *args):
        pass


class ComposeObjectiveFunction(ObjectiveFunction):
    def __init__(self, outer, inner):
        super().__init__()
        assert(isinstance(outer, ObjectiveFunction))
        assert(isinstance(inner, ObjectiveFunction))

        self.outer = outer
        self.inner = inner

    def objective(self, w):
        y1 = self.inner.objective(w)
        y = self.outer.objective(y1)
        return y

    def objective_and_gradient(self, w):
        y1, deriv1 = self.inner.objective_and_gradient(w)
        y, deriv2 = self.outer.objective_and_gradient(y1)

        def derive(x, deriv1, deriv2):
            return deriv1(deriv2(x))

        gradient = partial(derive, deriv1=deriv1, deriv2=deriv2)
        return y, gradient

    def objective_and_derivatives(self, w):
        y1, deriv1 = self.inner.objective_and_derivatives(w)
        y, deriv2 = self.outer.objective_and_derivatives(y1)

        def derivatives(dy, deriv1, deriv2):
            g2, hess2, lin2, jacob2 = deriv2(dy)
            g1, hess1, lin1, jacob1 = deriv1(g2)

            hess = partial(self.hessian, deriv1=deriv1, hess1=hess1, hess2=hess2, lin1=lin1, lin2=lin2, jacob1=jacob1)
            linear = lin1 and lin2
            hess_jacob = partial(self.hessian_and_jacobian_product, deriv1=deriv1, lin1=lin1, lin2=lin2, jacob1=jacob1, jacob2=jacob2)
            return g1, hess, linear, hess_jacob

        gradients = partial(derivatives, deriv1=deriv1, deriv2=deriv2)
        return y, gradients

    def hessian(self, d, deriv1, hess1, hess2, lin1, lin2, jacob1):
        if not lin2:
            h1, Jv1 = jacob1(d)
            h2 = hess2(Jv1)
            h2, _, _, _ = deriv1(h2)
        elif not lin1:
            Jv1 = None
            h1 = hess1(d)
            h2 = None

        if lin1 and lin2:
            h = None
        elif (not lin1) and (not lin2):
            h = h1 + h2
        elif lin1:
            h = h2
        else:
            h = h1

        return h

    def hessian_and_jacobian_product(self, d, deriv1, lin1, lin2, jacob1, jacob2):
        h1, Jv1 = jacob1(d)
        h2, Jv = jacob2(Jv1)
        if not lin2:
            h2, _, _, _ = deriv1(h2)

        if lin1 and lin2:
            h = []
        elif (not lin1) and (not lin2):
            h = h1 + h2
        elif lin1:
            h = h2
        else:  # if lin2
            h = h1
        return h, Jv


class StackObjectives(ObjectiveFunction):

    def __init__(self, f, g, eqlen=False):
        super().__init__()
        assert(isinstance(f, ObjectiveFunction))
        assert(isinstance(g, ObjectiveFunction))

        self.f = f
        self.g = g
        self.eqlen = eqlen

    def stack(self, y1, y2):
        return numpy.hstack((y1, y2))

    def objective(self, w):
        """
        % STACK is an MV2DF (see MV2DF_API_DEFINITION.readme) which
        % represents the new function, s(w), obtained by stacking the outputs of
        % f() and g() thus:
        %  s(w) = [f(w);g(w)]
        """
        """
        % if ~isa(f,'function_handle')
        %     f = const_mv2df([],f);
        % end
        % if ~isa(g,'function_handle')
        %     g = const_mv2df([],g);
        % end
        """
        y1 = self.f.objective(w)
        y2 = self.g.objective(w)
        n1 = len(y1)
        n2 = len(y2)
        if self.eqlen:
            assert n1 == n2, 'length(f(w))=%i must equal length(g(w))=%i.'.format(n1, n2)
        y = self.stack(y1=y1, y2=y2)
        return y

    def objective_and_gradient(self, w):
        y1, deriv1 = self.f.objective_and_gradient(w)
        y2, deriv2 = self.g.objective_and_gradient(w)
        y = self.stack(y1=y1, y2=y2)
        n1 = len(y1)
        n2 = len(y2)
        if self.eqlen:
            assert n1 == n2, 'length(f(w))=%i must equal length(g(w))=%i.'.format(n1, n2)

        def derive(g2, deriv1, deriv2, n1):
            return deriv1(g2[:n1]) + deriv2(g2[n1:])

        deriv = partial(derive, deriv1=deriv1, deriv2=deriv2, n1=n1)
        return y, deriv

    def objective_and_derivatives(self, w):
        y1, deriv1 = self.f.objective_and_derivatives(w)
        y2, deriv2 = self.g.objective_and_derivatives(w)
        y = self.stack(y1=y1, y2=y2)
        n1 = len(y1)
        n2 = len(y2)

        if self.eqlen:
            assert n1 == n2, 'length(f(w))=%i must equal length(g(w))=%i.'.format(n1, n2)

        def derivatives(dy, deriv1, deriv2):
            g1, hess1, lin1, hess_jacob1 = deriv1(dy[:n1])
            g2, hess2, lin2, hess_jacob2 = deriv2(dy[n1:])
            g = g1 + g2
            linear = lin1 and lin2
            hess = partial(self.hessian, hess1=hess1, hess2=hess2, lin1=lin1, lin2=lin2)
            hess_jacob = partial(self.hessian_and_jacobian_product, hess_jacob1=hess_jacob1, hess_jacob2=hess_jacob2, lin1=lin1, lin2=lin2)
            return g, hess, linear, hess_jacob

        gradients = partial(derivatives, deriv1=deriv1, deriv2=deriv2)
        return y, gradients

    def hessian(self, d, hess1, hess2, lin1, lin2):
        h1 = hess1(d)
        h2 = hess2(d)

        if lin1 and lin2:
            h = None
        elif (not lin1) and (not lin2):
            h = h1 + h2
        elif lin2:
            h = h1
        else:
            h = h2
        return h

    def hessian_and_jacobian_product(self, d, hess_jacob1, hess_jacob2, lin1, lin2):
        h1, Jv1 = hess_jacob1(d)
        h2, Jv2 = hess_jacob2(d)
        Jv = self.stack(Jv1, Jv2)

        if lin1 and lin2:
            h = None
        elif (not lin1) and (not lin2):
           h = h1 + h2
        elif lin2:
           h = h1
        else:
           h = h2

        return h, Jv


class ReplaceHessian(ObjectiveFunction):
    def __init__(self, f, cstep):
        super().__init__()
        assert(isinstance(f, ObjectiveFunction))
        assert(isinstance(cstep, bool))

        self.f = f
        self.cstep = cstep

    def objective(self, w):
        y = self.f.objective(w)
        return y

    def objective_and_gradient(self, w):
        y, derivf = self.f.objective_and_gradient(w)
        gradient = derivf
        return y, gradient

    def objective_and_derivatives(self, w):
        y, derivf = self.f.objective_and_gradient(w)

        def derivatives(dy, derivf, w):
            g = derivf(dy)
            linear = False
            hess = partial(self.hessian, dy=dy, w=w)
            hess_jacob = self.hessian_and_jacobian_product
            return g, hess, linear, hess_jacob

        gradients = partial(derivatives, derivf=derivf, w=w)
        return y, gradients

    def hessian(self, dx, dy, w):
        if self.cstep:
            # h = cstep_approxHess(dx, dy, f, w)
            x0 = w
            x = x0 + 1e-20j * dx
            _, deriv = self.f.objective_and_gradient(x)
            g = deriv(dy)
            p = 1e20 * numpy.imag(g)
            h = p
        else:
            # h = rstep_approxHess(dx, dy, f, w)
            x0 = w
            alpha = numpy.sqrt(numpy.finfo(x0.dtype).eps)
            x2 = x0 + alpha * dx
            _, deriv2 = self.f.objective_and_gradient(x2)
            x1 = x0 - alpha * dx
            _, deriv1 = self.f.objective_and_gradient(x1)
            g2 = deriv2(dy)
            g1 = deriv1(dy)
            x = (g2 - g1) / (2 * alpha)
            h = x

        return h

    def hessian_and_jacobian_product(self, *args):
        raise Exception('replace_hessian cannot compute Jv')
        # Jv = zeros(size(dy));


class SubVector(ObjectiveFunction):
    def __init__(self, size, first, length):
        super().__init__()
        self.size = size
        self.first = first
        self.last = first + length
        self.linear_transform = LinearTransform(
            map=partial(self.map),
            transmap=partial(self.transmap)
        )

    def map(self, w):
        return w[self.first:self.last]

    def transmap(self, w):
        g = numpy.zeros(self.size, dtype=w.dtype)
        g[self.first:self.last] = w
        return g

    def objective(self, w):
        return self.linear_transform.objective(w=w)

    def objective_and_gradient(self, w):
        return self.linear_transform.objective_and_gradient(w=w)

    def objective_and_derivatives(self, w):
        return self.linear_transform.objective_and_derivatives(w=w)

    def hessian(self, *args):
        pass

    def hessian_and_jacobian_product(self, *args):
        pass


class Transpose(ObjectiveFunction):
    def __init__(self, m, n):
        super().__init__()
        self.m = m
        self.n = n
        self.linear_transform = LinearTransform(
            map=partial(self.map),
            transmap=partial(self.transmap)
        )

    def map(self, w):
        # @(w) reshape(reshape(w,M,N).',[],1);
        return w.reshape(self.m, self.n, order='F').T.copy().flatten()

    def transmap(self, w):
        # @(w) reshape(reshape(w,N,M).',[],1);
        return w.reshape(self.n, self.m, order='F').T.copy().flatten()

    def objective(self, w):
        return self.linear_transform.objective(w=w)

    def objective_and_gradient(self, w):
        return self.linear_transform.objective_and_gradient(w=w)

    def objective_and_derivatives(self, w):
        return self.linear_transform.objective_and_derivatives(w=w)

    def hessian(self, *args):
        pass

    def hessian_and_jacobian_product(self, *args):
        pass


class SumOfFunctions(ObjectiveFunction):
    def __init__(self, weights, f, g):
        super().__init__()
        self.weights = weights
        self.f = f
        self.g = g
        self.n = len(weights)
        self.linear_stacked_transform = LinearTransform(
            map=partial(self.map),
            transmap=partial(self.transmap)
        ).compose_with(
            inner=StackObjectives(f=f, g=g, eqlen=True)
        )

    def map(self, s):
        return s.reshape(int(len(s)/self.n), self.n, order='F') @ self.weights

    def transmap(self, y):
        return (y[:,None] @ self.weights.T[None,:]).T.flatten()

    def objective(self, w):
        return self.linear_stacked_transform.objective(w=w)

    def objective_and_gradient(self, w):
        return self.linear_stacked_transform.objective_and_gradient(w=w)

    def objective_and_derivatives(self, w):
        return self.linear_stacked_transform.objective_and_derivatives(w=w)

    def hessian(self, *args):
        pass

    def hessian_and_jacobian_product(self, *args):
        pass


class Fusion(ObjectiveFunction):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores
        self.linear_transform = LinearTransform(
            map=partial(self.map),
            transmap=partial(self.transmap)
        )

    def map(self, w):
        return w[:-1].T @ self.scores + w[-1]

    def transmap(self, w):
        return numpy.hstack((self.scores @ w, w.sum()))

    def objective(self, w):
        return self.linear_transform.objective(w=w)

    def objective_and_gradient(self, w):
        return self.linear_transform.objective_and_gradient(w=w)

    def objective_and_derivatives(self, w):
        return self.linear_transform.objective_and_derivatives(w=w)

    def hessian(self, *args):
        pass

    def hessian_and_jacobian_product(self, *args):
        pass


class GeneralMatrixMultiplication(ObjectiveFunction):
    def __init__(self, m, k, n):
        super().__init__()
        self.m = m
        self.k = k
        self.n = n

    def extractA(self, w):
        A = w[:self.m * self.k].reshape(self.m, self.k, order='F').copy()
        return A

    def extractB(self, w):
        start = self.m * self.k
        B = w[start:(start + self.k * self.n)].reshape(self.k, self.n, order='F').copy()
        return B

    def compute_objective_terms(self, w):
        wf = w.T.flatten()
        A = self.extractA(wf)
        B = self.extractB(wf)
        M = A @ B
        prod = M.T.flatten()
        return prod, A, B

    def objective(self, w):
        y, _, _ = self.compute_objective_terms(w)
        return y

    def compute_gradient_terms(self, g2, A, B):
        M = g2.reshape(self.m, self.n, order= 'F').copy()
        Bp = A.T @ M
        Ap = M @ B.T
        g = numpy.concatenate((Ap.T.flatten(), Bp.T.flatten()))
        return g

    def objective_and_gradient(self, w):
        y, A, B = self.compute_objective_terms(w)

        def deriv(g2, A, B):
            g = self.compute_gradient_terms(g2=g2, A=A, B=B)
            return g

        gradient = partial(deriv, A=A, B=B)
        return y, gradient

    def objective_and_derivatives(self, w):
        y, A, B = self.compute_objective_terms(w)

        def derivatives(g2, A, B):
            g = self.compute_gradient_terms(g2=g2, A=A, B=B)
            linear = False
            hess = partial(self.hessian, g2=g2)
            hess_jacob = partial(self.hessian_and_jacobian_product, g2=g2, A=A, B=B)
            return g, hess, linear, hess_jacob

        gradients = partial(derivatives, A=A, B=B)
        return y, gradients

    def compute_hessian_components(self, dy, g2):
        Ady = self.extractA(dy)
        Bdy = self.extractB(dy)
        h = self.compute_gradient_terms(g2=g2, A=Ady, B=Bdy)
        return h, Ady, Bdy

    def hessian(self, dy, g2):
        h, _, _ = self.compute_hessian_components(dy=dy, g2=g2)
        return h

    def hessian_and_jacobian_product(self, dy, g2, A, B):
        h, Ady, Bdy = self.compute_hessian_components(g2=g2, dy=dy)
        M = Ady @ B + A @ Bdy
        Jv = M.T.flatten()
        return h, Jv


class SolveAXeqB(ObjectiveFunction):
    def __init__(self, m):
        super().__init__()
        # assert(isinstance(m, ))

        self.m = m

    def extract(self, w):
        mm = self.m ** 2
        A = w[:mm].reshape(self.m, self.m, order='F').copy()
        B = w[mm:].reshape(self.m, int((len(w) - mm) / self.m), order='F').copy()
        n = B.shape[1]
        return A, B, n

    def compute_objective_terms(self, w):
        A, B, n = self.extract(w)
        y = numpy.linalg.solve(A, B).T.flatten()
        return y, A, n

    def objective(self, w):
        y, _, _ = self.compute_objective_terms(w=w)
        return y

    def compute_gradient_terms(self, dy, n, A, X):
        assert(isinstance(dy, numpy.ndarray))
        DXt = dy.reshape(self.m, n, order='F').copy()
        DBt = numpy.linalg.solve(A.T, DXt)
        DAt = - DBt @ X.T[None, :]
        g = numpy.concatenate((DAt.T.flatten(), DBt.T.flatten()))
        return g, DBt

    def objective_and_gradient(self, w):
        y, A, n = self.compute_objective_terms(w=w)

        def deriv(dy, n, A, X):
            g, _ = self.compute_gradient_terms(dy=dy, n=n, A=A, X=X)
            return g

        gradient = partial(deriv, n=n, A=A, X=y)
        return y, gradient

    def objective_and_derivatives(self, w):
        y, A, n = self.compute_objective_terms(w=w)

        def derivatives(dy, n, A, X):
            g, DBt = self.compute_gradient_terms(dy=dy, n=n, A=A, X=X)
            linear = False
            hess = partial(self.hessian, A=A, X=X, DBt=DBt)
            hess_jacob = partial(self.hessian_and_jacobian_product, A=A, X=X, DBt=DBt)
            return g, hess, linear, hess_jacob

        gradients = partial(derivatives, n=n, A=A, X=y)

        return y, gradients

    def compute_hessian_terms(self, dy, A, X, DBt):
        dA, dB, _ = self.extract(dy)
        D_DBt = - numpy.linalg.solve(A.T, dA.T) @ DBt
        DX = numpy.linalg.solve(A, (dB - dA @ X[:, None]))
        D_DAt = - (D_DBt @ X.T[None, :] + DBt @ DX.T)
        h = numpy.concatenate((D_DAt.T.flatten(), D_DBt.T.flatten()))
        return h, dA, dB

    def hessian(self, dy, A, X, DBt):
        h, _, _ = self.compute_hessian_terms(dy=dy, A=A, X=X, DBt=DBt)
        return h

    def hessian_and_jacobian_product(self, dy, A, X, DBt):
        h, dA, dB = self.compute_hessian_terms(dy=dy, A=A, X=X, DBt=DBt)
        Jv = numpy.linalg.solve(A, (dB - dA @ X[:, None])).flatten()
        return h, Jv


class LinearTransformAdaptive(ObjectiveFunction):
    def __init__(self, map, transmap):
        super().__init__()
        self.map = map
        self.transmap = transmap

    def objective(self, w):
        y = self.map(w)
        return y

    def objective_and_gradient(self, w):
        y = self.map(w)
        wlen = len(w)
        gradient = partial(self.transmap, wlen=wlen)
        return y, gradient

    def objective_and_derivatives(self, w):
        y = self.map(w)
        wlen = len(w)

        def derivatives(d, wlen):
            g = partial(self.transmap, wlen=wlen)
            linear = True
            hess = self.hessian
            hess_jacob = self.hessian_and_jacobian_product
            return g, hess, linear, hess_jacob

        gradients = partial(derivatives, wlen=wlen)

        return y, gradients

    def hessian(self, d):
        h = None
        return h

    def hessian_and_jacobian_product(self, d):
        h = None
        Jd = self.map(d)
        return h, Jd


class LinearTransform(ObjectiveFunction):
    def __init__(self, map, transmap):
        super().__init__()
        self.map = map
        self.transmap = transmap

    def objective(self, w):
        y = self.map(w)
        return y

    def objective_and_gradient(self, w):
        y = self.map(w)
        gradient = self.transmap
        return y, gradient

    def objective_and_derivatives(self, w):
        y = self.map(w)

        def derivatives(d):
            g = self.transmap(d)
            linear = True
            hess = self.hessian
            hess_jacob = self.hessian_and_jacobian_product
            return g, hess, linear, hess_jacob

        gradients = derivatives

        return y, gradients

    def hessian(self, d):
        h = None
        return h

    def hessian_and_jacobian_product(self, d):
        h = None
        Jd = self.map(d)
        return h, Jd


class CllrObjective(ObjectiveFunction):
    def __init__(self, T, weights, logit_prior=0):
        # logit(prior) = -eta    aka the negated LLR threshold
        super().__init__()
        # assert(isinstance(T, ))
        # assert(isinstance(weights, ))
        # assert(isinstance(logit_prior, ))
        self.T = T
        self.weights = weights
        self.logit_prior = logit_prior

    def compute_objective_terms(self, w):
        scores = w.flatten()
        arg = (scores[..., :] + self.logit_prior) * self.T
        neglogp1 = -numpy.log(sigmoid(arg))
        y = neglogp1 @ self.weights
        return y, arg, neglogp1

    def objective(self, w):
        y, _, _ = self.compute_objective_terms(w=w)
        return y

    def compute_gradient_term(self, dy, logp2):
        g0 = -numpy.exp(logp2) * self.weights * self.T
        g = dy * g0
        return g, g0

    def objective_and_gradient(self, w):
        y, arg, _ = self.compute_objective_terms(w=w)
        neglogp2 = -numpy.log(sigmoid(-arg))

        def derivative(dy, logp2):
            g, _ = self.compute_gradient_term(dy=dy, logp2=logp2)
            return g

        gradient = partial(derivative, logp2=-neglogp2)
        return y, gradient

    def objective_and_derivatives(self, w):
        y, arg, neglogp1 = self.compute_objective_terms(w=w)
        neglogp2 = -numpy.log(sigmoid(-arg))

        def derivatives(dy, logp1, logp2):
            g, g0 = self.compute_gradient_term(dy=dy, logp2=logp2)
            linear = False
            hess = partial(self.hessian, dy=dy, g0=g0, logp1=logp1, logp2=logp2)
            hess_jacob = partial(self.hessian_and_jacobian_product, dy=dy, g0=g0, logp1=logp1, logp2=logp2)
            return g, hess, linear, hess_jacob
        
        gradients = partial(derivatives, logp1=-neglogp1, logp2=-neglogp2)
        return y, gradients

    def hessian(self, d, dy, g0, logp1, logp2):
        h = dy * (numpy.exp(logp1 + logp2) * self.weights * d)
        return h
    
    def hessian_and_jacobian_product(self, d, dy, g0, logp1, logp2):
        h = self.hessian(d=d, dy=dy, g0=g0, logp1=logp1, logp2=logp2)
        Jv = d.T @ g0
        return h, Jv
    

# TODO other objectives: boost, brier, mce, wmlr
# TODO quality


# this class remains unused in BOSARIS but is implemented there in some form (see train_system.m)
class PenaltyFunction(ObjectiveFunction):
    __penalty = None

    @abstractmethod
    def __init__(self, *args):
        super().__init__()
        """ Virtually private constructor. """
        if PenaltyFunction.__penalty is not None:
            raise Exception("This class is a singleton!")
        else:
            PenaltyFunction.__penalty = self

    @abstractmethod
    def objective(self, w):
        pass

    @abstractmethod
    def objective_and_gradient(self, w):
        pass

    @abstractmethod
    def objective_and_derivatives(self, w):
        pass

    @abstractmethod
    def hessian(self, *args):
        pass

    @abstractmethod
    def hessian_and_jacobian_product(self, *args):
        pass


class SumSquaresPenalty(PenaltyFunction):
    def __init__(self, scale):
        super().__init__()
        # assert(isinstance(lambda, ))
        self.scale = scale

    def objective(self, w):
        scale = self.scale
        if numpy.isscalar(scale):
            scale = numpy.ones((len(w), len(w)))

        y = 0.5 * w.T @ scale @ w
        return y

    def objective_and_gradient(self, w):
        scale = self.scale
        if numpy.isscalar(scale):
            scale = numpy.ones((len(w), len(w)))

        scale_w = scale @ w
        y = 0.5 * w.T @ scale_w

        def deriv(dy, scale_w):
            g = dy * scale_w
            return g

        gradient = partial(deriv, scale_w)
        return y, gradient

    def objective_and_derivatives(self, w):
        scale = self.scale
        if numpy.isscalar(scale):
            scale = numpy.ones((len(w), len(w)))

        scale_w = scale * w
        y = 0.5 * w.T @ scale_w

        def deriv(dy, scale, scale_w):
            linear = False
            g = dy @ scale_w
            hess = partial(self.hessian, dy, scale)
            return g, hess, linear

        gradients = partial(deriv, scale, scale_w)
        return y, gradients

    def hessian(self, d, dy, scale):
        h = dy @ scale * d
        return h

    def hessian_and_jacobian_product(self, d, dy, scale, scale_w):
        h = dy @ scale * d
        Jv = d.T @ scale_w
        return h, Jv
