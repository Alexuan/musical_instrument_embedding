from pybosaris.calibration.objectives import Fusion, LinearTransformAdaptive
import numpy


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__license__ = "LGPLv3"
__credits__ = ["Niko Brummer", "Edward de Villiers"]


class LinearFuser(object):
    def __init__(self, scores, w=None):
        self.w = w

        if numpy.ndim(scores) == 1:
            scores = scores[None, :]
        self.scores = scores

        wsz = scores.shape[0] + 1
        whead, wtail = self.splitvec_fh(wsz, w)
        if w is None:
            # params.get_w0 = @() numpy.zeros(wsz, 1)
            # % params.get_w0 = @() randn(wsz, 1);
            self.w = numpy.zeros(wsz)

        self.head = whead
        self.tail = wtail
        self.fusion_obj = Fusion(scores=scores)  # Fusion(whead, scores)

    def fusion(self):
        return self.fusion_obj.objective(self.head)

    @staticmethod
    def splitvec_fh(head_size, w=None ):
        # If head_size <0 then tail_size = - head_size

        tail_size = -head_size

        if head_size > 0:
            def transmap_head(y, wlen):
                w = numpy.zeros(wlen)
                w[:head_size] = y
                return w

            def transmap_tail(y, wlen):
                w = numpy.zeros(wlen)
                w[head_size:] = y
                return w

            def map_head(w):
                return w[:head_size]

            def map_tail(w):
                return w[head_size:]

        elif head_size < 0:
            def transmap_head(y, wlen):
                w = numpy.zeros(wlen)
                w[:-tail_size] = y
                return w

            def transmap_tail(y, wlen):
                w = numpy.zeros(wlen)
                w[-tail_size:] = y
                return w

            def map_head(w):
                return w[:-tail_size]

            def map_tail(w):
                return w[-tail_size:]

        else:
            raise Exception('head size cannot be 0')

        if w is not None:
            head = LinearTransformAdaptive(map=map_head, transmap=transmap_head).objective(w)
            tail = LinearTransformAdaptive(map=map_tail, transmap=transmap_tail).objective(w)
        else:
            head = LinearTransformAdaptive(map=map_head, transmap=transmap_head)
            tail = LinearTransformAdaptive(map=map_tail, transmap=transmap_tail)

        return head, tail
