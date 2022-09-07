from numpy import linspace, log, vstack, minimum, repeat, flipud, exp, zeros, isinf, infty
from pybosaris.libmath import sigmoid, optimal_llr_from_Popt, optimal_llr
from pybosaris.libperformance import fast_actDCF, ece, rocch_pava, cllr
from tikzplotlib import save as tikz_save
from copy import deepcopy as copy
import matplotlib.pyplot as mpl


class PriorLogOddsPlots(object):
    def __init__(self, tar=None, non=None, plo=linspace(-10, 10, 2001)):
        self.plo = plo
        self.legend_ECE = ['default']
        if tar is not None and non is not None:
            self.set_system(tar=tar, non=non)

    def add_legend_ece_entry(self, entry):
        self.legend_ECE.append(entry)

    def set_system(self, tar, non):
        self.tar = tar
        self.non = non

        pmiss, pfa, Popt, perturb = rocch_pava(self.tar, self.non, laplace=False)
        self.Pmiss = pmiss
        self.Pfa = pfa

        Ptar = sigmoid(self.plo)
        Pnon = sigmoid(-self.plo)
        self.defDCF = minimum(Ptar, Pnon)

        cdet = vstack([[Ptar, Pnon]]).T @ vstack((self.Pmiss, self.Pfa))
        self.minDCF = cdet.min(axis=1)
        self.eer = self.minDCF.max()

        tar_llr, non_llr = optimal_llr_from_Popt(Popt, perturb, Ntar=len(self.tar), Nnon=len(self.non))
        self.tar_llr = tar_llr
        self.non_llr = non_llr

        tar_llr_laplace, non_llr_laplace = optimal_llr(tar=tar, non=non, laplace=True)
        self.tar_llr_laplace = tar_llr_laplace
        self.non_llr_laplace = non_llr_laplace

    def save(self, filename, type='tikz', dpi=None, width='80pt', height='40pt'):
        if type == 'pdf':
            mpl.savefig(filename + '.pdf')
        elif type == 'png':
            mpl.savefig(filename + '.png')
        elif type == 'tikz':
            self.__save_as_tikzpgf__(outfilename=filename, dpi=dpi, width=width, height=height)
        else:
            raise ValueError('unknown save format')

    def __save_as_tikzpgf__(self, outfilename,
                            dpi=None,
                            width='80pt',
                            height='40pt',
                            extra_tikzpicture_parameters=['[font=\\scriptsize]']):
        # see: https://codeocean.com/algorithm/154591c8-9d3f-47eb-b656-3aff245fd5c1/code

        def replace_tick_label_notation(tick_textpos):
            tick_label = tick_textpos.get_text()
            if 'e' in tick_label:
                tick_label = int(tick_label.replace('1e', '')) - 2
                tick_textpos.set_text('%f' % (10**(int(tick_label)-2)))

        if dpi is not None:
            mpl.gcf().set_dpi(dpi)

        mpl.gca().set_title('')

        for tick_textpos in mpl.gca().get_xmajorticklabels():
            replace_tick_label_notation(tick_textpos)
        for tick_textpos in mpl.gca().get_ymajorticklabels():
            replace_tick_label_notation(tick_textpos)

        tikz_save(outfilename,
                  figurewidth=width,
                  figureheight=height,
                  extra_tikzpicture_parameters=extra_tikzpicture_parameters)

    def plot_dcf(self, normalize=False, plot_err=False, flip_xaxis=False):
        actDCF = fast_actDCF(self.tar, self.non, self.plo, normalize=normalize)
        eer = repeat(self.eer, len(self.plo))

        defDCF = copy(self.defDCF)
        minDCF = copy(self.minDCF)
        if normalize:
            minDCF /= defDCF
            actDCF /= defDCF
            eer /= defDCF
            defDCF /= defDCF

        mpl.figure("DCF")

        if flip_xaxis:
            # plo remains as is, we just flip the other arrays
            mpl.xlabel('LLR threshold')
            defDCF = flipud(defDCF)
            minDCF = flipud(minDCF)
            actDCF = flipud(actDCF)
            eer = flipud(eer)
        else:
            mpl.xlabel('logit(\\tilde\\pi)')

        if not normalize:
            mpl.ylabel('DCF')
        else:
            mpl.ylabel('NBER')

        mpl.plot(self.plo, defDCF, 'k', linewidth=2)
        mpl.plot(self.plo, minDCF, 'k--', linewidth=2)
        mpl.plot(self.plo, actDCF, 'g', linewidth=1)
        if plot_err:
            mpl.plot(self.plo, eer, 'k:')
        mpl.ylim([0, 1.4 * max(0.5, actDCF.min())])

    def plot_ece(self, normalize=False, display_actual=True, line_style_minECE='b--'):
        defECE = (sigmoid(self.plo) * -log(sigmoid(self.plo)) + sigmoid(-self.plo) * -log(sigmoid(-self.plo))) / log(2)
        minECE = ece(tar=self.tar_llr, non=self.non_llr, plo=self.plo)
        actECE = ece(tar=self.tar, non=self.non, plo=self.plo)

        if normalize:
            minECE /= defECE
            actECE /= defECE
            defECE /= defECE

        mpl.figure("ECE")

        mpl.xlabel('logit(\\pi)')

        if not normalize:
            mpl.ylabel('ECE')
        else:
            mpl.ylabel('NECE')

        if len(self.legend_ECE) == 1:
            mpl.plot(self.plo, defECE, 'k', linewidth=2)
        mpl.plot(self.plo, minECE, line_style_minECE, linewidth=2)
        if display_actual:
            mpl.plot(self.plo, actECE, 'r', linewidth=1)
            mpl.ylim([0, 1.4 * max(1, actECE.min())])
        else:
            mpl.ylim([0, 1.4])

    def get_delta_DCF(self):
        return 1 - cllr(self.tar_llr, self.non_llr)

    def get_delta_ECE(self):
        def int_ece(x):
            idx = (~isinf(x)) & (x != 0)
            contrib = zeros(len(x))
            contrib[x == infty] = 0.25
            xx = x[idx]
            LR = exp(xx)
            contrib[idx] = (LR**2 - 4*LR + 2*xx + 3) / (4*(LR - 1)**2)
            LRm1 = exp(xx) - 1
            contrib[idx] = 0.25 - 1/(2*LRm1) + xx / (2*LRm1**2)
            return contrib.mean()

        int_diff_ece = int_ece(self.tar_llr) + int_ece(-self.non_llr)
        return int_diff_ece / log(2)
