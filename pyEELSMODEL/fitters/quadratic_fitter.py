import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.components.fixedpattern import FixedPattern
from pyEELSMODEL.components.linear_background import LinearBG
from pyEELSMODEL.components.constrained_gdoslin import ConstrainedGDOSLin
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.components.powerlaw import PowerLaw


class QuadraticFitter(LinearFitter):
    '''
    A linear fitter class which uses as input a Spectrum or Multispectrum and
    Model.
    '''

    def __init__(self, spectrum, model, use_weights=True):

        '''
        Initialises a QuadraticFitter instance.

        Parameters
        ----------
        spectrum : Spectrum or Multispectrum
            The experimental data used which needs to be fitted
        model : Model
            The model used to fit the experimental data. Model should only
            contain linear parameters. The background should also contain the
            right background parameters

        Returns
        -------
        An instance of a QuadraticFitter.

        '''
        # model can only be check if it is add
        if not model.islinear():
            raise ValueError(r'There are non-linear parameters in the model')

        super().__init__(spectrum, model, use_weights=use_weights)

        self.G_matrix = None
        self.h_vector = None
        self.eq_idx = []
        self.use_constrain_coreloss = True  # chose to let the coreloss model
        # to be positive constrained

    @property
    def G_matrix(self):
        return self._G_matrix

    @G_matrix.setter
    def G_matrix(self, G):
        self._G_matrix = G

    @property
    def h_vector(self):
        return self._h_vector

    @h_vector.setter
    def h_vector(self, h):
        self._h_vector = h

    def _linearfit(self, x, y):
        """
        Estimates the linear fit from x and y

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        n = x.size
        noemer_a = n*np.sum(x**2) - np.sum(x)**2

        a = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/noemer_a
        b = 1/n*(np.sum(y) - a*np.sum(x))

        return a*x+b

    def calculate_weights(self, eps=1.):
        """
        Small function which estimates the weigths used in the quadratic
        programming. It tries to fit a powerlaw to the spectrum and uses this
        as input. If that does not work, a linear fit is performed which used
        this result as input for the weights.

        Args:
            eps (float):Error terms added to weigths. Defaults to 1..

        Returns:
            weigths (np.array): Weights used in the quadratic programming
            describing poisson process
        """
        powlaw_bg = PowerLaw(self.spectrum.get_spectrumshape(), 1., 3.)
        # Pass dummies: A = 1. and r = 3.
        check = powlaw_bg.autofit(self.spectrum, 0, self.spectrum.size)
        ndata = powlaw_bg.data

        if not check:
            # do linear fit on data
            x = self.spectrum.energy_axis
            y = self.spectrum.data
            ndata = self._linearfit(x, y)
            ndata[ndata < 0] = 0

        # powlaw_bg.autofit_twowindow(self.spectrum, 0, self.spectrum.size)
        # Quick and dirty fit. ~20% faster overall.

        msk_include = np.logical_not(self.spectrum.exclude)
        weights = ndata[msk_include]

        return 1. / (weights + eps)

    def perform_fit(self):
        Am = self.A_matrix
        if Am is None or self.model.ischanged() or not self.same_A_matrix:
            self.calculate_A_matrix()  # makes it slow
            self.model.setchanged(False)
            self.get_quadratic_bounds()

        if self.model.hasconvolutor():
            A = self.convolute_A_matrix()[np.invert(self.spectrum.exclude), :]
        else:
            A = self.A_matrix[np.invert(self.spectrum.exclude), :]

        self.calculate_y_vector()
        y = self.y_vector

        if self.use_weights:
            weights = self.calculate_weights()
            tmp = np.sqrt(weights)
            AW = A * tmp[:, np.newaxis]
            yW = y * tmp
        else:
            AW = A
            yW = y

        G = self.G_matrix
        h = self.h_vector

        P = np.dot(np.transpose(AW), AW)
        q = -np.dot(np.transpose(AW), yW)
        result = solve_qp(P, -q, G.T, h, self.number_equality)[0]

        self.coeff = result

    def get_quadratic_bounds(self):
        """
        another method to get the boundaries
        :return:
        """
        if self.A_matrix is None:
            raise ValueError(r'A matrix should first be calculated before G'
                             r' can be determined')

        # following the convention of the quadratic programming, the equality
        # constrains should
        # be placed before the inequality ones

        inequality_arrays = []  # array stores the inequality conditions
        equality_arrays = []  # array stores the equality conditions

        used_comp = []  # list to make sure that the same component with
        # different
        # parameters does not contribute redundant rows to the constrains
        for param in (self.model.getfreelinparameters()):
            comp = self.model.getcomponentbyparameter(param)

            already_used = comp in used_comp
            is_bg = (type(comp) is LinearBG) & (not already_used)
            is_constrained_gdoslin = (type(comp) is ConstrainedGDOSLin) & (not already_used)

            is_coreloss = (isinstance(comp, CoreLossEdge)) & self.use_constrain_coreloss
            is_fixed = (isinstance(comp, FixedPattern)) & self.use_constrain_coreloss

            if is_coreloss or is_fixed:
                cst_array = np.zeros((1, self.A_matrix.shape[1]))
                indey = self.get_param_index(param)
                cst_array[:, indey] = 1
                used_comp.append(comp)
                inequality_arrays.append(cst_array)

            elif is_bg:
                indey = self.get_param_index(param)
                bgG = comp.get_convex_boundaries()
                # if no constrains are given it should skip the rest
                if bgG is None:
                    continue
                cst_array = np.zeros((bgG.shape[0], self.A_matrix.shape[1]))
                # cst_array[:, indey:indey+bgG.shape[1]] = bgG
                # used_comp.append(comp)
                inequality_arrays.append(cst_array)

            elif is_constrained_gdoslin:
                indey = self.get_param_index(param)

                g_mtrx = comp.get_equality_constraints()
                cst_array = np.zeros((g_mtrx.shape[0], self.A_matrix.shape[1]))
                cst_array[:, indey:(indey + g_mtrx.shape[1])] = g_mtrx
                used_comp.append(comp)
                equality_arrays.append(cst_array)

        if len(equality_arrays) == 0:
            self.G_matrix = np.concatenate(inequality_arrays, axis=0)
            self.number_equality = 0

        elif len(inequality_arrays) == 0:
            self.G_matrix = np.concatenate(equality_arrays, axis=0)
            self.number_equality = self.G_matrix.shape[0]

        else:
            eq = np.concatenate(equality_arrays, axis=0)
            ineq = np.concatenate(inequality_arrays, axis=0)
            self.G_matrix = np.concatenate((eq, ineq), axis=0)
            self.number_equality = eq.shape[0]

        self.h_vector = np.zeros(self.G_matrix.shape[0])

    def plot_conv(self):
        """
        Shows the convoluted components of the model.
        """
        conv_A_matrix = self.convolute_A_matrix()
        self.set_fit_values()
        indices = [0]
        names = []
        for comp in self.model.components:
            indices.append(len(comp.getfreelinparameters())+indices[-1])
            names.append(comp.name)

        print(indices)

        fig, ax = plt.subplots()
        Eax = self.spectrum.energy_axis
        ax.plot(Eax, self.spectrum.data, color='black',
                label='Experimental data')
        for ii in range(len(indices)-1):
            prod = conv_A_matrix[:, indices[ii]:indices[ii+1]]\
                   * self.coeff[indices[ii]:indices[ii+1]]
            ax.plot(Eax, prod.sum(1), label=names[ii])
        ax.plot(Eax, self.model.data, label='Model')
        ax.legend()
