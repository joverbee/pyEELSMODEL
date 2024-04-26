"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import nnls

from pyEELSMODEL.core.fitter import Fitter
from pyEELSMODEL.components.MScatter.mscatter import Mscatter


class LinearFitter(Fitter):
    """
    A linear fitter class which uses as input a Spectrum or Multispectrum and
    Model.
    """

    def __init__(self, spectrum, model, method='ols', use_weights=False):

        """
        Initialises a LinearFitter instance. Two different methods can be
        used which is the ordinary least squares (ols) or the non-negative
        least squares (nnls). The nnls makes that the values of the fitted
        parameters cannot become negative.

        Parameters
        ----------
        spectrum : Spectrum or Multispectrum
            The experimental data used which needs to be fitted
        model : Model
            The model used to fit the experimental data. Model should only
            contain linear parameters.
        method: str, optional
            String indicating which linear fitting procedure should be used. .
            Available fitters are: ols and nnls
        use_weights: boolean
            Indicating if the weights are being used during the fit. The
            weigths assume poisson noise.

        Returns
        -------
        An instance of a LinearFitter.

        """
        # model can only be check if it is add
        if not model.islinear():
            raise ValueError(r'There are non-linear parameters in the model')

        super().__init__(spectrum, model)

        self.method = method
        self.A_matrix = None
        self.y_vector = None
        # a list where each component of the A matrix column is saved
        self.component_list = []
        self.same_A_matrix = True  # use the same A matrix all the time
        self.use_weights = use_weights

    @property
    def A_matrix(self):
        return self._A_matrix

    @A_matrix.setter
    def A_matrix(self, A):
        self._A_matrix = A

    @property
    def y_vector(self):
        return self._y_vector

    @y_vector.setter
    def y_vector(self, y):
        self._y_vector = y

    def calculate_A_matrix(self):
        """
        Get derivative matrix. This is a matrix which holds for each energy in
        the spectrum the first derivative of the value at that energy with
        respect to each free parameter in the model.
        This A matrix needs to be convolved to include the low loss scattering

        """
        ncol = self.model.getnumfreeparameters()
        A = np.zeros((self.spectrum.size, ncol))
        component_list = []

        for i, param in enumerate(self.model.getfreelinparameters()):
            # put all parameters to 0 except 1
            for par in self.model.getfreelinparameters():
                if par == param:
                    # this could cause trouble as it ignores any scaling issues
                    par.setvalue(1)
                else:
                    par.setvalue(0)
            self.model.calculate(use_ll=False)
            component_list.append(self.model.getcomponentbyparameter(param))
            A[:, i] = self.model.data

        self.A_matrix = A
        self.component_list = component_list

    def convolute_A_matrix(self):
        """
        Convolutes the A matrix with the low loss. This is necessary to
        take the low loss into account. It does this by first checking which
        components need to be convolved. The next step is to convolve each
        row of the A matrix with low loss.
        This low loss convolution increases the fitting time. Other and faster
        ways can be exploited to have a faster quantification.

        Returns
        -------
        A_matrix_ll: numpy 2d array
            The convolved A matrix with the low loss.

        """

        for comp in self.model.components:
            if isinstance(comp, Mscatter):
                ll_comp = comp

        A_matrix_ll = np.empty(self.A_matrix.shape)
        ll_comp.new_ll = True
        for i in range(self.A_matrix.shape[1]):
            if self.component_list[i].getcanconvolute():
                ndata = self.A_matrix[:, i]
                ll_comp.data = ndata
                ll_comp.calculate()
                ll_comp.new_ll = False
                A_matrix_ll[:, i] = ll_comp.data
            else:
                A_matrix_ll[:, i] = self.A_matrix[:, i]
        ll_comp.new_ll = True
        return A_matrix_ll

    def convolute_A_matrix_(self):
        """
        Test convolution using the matrix approach however this seems
        to be slower than the original which loops over A matrix
        """
        for comp in self.model.components:
            if isinstance(comp, Mscatter):
                ll_comp = comp

        A_matrix_ll = ll_comp.calculate_A_matrix(self.A_matrix)
        for i in range(self.A_matrix.shape[1]):
            if not self.component_list[i].getcanconvolute():
                A_matrix_ll[:, i] = self.A_matrix[:, i]
        return A_matrix_ll

    def calculate_y_vector(self):
        """
        Calculates the y vector for the linear least squares fitting.
        The y vector is the raw data where the excluded regions are not
        included.

        Returns
        -------
        y: numpy array
            The y vector for the linear least squares fitting

        """
        y = self.spectrum.data[np.invert(self.spectrum.exclude)]
        self.y_vector = y

    def calculate_weights(self):
        """
        The weights are calculated assuming the data is poisson noise.
         W=1/(sigma**2), and sigma**2 = N which is the number of electrons.
         If the data itself is lower than 1, which could happen due to dark
         noise, then the value is set to 1. Note that the data itself needs
         to be expressed in electron (right pppc) for it to make most sense.

         Returns
        -------
        weights: numpy array
            The weigths used in the fitting procedure for poisson noise.

        """
        ndata = np.copy(self.y_vector)
        ndata[ndata < 1] = 1
        weigths = 1/ndata
        return weigths

    def perform_fit(self):
        """
        Fits the spectrum using the linear least squares method.
        """

        if self.A_matrix is None or \
                self.model.ischanged() or \
                not self.same_A_matrix:
            self.calculate_A_matrix()  # makes it slow
            self.model.setchanged(False)

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

        if self.method == 'nnls':
            try:
                coeff, cost = nnls(AW, yW)
                self.coeff = coeff
                self.error = cost
            except Exception:
                self.coeff = np.nan
                self.error = np.nan

        elif self.method == 'ols':
            coeff, resi, rank, sing = np.linalg.lstsq(AW, yW, rcond=-1)
            self.coeff = coeff
            if resi.size == 0:
                self.error = np.inf
            else:
                self.error = resi[0]

        else:
            print('Fitting method is wrong')

    def pearson_correlation_matrix(self):
        """
        Calculates the pearson correlation matrix
        Returns
        -------
        correlation_matrix: 2d numpy array
            The pearson correlation matrix
        """
        self.calculate_A_matrix()
        A = self.A_matrix

        correlation_matrix = np.zeros((A.shape[1], A.shape[1]))
        for i in range(A.shape[1]):
            for j in range(A.shape[1]):
                r = stats.pearsonr(A[:, i], A[:, j])[0]
                correlation_matrix[i, j] = r

        return correlation_matrix

    def show_pearson_correlation(self):
        correlation = self.pearson_correlation_matrix()
        label_list = self.getlabelist()
        fig, ax = plt.subplots()
        ax.imshow(correlation, vmin=-1, vmax=1, cmap='bwr')
        ax.set_xticks(np.arange(len(label_list)))
        ax.set_yticks(np.arange(len(label_list)))
        ax.set_xticklabels(label_list, rotation=45, ha="right")
        ax.set_yticklabels(label_list, rotation=45)
        fig.set_tight_layout(True)

    def partial_derivative(self, parameter):
        """
        Calculates the partial derivative of the parameter. When the entire
        model is linear. This is taking the rows (or columns) of the A matrix
        which needs to be convoluted when low loss is used.

        Parameters
        ----------
        parameter: Parameter
            The parameter from which the partial derivative will be
            calculated.
        fraction: float
            The fraction of the step the parameter does to calculate
            the numerical gradient. (default: 0.001)

        Returns
        -------
        parial: numpy array (1D)
            The partial derivative for the given parameter.
        """

        if self.model.hasconvolutor():
            A_matrix = self.convolute_A_matrix()
        else:
            A_matrix = self.A_matrix

        index = self.get_param_index(parameter)
        return A_matrix[np.invert(self.spectrum.exclude), index]
