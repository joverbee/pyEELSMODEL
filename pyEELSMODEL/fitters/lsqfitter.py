'''
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
'''
import numpy as np
from scipy.optimize import least_squares

from pyEELSMODEL.core.fitter import Fitter
import logging
logger = logging.getLogger(__name__)

class LSQFitter(Fitter):
    '''

    '''

    def __init__(self, spectrum, model, use_bounds=False, method='lm'):
        '''
        Initialises a Least Squares Fitter instance.

        Parameters
        ----------
        spectrum : Spectrum or Multispectrum
            The experimental data used which needs to be fitted
        model : Model
            The model used to fit the experimental data. Model should only contain
            linear parameters.
        use_bounds: bool, optional
            Bool indicating which if the boundaries of the parameters are used (default:False)

        Returns
        -------
        An instance of a LSQFitter.

        '''

        super().__init__(spectrum, model)
        self.estimator = self.residuals
        self.method = method
        self.use_bounds = use_bounds
        self.use_weights = False

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

    @property
    def use_bounds(self):
        return self._use_bounds

    @use_bounds.setter
    def use_bounds(self, b):
        if self.method == 'lm':
            logger.info(r'lm algorithm cannot use bounds')
            self._use_bounds=False
        else:
            self._use_bounds=b

    def residuals(self, x0, m, s):
        for index, param in enumerate(m.getfreeparameters()):
            param.setvalue(x0[index])
        m.calculate()

        if self.use_weights:
            # weights = 1/(np.sqrt(np.abs(s.data[np.invert(s.exclude)])+0.01))
            # weights = self.weights[np.invert(s.exclude)]
            weights = self.weights[np.invert(s.exclude)]
            # weights = np.sqrt(np.abs(s.data[np.invert(s.exclude)]))
        else:
            weights = np.ones(s.data[np.invert(s.exclude)].size)

        return np.sqrt(weights)*(s.data[np.invert(s.exclude)] - m.data[np.invert(s.exclude)])

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        #todo select which methods can be used an tell which needs to have
        methods = ['trf', 'lm', 'dogbox']
        if method in methods:
            self._method = method
        else:
            raise ValueError(r'Given method does not exist for this function')

    def bounds_lsq(self):
        """
        Boundaries for lsq is different as used for the ml fitter


        :return:
        """
        bounds = self.get_bounds()
        low = []
        up = []
        for bound in bounds:
            low.append(bound[0])
            up.append(bound[1])
        return [tuple(low), tuple(up)]

    def perform_fit(self):

        x0 = self.get_start_parameters()

        for index, param in enumerate(self.model.getfreeparameters()):
            param.setvalue(x0[index])
        self.model.calculate()
        ndata = np.copy(self.spectrum.data)
        ndata[ndata < 1] = 1
        self.weights = 1/ndata


        argo = (self.model, self.spectrum)
        if self.use_bounds:
            bounds = self.bounds_lsq()
        else:
            bounds = (-np.Inf), np.Inf

        resLS = least_squares(self.estimator, x0, args=argo, method=self.method,
                              bounds=bounds)

        self.coeff = resLS.x
        self.error = resLS.cost









