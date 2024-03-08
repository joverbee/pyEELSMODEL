"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import numpy as np
from scipy.optimize import minimize

from pyEELSMODEL.core.fitter import Fitter
import logging
logger = logging.getLogger(__name__)


class MinimizeFitter(Fitter):
    """
    Fitter class which uses the minimization algorithms of scipy for
    finding the optimal parameters. Mainly for the maximum likelihood
    estimator but others can also be implemented.

    """

    def __init__(self, spectrum, model, method='nelder-mead', estimator='ML'):
        super().__init__(spectrum, model)
        self.estimator = estimator
        self.method = method

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        if estimator == 'ML':
            self._estimator = self.MLE_function
            self.jacobian = self.jacobian_ML

        # if estimator is 'ML' and not self.model.hasconvolutor():
        #     self._estimator = self.MLE_function
        #     self.jacobian = self.jacobian_ML

        # elif estimator is 'ML' and self.model.hasconvolutor():
        #     self._estimator = self.MLE_function
        #     self.jacobian = '2-point'

        elif estimator == 'LSQ':
            self._estimator = self.lsq_function
            self.jacobian = '2-point'
        else:
            logger.warning(r'Estimator is not known, maximum'
                           r' likelihood is chosen')
            self._estimator = self.MLE_function
            self.jacobian = self.jacobian_ML

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        self._method = method

    @property
    def jacobian(self):
        """
        :return:
        """
        return self._jacobian

    @jacobian.setter
    def jacobian(self, jacobian):
        """
        Only when the gradients are used, the jacobian will be used.
        :param jacobian:
        :return:
        """
        if self.usegradients:
            self._jacobian = jacobian
        else:
            self._jacobian = '2-point'

    def MLE_function(self, x0, m, s):
        """
        The maximum likelihood estimator
        :param x0:
        :param m:
        :param s:
        :return:
        """
        for index, param in enumerate(m.getfreeparameters()):
            param.setvalue(x0[index])
        m.calculate()

        if any(m.data < 0):
            return np.inf

        # boolean1 = (s.data > 1.) & (m.data > 1.)
        # when the experimental dat is smaller than zero, then the
        # stirling approx will give wrong result
        boolean1 = m.data > 0
        boolean2 = np.invert(s.exclude)
        boolean = np.logical_and(boolean1, boolean2)

        # paper where factor two is used
        # (https://doi.org/10.1016/S0168-9002(00)00756-7)
        ML = 2*np.sum(s.data[boolean] * np.log(m.data[boolean])
                      - m.data[boolean])
        return -ML

    def jacobian_ML(self, x0, m, s):
        # todo can still have errors
        for index, param in enumerate(m.getfreeparameters()):
            param.setvalue(x0[index])
        m.calculate()

        if any(m.data < 0):
            return np.inf

        boolean1 = m.data > 0
        boolean2 = np.invert(s.exclude)
        boolean = np.logical_and(boolean1, boolean2)

        jac = []
        for index, param in enumerate(m.getfreeparameters()):
            comp = m.getcomponentbyparameter(param)
            gradient = comp.getgradient(param)
            j = 2 * np.sum((1 - s.data[boolean]/m.data[boolean])
                           * (gradient[boolean]))
            jac.append(j)

        return jac

    def jacobian_LSQ(self, x0, m, s):
        # todo needs to be implemented

        return 1

    def lsq_function(self, x0, m, s):
        """
        Least squares minimization, preferably use LSQFitter
        """
        for index, param in enumerate(m.getfreeparameters()):
            param.setvalue(x0[index])
        m.calculate()
        inv_ = np.invert(s.exclude)
        res = np.sum(np.abs((s.data[inv_] - m.data[inv_])))
        return res

    def perform_fit(self):
        """
        Does the fit on a single spectrum where the starting parameters are
        given by x0.

        """
        x0 = self.get_start_parameters()
        bounds = self.get_bounds()
        argo = (self.model, self.spectrum)

        resML = minimize(self.estimator, x0, args=argo, method=self.method,
                         tol=1e-12, bounds=bounds, jac=self.jacobian)

        self.coeff = resML.x
