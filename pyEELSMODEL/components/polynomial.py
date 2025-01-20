"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Polynomial(Component):
    """
    A Polynomial component. The constant value of the polynomials
    are all set to 1 as initial value. The coefficients of the powers are
    normalized since they produce numerical results because the x-axis
    is useually expressed in eV.

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model
    order : uint
        The order of the polynomial.

    Returns
    -------
    """

    def __init__(self, specshape, order=1):
        """

        :param specshape:
        :param order:
        """
        super().__init__(specshape)
        self.setname('Polynomial')
        self.order = order + 1

        for i in range(self.order):
            name = 'x' + str(self.order - i - 1)
            # changed
            p = Parameter(name, 1)
            p.setlinear(True)
            p.sethasgradient(True)
            self._addparameter(p)

        self._setname('Polynomial')

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        is_changed = False
        for param in self.parameters:
            if param.ischanged():
                is_changed = True

        if is_changed:
            coeff = []
            for param in self.parameters:
                coeff.append(param.getvalue())

            # normalize the coefficients since they will produce
            # numerical problems
            coeffn = np.zeros(len(coeff))
            for i in range(coeffn.size):
                coeffn[i] = coeff[i] / (
                            self.energy_axis ** (self.order - i - 1)).sum()

            p = np.poly1d(coeffn)
            self.data = p(self.energy_axis)
        self.setunchanged()  # put parameters to unchanged

    def getgradient(self, parameter):
        for index, param in enumerate(self.parameters):
            if param is parameter:
                param_order = len(self.parameters) - (index + 1)

        if param_order is None:
            logger.info('Parameter is non existing in this component')

        return self.energy_axis ** (param_order)
