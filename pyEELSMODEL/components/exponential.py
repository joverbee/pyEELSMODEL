"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Exponential(Component):
    """
        Initialises an Exponential component

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        A : float
            Amplitude of the exponential

        b: float
            The constant in the exponential

        Returns
        -------

    """

    def __init__(self, specshape, A, b):
        super().__init__(specshape)

        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(0, np.inf)
        p1.sethasgradient(True)
        self._addparameter(p1)

        p2 = Parameter('b', b)
        p2.setlinear(False)
        p2.sethasgradient(True)
        p2.setboundaries(np.inf, 0)

        self._addparameter(p2)

        self._setname('Exponential')

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        if p1.ischanged() or p2.ischanged():
            A = p1.getvalue()
            b = p2.getvalue()
            self.data = self.exponential_function(A, b)
        self.setunchanged()  # put parameters to unchanged

    def exponential_function(self, A, b):
        # if sigma<self.dispersion: #sigma < dispersion makes no sense
        #    sigma=self.dispersion
        return A * np.exp(b * self.energy_axis)

    def autofit(self, spectrum, istart, istop):
        """
        Perform a linear least square fitting to get an estimate of the
        exponential fit.

        Parameters
        ----------
        spectrum: Spectrum
            The spectrum used to get the estimate from
        istart: uint
            The starting index over which to get the linear least square fit
        istop: uint
            The ending index over which to get the linear least square fit

        """
        if istart > istop:
            logger.warning(r'Start index is larger than stop index')
            return

        subdata = spectrum.data[istart:istop]
        boolean = subdata >= 1.
        ln_y = np.log(subdata[boolean])

        A = np.zeros((boolean.sum(), 2))
        A[:, 0] = np.ones(boolean.sum())
        A[:, 1] = self.energy_axis[istart:istop][boolean]
        coeff, resi, rank, sing = np.linalg.lstsq(A, ln_y, rcond=-1)
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p1.setvalue(np.exp(coeff[0]))
        p2.setvalue(coeff[1])
        self.calculate()

    def getgradient(self, parameter):
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        A = p1.getvalue()
        b = p2.getvalue()
        en = self.energy_axis

        if parameter == p1:
            return self.exponential_function(1, b)
        if parameter == p2:
            return en * self.exponential_function(A, b)

        return None
