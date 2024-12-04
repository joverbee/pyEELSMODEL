"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np


class Gaussian(Component):
    """
    A Gaussian component.

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model

    A : float
        Amplitude of the gaussian function.

    centre: float
        The peak position of the gaussian function.

    fwhm: float
        The full width halve maximum of the gaussian function.

    Returns
    -------
    """

    def __init__(self, specshape, A, centre, fwhm):
        super().__init__(specshape)

        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(0, np.inf)
        p1.sethasgradient(True)
        self._addparameter(p1)

        p2 = Parameter('centre', centre)
        p2.setlinear(False)
        p2.sethasgradient(True)
        self._addparameter(p2)

        p3 = Parameter('fwhm', fwhm)
        p3.setlinear(False)
        p3.sethasgradient(True)
        p3.setboundaries(self.dispersion / 10,
                         np.inf)  # fwhm << dispersion makes no sense
        self._addparameter(p3)

        self._setname('Gaussian')

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p3 = self.parameters[2]
        if p1.ischanged() or p2.ischanged() or p3.ischanged():
            A = p1.getvalue()
            centre = p2.getvalue()
            fwhm = p3.getvalue()
            sigma = np.abs(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            self.data = self.gaussian_function(A, centre, sigma)
        self.setunchanged()  # put parameters to unchanged

    def gaussian_function(self, A, centre, sigma):
        # if sigma<self.dispersion: #sigma < dispersion makes no sense
        #    sigma=self.dispersion
        return A * np.exp(-0.5 * ((self.energy_axis - centre) / sigma) ** 2)

    def getgradient(self, parameter):
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p3 = self.parameters[2]
        A = p1.getvalue()
        centre = p2.getvalue()
        fwhm = p3.getvalue()
        sigma = np.abs(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        en = self.energy_axis

        if parameter == p1:
            self.gradient[0] = self.gaussian_function(1, centre, sigma)
            return self.gradient[0]
        if parameter == p2:
            self.gradient[1] = ((en - centre) / sigma ** 2) \
                               * self.gaussian_function(A, centre, sigma)
            return self.gradient[1]
        if parameter == p3:
            self.gradient[2] = (en - centre) ** 2 \
                / (sigma ** 2 * np.abs(fwhm)) \
                * self.gaussian_function(A, centre, sigma)
            return self.gradient[2]

        return None
