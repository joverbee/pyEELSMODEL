"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from scipy import special
import numpy as np
from pyEELSMODEL.core.parameter import Parameter


class Voigt(Component):
    """
        Initialises a voigt component. The voigt function is the convolution
        of a gaussian with a lorentzian function.


    """
    def __init__(self, specshape, A, centre, gamma, sigma):
        """

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        A : float
            Amplitude of the voigt.

        centre: float
            The position of the peak position

        gamma: float
            The full width half maximum of the gaussian.

        sigma: float
            The standard deviation of the lorentzian.

        Returns
        -------
        """
        super().__init__(specshape)

        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(0, np.inf)
        p1.sethasgradient(False)
        self._addparameter(p1)

        p2 = Parameter('centre', centre)
        p2.setboundaries(-np.inf, np.inf)
        p2.sethasgradient(False)
        self._addparameter(p2)

        p3 = Parameter('gamma', gamma)
        p3.setboundaries(self.dispersion/10, np.inf)
        p3.sethasgradient(False)
        self._addparameter(p3)

        p4 = Parameter('sigma', sigma)
        p4.setboundaries(self.dispersion/10, np.inf)
        p4.sethasgradient(False)
        self._addparameter(p4)

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        E = self.energy_axis
        self.data = self.voigt(E, self.parameters[0].getvalue(),
                               self.parameters[1].getvalue(),
                               self.parameters[2].getvalue(),
                               self.parameters[3].getvalue())

    def voigt(self, x, A, centre, gamma, sigma):
        """
        Voigt function.
        #todo add citation where to find definition of this calculation
        """
        z = ((x-centre) + 1j*gamma)/(np.sqrt(2)*sigma)
        # w_z = np.exp(-z**2)*special.erfc(-1j*z)
        w_z = special.wofz(z)
        V = np.real(w_z)/(sigma*np.sqrt(2*np.pi))
        V[np.isnan(V)] = 0
        V[np.isinf(V)] = 0
        V = V/V.max()
        return A*V
        # return A*np.pi**-1*0.5*dT/((x-x0)**2+(0.5*dT)**2)

    def get_fwhm(self):
        sigma = self.parameters[2].getvalue()
        fg = 2 * sigma * np.sqrt(2 * np.log(2))
        gamma = self.parameters[3].getvalue()
        fl = 2 * gamma
        fwhm = 0.5346 * fl + np.sqrt(0.166 * fl**2+fg**2)
        return fwhm

    def getgradient(self, parameter):
        return None

# todo add gradients
