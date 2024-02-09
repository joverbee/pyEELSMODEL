'''
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
'''
from pyEELSMODEL.core.component import Component
from scipy import special
import numpy as np
from pyEELSMODEL.core.parameter import Parameter


class Voigt(Component):
    """
        Initialises a voigt component. The voigt function is the convolution of a
        gaussian with a lorentzian function.

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
    def __init__(self, specshape, A, centre, gamma, sigma):
        super().__init__(specshape)

        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(0, np.Inf)
        p1.sethasgradient(False)
        self._addparameter(p1)

        p2 = Parameter('centre', centre)
        p2.setboundaries(-np.Inf, np.Inf)
        p2.sethasgradient(False)
        self._addparameter(p2)

        p3 = Parameter('gamma', gamma)
        p3.setboundaries(self.dispersion/10, np.Inf)
        p3.sethasgradient(False)
        self._addparameter(p3)

        p4 = Parameter('sigma', sigma)
        p4.setboundaries(self.dispersion/10, np.Inf)
        p4.sethasgradient(False)
        self._addparameter(p4)

    def calculate(self):
        E = self.energy_axis
        self.data = self.voigt(E, self.parameters[0].getvalue(), self.parameters[1].getvalue(),
                                 self.parameters[2].getvalue(), self.parameters[3].getvalue())

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
        fg = 2*sigma*np.sqrt(2*np.log(2))
        gamma = self.parameters[3].getvalue()
        fl = 2*gamma
        fwhm = 0.5346*fl+ np.sqrt(0.166*fl**2+fg**2)
        return fwhm




    #todo implement the analytical gradient of this exsits.

    # def getgradient(self, parameter):
    #     # to think about, we should only calculate gradients if parameters have changed
    #     # but after calculate the params are set to unchanged
    #     # it could make sense to always calculate the gradient whenever we calculate
    #     p1 = self.parameters[0]
    #     p2 = self.parameters[1]
    #     p3 = self.parameters[2]
    #     A = p1.getvalue()
    #     centre = p2.getvalue()
    #     FWHM = p3.getvalue()
    #
    #     en = self.energy_axis
    #     denom = ((en - centre) ** 2 + (0.5 * FWHM) ** 2)
    #
    #     if parameter == p1:
    #         self.gradient[0] = (FWHM ** 2 / 4) * (1 / denom)
    #         return self.gradient[0]
    #     if parameter == p2:
    #         self.gradient[1] = (A * FWHM ** 2 / 2) * (en - centre) / denom ** 2
    #         return self.gradient[1]
    #     if parameter == p3:
    #         self.gradient[2] = 0.5 * A * FWHM * (1 / denom - (FWHM ** 2 / 4) * (1 / denom ** 2))
    #         return self.gradient[2]
    #
    #     return None

# todo add gradients
# make dE and A independent, if dE scales, the peak height shouldnt change, fitter will be more stable
