"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
import numpy as np
from pyEELSMODEL.core.parameter import Parameter


class Lorentzian_sq(Component):
    def __init__(self, specshape, A, centre, fwhm):
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

        p3 = Parameter('FWHM', fwhm)
        p3.setboundaries(0, np.Inf)
        p3.sethasgradient(False)
        self._addparameter(p3)

    def calculate(self):
        E = self.energy_axis
        self.data = self.lorentz_sq(E, self.parameters[0].getvalue(), self.parameters[1].getvalue(),
                                 self.parameters[2].getvalue())

    def lorentz_sq(self, x, A, x0, dT):
        # as it was in cpp
        # const double cts=height*(pow(FWHM,2.0)/4.0) * ( 1.0/ ( pow((en-Epeak),2.0) + pow((FWHM/2.0),2.0) ) );
        # return A*(dT**2/4)*(1/((x-x0)**2+(0.5*dT)**2))
        return A * (1 / ((x - x0) ** 2 + (0.5 * dT) ** 2))**2
        # return A*np.pi**-1*0.5*dT/((x-x0)**2+(0.5*dT)**2)

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
