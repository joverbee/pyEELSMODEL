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
        p1.setboundaries(0, np.inf)
        p1.sethasgradient(False)
        self._addparameter(p1)

        p2 = Parameter('centre', centre)
        p2.setboundaries(-np.inf, np.inf)
        p2.sethasgradient(False)
        self._addparameter(p2)

        p3 = Parameter('FWHM', fwhm)
        p3.setboundaries(0, np.inf)
        p3.sethasgradient(False)
        self._addparameter(p3)

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        E = self.energy_axis
        self.data = self.lorentz_sq(E, self.parameters[0].getvalue(),
                                    self.parameters[1].getvalue(),
                                    self.parameters[2].getvalue())

    def lorentz_sq(self, x, A, x0, dT):
        # return A*(dT**2/4)*(1/((x-x0)**2+(0.5*dT)**2))
        return A * (1 / ((x - x0) ** 2 + (0.5 * dT) ** 2))**2

    def getgradient(self, parameter):
        return None
