"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
from scipy.special import eval_legendre

class LegendrePolynomial(Component):
    """
    Fast background model, where the number of components can be varied
    """

    def __init__(self, specshape, n=5):
        super().__init__(specshape)


        for i in range(n+1):
            pname = 'a'+str(i)
            p=Parameter(pname,1.0,True)
            p.setboundaries(-np.Inf, np.Inf)
            p.setlinear(True) #is this true as we will multiply this with another cross section
            self._addparameter(p)


        self.n = n #number of terms in the sum
        self._setcanconvolute(False)  # don't convolute the background it only gives problems and adds no extra physics
        self._setshifter(False)  # it is not a shifter type of component
        self.setdisplayname('Legendre')


    def calculate(self):
        changes = False
        for param in self.parameters:
            changes=changes or param.ischanged()
        if changes:
            Alist = []
            for i in range(self.n+1):
                p = self.parameters[i]
                Alist.append(p.getvalue())

            self.data = self.legendre_polynomial(Alist)
        self.setunchanged()  # put parameters to unchanged

    def legendre_polynomial(self, Alist):
        x = np.linspace(-1,1, self.size)
        signal = np.zeros(self.size)
        for i in range(len(Alist)):
            signal += Alist[i]*eval_legendre(i, x)

        return signal

    def getgradient(self, parameter):
        """calculate the analytical partial derivative wrt parameter j
        returns true if succesful, gradient is stored in component.gradient
        """
        #todo implement the analytical gradient
        return None

# TODO add functions for auto-setting of the parameters and power law fit
# see powerlaw.cpp
