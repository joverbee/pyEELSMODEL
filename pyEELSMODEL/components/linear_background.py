"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np


class LinearBG(Component):
    """
    Fast background model, where the number of components can be varied.

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model
    rlist : list
        List of the r values used in the linear background model.
        (default: [1,2,3,4,5])

    Returns
    -------
    """

    def __init__(self, specshape, rlist=[1, 2, 3, 4, 5]):
        super().__init__(specshape)

        n = len(rlist)
        for i in range(n):
            pname = 'a' + str(i)
            p = Parameter(pname, 1.0, True)
            p.setboundaries(-np.Inf, np.Inf)
            # is this true as we will multiply this with another cross section
            p.setlinear(True)
            self._addparameter(p)

            qname = 'r' + str(i)
            q = Parameter(qname, rlist[i], changeallowed=False)
            q.sethasgradient(False)
            self._addparameter(q)

        self.n = n  # number of terms in the sum

        # don't convolute the background it only gives problems and
        # adds no extra physics
        self._setcanconvolute(False)
        self._setshifter(False)  # it is not a shifter type of component

    def calculate(self):
        changes = False
        for param in self.parameters:
            changes = changes or param.ischanged()
        if changes:
            Alist = []
            rlist = []
            for i in range(self.n):
                p = self.parameters[2 * i]
                Alist.append(p.getvalue())
                q = self.parameters[2 * i + 1]
                rlist.append(q.getvalue())

            self.data = self.linear_background(Alist, rlist)
        self.setunchanged()  # put parameters to unchanged

    def linear_background(self, Alist, rlist):
        E = self.energy_axis
        Estart = E[0]
        if Estart < 1:
            Estart = 1

        mask = (E > 0)  # only meaningfull when E>0 and r>0
        signal = np.zeros(E.size)
        for i in range(len(Alist)):
            signal += mask * (Alist[i] * (E / Estart) ** (-rlist[i]))
        return signal

    def getgradient(self, parameter):
        """calculate the analytical partial derivative wrt parameter j
        returns true if succesful, gradient is stored in component.gradient
        """
        # todo implement the analytical gradient
        for ii, param in enumerate(self.parameters):
            if param is parameter:
                index = ii

        if parameter in self.parameters[::2]:
            r = self.parameters[index + 1].getvalue()
            partial = self.linear_background([1], [r])
            return partial

        else:
            return None

    def get_rlist(self):
        rlist = []
        for param in self.parameters:
            if param.linear:
                pass
            else:
                rlist.append(param.getvalue())
        return rlist
