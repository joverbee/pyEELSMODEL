"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np


class FastBG2(Component):
    """
    Fast background model, fully linear and using two terms to model the
    background
    """

    def __init__(self, specshape, A1=1, A2=1, r1=2.5, r2=3.5):
        super().__init__(specshape)
        p0 = Parameter('A1', A1)
        p0.setlinear(True)
        p0.setboundaries(-np.inf, np.inf)
        p0.sethasgradient(True)
        self._addparameter(p0)

        p1 = Parameter('r1', r1, changeallowed=False)
        p1.sethasgradient(False)
        self._addparameter(p1)

        p2 = Parameter('A2', A2)
        p2.setlinear(True)
        p2.setboundaries(-np.inf, np.inf)
        p2.sethasgradient(True)
        self._addparameter(p2)

        p3 = Parameter('r2', r2, changeallowed=False)
        p3.sethasgradient(False)
        self._addparameter(p3)
        # don't convolute the background it only gives problems and
        # adds no extra physics
        self._setcanconvolute(False)
        self._setshifter(False)  # it is not a shifter type of component

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p0 = self.parameters[0]
        p1 = self.parameters[1]
        p2 = self.parameters[2]
        p3 = self.parameters[3]

        if p0.ischanged() or p1.ischanged() or p2.ischanged() \
                or p3.ischanged():
            A1 = p0.getvalue()
            r1 = p1.getvalue()
            A2 = p2.getvalue()
            r2 = p3.getvalue()
            self.data = self.fast_background(A1, r1, A2, r2)
        self.setunchanged()  # put parameters to unchanged

    def fast_background(self, A1, r1, A2, r2):
        E = self.energy_axis
        Estart = E[0]
        if Estart < 1:
            Estart = 1

        mask = (E > 0)  # only meaningfull when E>0 and r>0
        # return mask*(A1 * (E/Estart) ** (-r1) + A2 * (E/Estart) ** (-r2))
        return mask * (A1 * E ** (-r1) + A2 * E ** (-r2))

    def getgradient(self, parameter):
        """calculate the analytical partial derivative wrt parameter j
        returns true if succesful, gradient is stored in component.gradient
        """
        p0 = self.parameters[0]
        p1 = self.parameters[1]
        p2 = self.parameters[2]
        p3 = self.parameters[3]

        r1 = p1.getvalue()
        r2 = p3.getvalue()

        E = self.energy_axis
        Estart = E[0]
        if Estart < 1:
            Estart = 1
        mask = (E > 0)  # only meaningfull when E>0 and r>0
        if parameter == p0:
            self.gradient[0] = mask * E ** (-r1)
            return self.gradient[0]
        elif parameter == p2:
            self.gradient[2] = mask * E ** (-r2)
            return self.gradient[2]
        else:
            return False


class FastBG3(Component):
    """
    Fast background model, fully linear and using three terms to model the
    background
    """

    def __init__(self, specshape, A1=1, A2=1, A3=1, r1=2, r2=3, r3=4):
        super().__init__(specshape)
        p0 = Parameter('A1', A1)
        p0.setlinear(True)
        p0.setboundaries(-np.inf, np.inf)
        p0.sethasgradient(True)
        self._addparameter(p0)

        p1 = Parameter('r1', r1, changeallowed=False)
        p1.sethasgradient(False)
        self._addparameter(p1)

        p2 = Parameter('A2', A2)
        p2.setlinear(True)
        p2.setboundaries(-np.inf, np.inf)
        p2.sethasgradient(True)
        self._addparameter(p2)

        p3 = Parameter('r2', r2, changeallowed=False)
        p3.sethasgradient(False)
        self._addparameter(p3)

        p4 = Parameter('A3', A3)
        p4.setlinear(True)
        p4.setboundaries(-np.inf, np.inf)
        p4.sethasgradient(True)
        self._addparameter(p4)

        p5 = Parameter('r3', r3, changeallowed=False)
        p5.sethasgradient(False)
        self._addparameter(p5)
        # don't convolute the background it only gives problems
        # and adds no extra physics
        self._setcanconvolute(False)
        self._setshifter(False)  # it is not a shifter type of component

    def calculate(self):
        p0 = self.parameters[0]
        p1 = self.parameters[1]
        p2 = self.parameters[2]
        p3 = self.parameters[3]
        p4 = self.parameters[4]
        p5 = self.parameters[5]

        if p0.ischanged() or p1.ischanged() or p2.ischanged() \
                or p3.ischanged() or p4.ischanged() or p5.ischanged():
            A1 = p0.getvalue()
            r1 = p1.getvalue()
            A2 = p2.getvalue()
            r2 = p3.getvalue()
            A3 = p4.getvalue()
            r3 = p5.getvalue()
            self.data = self.fast_background3(A1, r1, A2, r2, A3, r3)
        self.setunchanged()  # put parameters to unchanged

    def fast_background3(self, A1, r1, A2, r2, A3, r3):
        E = self.energy_axis
        Estart = E[0]
        if Estart < 1:
            Estart = 1

        mask = (E > 0)  # only meaningfull when E>0 and r>0
        return mask * (A1 * (E / Estart) ** (-r1) + A2 * (E / Estart) ** (
            -r2) + A3 * (E / Estart) ** (-r3))

    def getgradient(self, parameter):
        """calculate the analytical partial derivative wrt parameter j
        returns true if succesful, gradient is stored in component.gradient
        """
        p0 = self.parameters[0]
        p1 = self.parameters[1]
        p2 = self.parameters[2]
        p3 = self.parameters[3]
        p4 = self.parameters[4]
        p5 = self.parameters[5]

        r1 = p1.getvalue()
        r2 = p3.getvalue()
        r3 = p5.getvalue()

        E = self.energy_axis
        Estart = E[0]
        if Estart < 1:
            Estart = 1
        mask = (E > 0)  # only meaningfull when E>0 and r>0
        if parameter == p0:
            self.gradient[0] = mask * (E) ** (-r1)
            return self.gradient[0]
        elif parameter == p2:
            self.gradient[2] = mask * (E) ** (-r2)
            return self.gradient[2]
        elif parameter == p4:
            self.gradient[2] = mask * (E) ** (-r3)
            return self.gradient[4]
        else:
            # throw Componenterr::bad_index()
            return False
