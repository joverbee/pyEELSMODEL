"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np


class Offset(Component):
    """
    An offset component
    """

    def __init__(self, specshape, A):
        """
        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        A : float
            The value of the offset.

        Returns
        -------
        """
        super().__init__(specshape)

        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(-np.inf, np.inf)
        p1.sethasgradient(True)
        self._addparameter(p1)
        self.name = 'Offset'

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]
        if p1.ischanged():
            A = p1.getvalue()
            self.data = A * np.ones(self.size)
        self.setunchanged()  # put parameters to unchanged

    def getgradient(self, parameter):
        p1 = self.parameters[0]
        if parameter == p1:
            self.gradient[0] = np.ones(self.energy_axis)
            return self.gradient[0]
        return None
