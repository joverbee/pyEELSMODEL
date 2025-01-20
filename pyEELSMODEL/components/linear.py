from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np


class Linear(Component):
    """
    A Linear component.
    """

    def __init__(self, specshape, m, q):
        """
        Initialize the Linear object
        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model.

        m : float
            The slope of the linear function.

        q: float
            The constant value of the linear function.

        Returns
        -------
        """
        super().__init__(specshape)

        p1 = Parameter('m', m)
        p1.setlinear(True)
        p1.sethasgradient(True)
        self._addparameter(p1)

        p2 = Parameter('q', q)
        p2.setlinear(True)
        p2.sethasgradient(True)
        self._addparameter(p2)

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        if p1.ischanged() or p2.ischanged():
            m = p1.getvalue()
            q = p2.getvalue()
            self.data = self.linear_function(m, q)
        self.setunchanged()  # put parameters to unchanged

    def linear_function(self, m, q):
        return m * self.energy_axis + q

    def getgradient(self, parameter):
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        if parameter == p1:
            self.gradient[0] = self.linear_function(1, 0)
            return self.gradient[0]
        if parameter == p2:
            self.gradient[1] = np.ones(self.size)
            return self.gradient[1]
        return None
