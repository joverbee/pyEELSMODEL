"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
from pyEELSMODEL.misc import physical_constants as pc
from pyEELSMODEL.misc.se_dos import dsigma_dE_SE
import math


class SE_Excitation(Component):
    """
    A physical model of the single electron excitation from the valence states.
    Definition is calculated from the Krivanek paper (). It seems there are some typos
    in the formulas so do not trust this. 
    """

    # i think the ZL peak is missing?
    def __init__(self, specshape, A, E0, beta, alpha, Eb, Z, q_steps=100):
        super().__init__(specshape)

        p0 = Parameter('A', A)
        p0.setlinear(True)
        p0.setboundaries(0, np.Inf)
        p0.sethasgradient(False)
        self._addparameter(p0)  # height zero loss peak

        p1 = Parameter('E0', E0, changeallowed=False)  # eV
        p1.sethasgradient(False)
        self._addparameter(p1)

        p2 = Parameter('beta', beta, changeallowed=False)  # eV
        p2.sethasgradient(False)
        self._addparameter(p2)

        p3 = Parameter('alpha', alpha, changeallowed=False)  # eV
        p3.sethasgradient(False)
        self._addparameter(p3)

        self.onset_energy = Eb
        self.Z = Z
        self.q_steps=q_steps

        self.cross_section = self.se_function()


    def calculate(self):
        pA=self.parameters[0]
        p2=self.parameters[1]
        p3=self.parameters[2]
        p4=self.parameters[3]

        if p2.ischanged() or p3.ischanged() or p4.ischanged():
            self.cross_section=self.se_function()
            self.data = pA.getvalue()*self.cross_section

        if pA.ischanged():
            self.data = pA.getvalue()*self.cross_section
        self.setunchanged()


    def se_function(self):
        cross_section = 1e28*dsigma_dE_SE(self.energy_axis, self.Z, self.onset_energy,
                                                          self.parameters[1].getvalue(),self.parameters[2].getvalue(),
                                                          self.parameters[3].getvalue(), q_steps=self.q_steps)

        return cross_section
