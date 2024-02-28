"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
from pyEELSMODEL.misc import physical_constants as pc
import math
class Plasmon(Component):
    """
    A physical model of ZL and plasmon peaks This is not very adquate to use
    for model based fitting
    """


# i think the ZL peak is missing?
    def __init__(self, specshape, A, Ep, Ew, n, tlambda, beta, E0):
        super().__init__(specshape)

        p0 = Parameter('A',A)
        p0.setlinear(True)
        p0.setboundaries(0,np.Inf)
        p0.sethasgradient(False)
        self._addparameter(p0)  #height zero loss peak
      
        p1 = Parameter('Ep',Ep, changeallowed=False) #eV
        p1.sethasgradient(False)
        self._addparameter(p1)
        

        p2 = Parameter('Ew',Ew, changeallowed=False) #eV
        p2.sethasgradient(False)
        self._addparameter(p2)  

        p3 = Parameter('Eb',1, changeallowed=False) #Binding energy has not been taken into account
        p3.sethasgradient(False)
        self._addparameter(p3)  #height zero loss peak
       
        p4 = Parameter('n',n, changeallowed=False)
        p4.sethasgradient(False)
        self._addparameter(p4) 

        p5 = Parameter('tlambda',tlambda, changeallowed=False)
        p5.sethasgradient(False)
        self._addparameter(p5) 

        p6 = Parameter('beta',beta, changeallowed=False)
        p6.sethasgradient(False)
        self._addparameter(p6)

        p7 = Parameter('E0',E0, changeallowed=False)
        p7.sethasgradient(False)
        self._addparameter(p7)  #height zero loss peak
        


    def calculate(self):
        self.data = self.plasmon_function()


    def plasmon_function(self):
        for i in range(self.parameters[4].getvalue()):
            fraction = self.poisson_fraction(i+1, self.parameters[5].getvalue())
            plasmon = fraction*self.calculate_plasmon(self.parameters[1].getvalue()*(i+1),
                                                      self.parameters[2].getvalue(), self.parameters[3].getvalue(),
                                                      self.parameters[6].getvalue(), self.parameters[7].getvalue())
            if i == 0:
                tot_plasmon = np.copy(plasmon)
            else:
                tot_plasmon += plasmon


        return  (np.exp(self.parameters[5].getvalue())-1)*self.parameters[0].getvalue()*tot_plasmon/tot_plasmon.sum()


    def calculate_plasmon(self, Ep, Ew, Eb, beta, E0):
        """
        The drude model of the plasmon from which the enerrgy differential cross section
        is calculated. The description of hte formula can be found in the Egerton book (p.139, formula 3.43c)
        :param Ep: Energy of the plasmon
        :param Ew: Width of the plasmon
        :param Eb: Binding energy of the bulk material
        :return: volume plasmon energy loss spectrum
        """
        E = self.energy_axis
        eps = 1 - Ep**2/(E**2-Eb**2+E*Ew*1j)
        inv_eps = np.imag(-1/eps)

        theta_E = E/(pc.joule_to_eV(pc.gamma(E0)*pc.m0()*pc.speed_electron(E0)**2))

        prefactor = (np.pi*(1e9*pc.a0())*pc.joule_to_eV(pc.m0()*pc.speed_electron(E0)**2))**-1 #nm, borh radius in nm
        anglog = np.zeros(theta_E.size)
        anglog[E>0] = np.log(1.+ beta**2/theta_E[E>0]**2)
        volint = prefactor * inv_eps * anglog
        volint[E<=0] = 0 # else it will give back nan, which gives trouble
        return volint

    def poisson_fraction(self, n, tlambda):
        """

        :param n: integer value of the plasmon
        :param tlambda: thickness over mean free path (lambda) ratio
        :return: fraction
        """
        nfac = math.factorial(n)
        fraction = np.exp(-tlambda)*tlambda**n/ nfac
        return fraction

