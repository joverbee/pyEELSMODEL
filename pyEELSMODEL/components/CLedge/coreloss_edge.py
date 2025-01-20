"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
from pyEELSMODEL import __file__
import os
import h5py
import logging
import pyEELSMODEL.misc.hydrogen_gdos as hdos

logger = logging.getLogger(__name__)


class CoreLossEdge(Component):
    """
    CoreLossEdge generic component class to derive core loss edge
    implementations from.


    """

    def __init__(self, specshape, A, E0, alpha, beta, element, edge, eshift=0,
                 q_steps=100):
        """
        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        A : float
            Amplitude of the edge

        E0: float [V]
            The acceleration voltage of the microscope in volts

        alpha: float [rad]
            The convergence angle of the incoming probe in radians,

        beta: float [rad]
            The collection angle in radians.

        element: string
            The name of the element from which to calculate the edge model.

        edge: string
            The type of edge. (K1, L1, L2, L3, M1, etc.)

        eshift: float [eV]
            The shift of the onset energy with respect to the literature value.
            (default: 0)

        q_steps: int
            The number of q points taken into account for the integration over
            the momentum space. The larger the number of q_steps the more
            accurate the calculation. (default: 100)

        """
        super().__init__(specshape)

        self.elements_dir = os.path.dirname(__file__)
        # print(self.elements_dir)
        self.elements_name = 'element_info.hdf5'

        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(-np.inf, np.inf)
        p1.sethasgradient(True)
        self._addparameter(p1)

        p2 = Parameter('E0', E0, changeallowed=False)
        self._addparameter(p2)

        p3 = Parameter('beta', beta, changeallowed=False)
        self._addparameter(p3)

        p4 = Parameter('alpha', alpha, changeallowed=False)
        self.parameters.append(p4)

        self.eshift_ = eshift
        self.set_element(element)
        self.set_Z()
        self.set_edge(edge)
        self.set_onset_energy()
        self.q_steps = q_steps
        name = element + ' ' + edge + ' edge' + str(self.onset_energy) + ' eV'
        self.setdisplayname(name)

    def get_elements_dir(self):
        return os.path.join(self.elements_dir, self.elements_name)

    def get_elements(self):
        """
        Returns a list of allowed element names
        """
        with h5py.File(self.get_elements_dir(), 'r') as f:
            elem_list = list(f.keys())
        return elem_list

    def set_element(self, element):
        """
        Sets given element to the CoreLossEdge.

        Parameters
        ----------
        element: string
            The element to which to set it.

        """
        ky = self.get_elements()
        if element in ky:
            self.element = element
        else:
            raise ValueError('Not a valid element, here are some examples of '
                             'valid elements: {}, {},'
                             ' {}'.format(ky[0], ky[10], ky[30]))

    def set_Z(self):
        """
        Set the atomic number Z of the element from the CoreLossEdge.
        Needed for the hydrogenic calculations
        """
        with h5py.File(self.get_elements_dir(), 'r') as f:
            self.Z = f[self.element].attrs['Z']

    def set_onset_energy(self):
        with h5py.File(self.get_elements_dir(), 'r') as f:
            onset = f[self.element][self.edge].attrs['onset_energy']
            onset += self.eshift_
            prf = f[self.element][self.edge].attrs['occupancy_ratio']

            self.onset_energy = onset
            self.prefactor = prf

    def set_edge(self, edge):
        """
        Checks if the given edge is valid and adds the directories of the
        :param edge:
        :return:
        """
        edge_list = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
                     'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
        if not isinstance(edge, str):
            raise TypeError(
                'Edge should be a string: K1, L1, L2, L3, M2, M3, M4, M5, N4,'
                ' N5', 'N6', 'N7')
        if edge in edge_list:
            self.edge = edge
        else:
            raise ValueError(
                'Edge should be: K1, L1, L2, L3, M2, M3, M4, M5, N4, N5', 'N6',
                'N7')

    def calculate_cross_section(self):
        print('if you see this you have called calculate_cross section '
              'on a coreloss edge that didnt implement '
              'calculate_cross_section')

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        pA = self.parameters[0]
        p2 = self.parameters[1]
        p3 = self.parameters[2]
        p4 = self.parameters[3]

        if p2.ischanged() or p3.ischanged() or p4.ischanged():
            self.cross_section = self.calculate_cross_section()
            self.data = pA.getvalue() * self.cross_section

        if pA.ischanged():
            self.data = pA.getvalue() * self.cross_section
        self.setunchanged()

    def getgradient(self, parameter):
        # gradient with respect to weight is easy, others are too difficult
        pA = self.parameters[0]
        if parameter == pA:
            self.gradient[0] = self.cross_section
            return self.gradient[0]
        else:
            return None

    def get_convergence_correction_factor(self, nsamples=100):
        """
        Calculates the convergence correction for each different theta values.
        This depends on the geometry of the setup.

        :return:
        """
        alpha = self.parameters[3].getvalue()
        beta = self.parameters[2].getvalue()
        theta_array = np.exp(
            np.linspace(np.log(1e-9), np.log(alpha + beta), nsamples))
        F = hdos.convergence_correction_factor(alpha, beta, theta_array)
        return F, theta_array
