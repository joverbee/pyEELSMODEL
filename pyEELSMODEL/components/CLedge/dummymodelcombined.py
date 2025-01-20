from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.components.CLedge.dummymodel import DummyEdge
import numpy as np
import logging

logger = logging.getLogger(__name__)


# todo not needed for

class DummyEdgeCombined(CoreLossEdge):
    """
    DummyEdge is a first approximation of the edge. This assumes each edge has
    a different onset energy but the powerlaw value is -3.
    Note that the parameters E0, alpha, beta do not influence the
    dummyedge model.

    """

    def __init__(self, specshape, A, E0, alpha, beta, element, edge, eshift=0):
        """
        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        A : float
            Amplitude of the edge

        E0: float [V]
            The acceleration voltage of the microscope

        alpha: float [rad]
            The convergence angle of the incoming probe

        beta: float [rad]
            The collection angle

        element: string
            The name of the element from which to calculate the edge model.

        edge: string
            The type of edge. (K, L or M)

        eshift: float [eV]
            The shift of the onset energy with respect to the literature value.
            (default: 0)


        Returns
        -------
        """

        self.xsectionlist = []
        if edge == 'K':
            xsectionK = DummyEdge(specshape, A, E0, alpha, beta, element, 'K')
            self.xsectionlist.append(xsectionK)
            super().__init__(specshape, A, E0, alpha, beta, element, 'K')
            name = element + ' K edge: ' + str(self.onset_energy) + ' eV'
            self.setdisplayname(name)
        if edge == 'L':

            xsectionL3 = DummyEdge(specshape, A, E0, alpha, beta, element,
                                   'L3')
            # xsectionL3.prefactor = 2/3
            try:
                xsectionL2 = DummyEdge(specshape, A, E0, alpha, beta, element,
                                       'L2')
                # xsectionL2.prefactor = 1 / 3
            except Exception:
                print('No L2 is given which means it is the same as L3')
                xsectionL2 = DummyEdge(specshape, A, E0, alpha, beta, element,
                                       'L3')
                xsectionL2.prefactor = 1 / 2

            xsectionL2.parameters[0].couple(xsectionL3.parameters[0])
            xsectionL2.parameters[1].couple(xsectionL3.parameters[1])
            xsectionL2.parameters[2].couple(xsectionL3.parameters[2])
            xsectionL2.parameters[3].couple(xsectionL3.parameters[3])

            self.xsectionlist.append(xsectionL3)
            self.xsectionlist.append(xsectionL2)
            super().__init__(specshape, A, E0, alpha, beta, element, 'L3',
                             eshift=eshift, q_steps=100)
            name = element + 'L3,2,1 edge: ' + str(self.onset_energy) + ' eV'
            self.setdisplayname(name)

        elif edge == 'M':
            xsectionM5 = DummyEdge(specshape, A, E0, alpha, beta, element,
                                   'M5')
            xsectionM4 = DummyEdge(specshape, A, E0, alpha, beta, element,
                                   'M4')

            xsectionM4.parameters[0].couple(xsectionM5.parameters[0])
            xsectionM4.parameters[1].couple(xsectionM5.parameters[1])
            xsectionM4.parameters[2].couple(xsectionM5.parameters[2])
            xsectionM4.parameters[3].couple(xsectionM5.parameters[3])

            self.xsectionlist.append(xsectionM5)
            self.xsectionlist.append(xsectionM4)

            super().__init__(specshape, A, E0, alpha, beta, element, 'M5',
                             eshift=eshift, q_steps=100)
            name = element + 'M5,4,3,2 edge' + str(self.onset_energy) + ' eV'
            self.setdisplayname(name)

        elif edge == 'N':
            logger.error('N edge not implemented yet')

        self.manageparameters()
        self.calculate()

    def manageparameters(self):
        # erase existing parameter list and replace with longer list
        self.parameters = []
        self.gradient = []
        # first 4 paramters however remain same meaning but they are
        # coupled with the higher ones
        for xsection in self.xsectionlist:
            for par in xsection.parameters:
                self._addparameter(par)

    def set_onset_energy(self):
        self.onset_energy = self.xsectionlist[
            0].onset_energy  # first part of cross section determines onset

    def set_edge(self, edge):
        self.edge = self.xsectionlist[0].edge

    def calculate_cross_section(self):
        cross_section = np.zeros(self.size)
        for xsection in self.xsectionlist:
            if xsection.suppress==False:
                cross_section += xsection.calculate_cross_section()
            else:
                print('subcomponent is suppressed, to unset call setsuppress(False) on the specific subcomponent:',xsection.name)            
        return cross_section

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
