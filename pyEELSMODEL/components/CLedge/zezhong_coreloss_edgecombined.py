from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge

from pyEELSMODEL.components.CLedge.zezhong_coreloss_edge import \
    ZezhongCoreLossEdge
from pyEELSMODEL.database.Zhang.download import download_file
import numpy as np
import logging
import h5py
import os
from pyEELSMODEL import __file__

logger = logging.getLogger(__name__)


# todo implement method

class ZezhongCoreLossEdgeCombined(CoreLossEdge):
    """
    Calculates the coreloss edges for a group of edges using the GOS from
    Zezhong Zang. For instance, the L edge calculates the L3, L2 and L1 edges
    and puts them together with the appropriate prefactors.

    """

    def __init__(
            self,
            specshape,
            A,
            E0,
            alpha,
            beta,
            element,
            edge,
            eshift=0,
            q_steps=100,
            dir_path=None,
    ):
        """
        Coreloss edges which are calculated by Zezhong Zhang.
        https://zenodo.org/records/11199911


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
              The shift of the onset energy with respect to the literature
              value. (default: 0)

        q_steps: int
             The number of q points taken into account for the integration over
             the momentum space. The larger the number of q_steps the more
             accurate the calculation. (default: 100)

        Returns
        -------

        """

        if dir_path is None:
            self.dir_path = os.path.dirname(
                os.path.dirname(__file__) + "/../pyEELSMODEL/database/Zhang/"
            )
        else:
            self.dir_path = dir_path
        self.file = os.path.join(self.dir_path, "Dirac_GOS.gosh")
        if not os.path.exists(self.file):
            download_file(filename=self.file)

        self.xsectionlist = []
        max_edge = self.check_maximum_edge(specshape, element, edge)

        if edge == "K":
            xsectionK = ZezhongCoreLossEdge(
                specshape,
                A,
                E0,
                alpha,
                beta,
                element,
                "K1",
                eshift=eshift,
                q_steps=q_steps,
                dir_path=self.dir_path,
            )
            self.xsectionlist.append(xsectionK)
            super().__init__(specshape, A, E0, alpha, beta, element, "K1",
                             q_steps=q_steps)
            name = element + " K edge: " + str(self.onset_energy) + " eV"
            self.setdisplayname(name)

        elif (edge == "L") | (edge == "M") | (edge == "N") | (edge == "O"):
            start_edge = edge + str(max_edge)
            xsection_op = ZezhongCoreLossEdge(
                specshape,
                A,
                E0,
                alpha,
                beta,
                element,
                start_edge,
                eshift=eshift,
                q_steps=q_steps,
                dir_path=self.dir_path,
            )
            self.xsectionlist.append(xsection_op)
            for i in range(max_edge - 1):
                try:
                    next_edge = edge + str(max_edge - i - 1)
                    xsection = ZezhongCoreLossEdge(
                        specshape,
                        A,
                        E0,
                        alpha,
                        beta,
                        element,
                        next_edge,
                        eshift=eshift,
                        q_steps=q_steps,
                        dir_path=self.dir_path,
                    )
                    xsection.parameters[0].couple(xsection_op.parameters[0])
                    xsection.parameters[1].couple(xsection_op.parameters[1])
                    xsection.parameters[2].couple(xsection_op.parameters[2])
                    xsection.parameters[3].couple(xsection_op.parameters[3])
                    self.xsectionlist.append(xsection)
                    # print(next_edge + " is used")

                except Exception:
                    print(next_edge + " is NOT implemented")

            super().__init__(
                specshape,
                A,
                E0,
                alpha,
                beta,
                element,
                start_edge,
                eshift=eshift,
                q_steps=100,
            )
            name = element + edge + " edge: " + str(self.onset_energy) + " eV"
            self.setdisplayname(name)

        self.manageparameters()
        self.calculate()

    def check_maximum_edge(self, specshape, element, edge):
        """
        Check which is the lowest energy loss edge (the highest number)
        available. Some elements only have a L1 but do not have a L3 edge.
        This function can tell which number is the highest for the given
        element and edge.

        Parameters
        ----------
        element : string
            The element from which the information should be retrieved.
        edge : string
            The edge from which the information should be retrieved.

        Returns
        -------
        max_value: uint > 0
            The integer from the edge having the lowest energy onset.

        """
        with h5py.File(self.file, "r") as f:
            # print('file is opened')
            edges = list(f[element].keys())
            onset_energy_list = list(
                f[element][i]["metadata"].attrs["onset_energy_guess"]
                for i in edges
            )
        sub_edges = []
        sub_onsets = []
        for char in edges:
            if edge in char:
                sub_edges.append(int(char[-1]))
                sub_onsets.append(onset_energy_list[edges.index(char)])
        sub_onsets = np.array(sub_onsets)
        sub_edges = np.array(sub_edges)

        begin = specshape.offset
        end = specshape.offset + specshape.size * specshape.dispersion
        mask = np.logical_and((sub_onsets > begin), (sub_onsets < end))
        # mask = (sub_onsets>begin) & (sub_onsets<end)
        mask = np.ravel(mask)
        # sub_edges = np.array(sub_edges)[mask]

        av_e = np.array(sub_edges)[mask]
        co = np.argsort(av_e)[::-1]

        # stores all the edges used in the calculation of the components
        # self.available_edges = av_e[co]
        # self.available_onsets = np.array(sub_onsets)[mask].flatten()[co]

        return max(av_e[co])

    def manageparameters(self):
        self.parameters = (
            []
        )  # erase existing parameter list and replace with longer list
        self.gradient = []
        # first 4 paramters however remain same meaning
        # but they are coupled with the higher ones
        for xsection in self.xsectionlist:
            for par in xsection.parameters:
                self._addparameter(par)

    def set_onset_energy(self):
        """
        Sets the onset energy of the grouped edge.
        """
        self.onset_energy = self.xsectionlist[
            0
        ].onset_energy  # first part of cross section determines onset

    def set_edge(self, edge):
        self.edge = self.xsectionlist[0].edge

    def calculate_cross_section(self):
        """
        Calculates the cross section of the combined edge
        """
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
