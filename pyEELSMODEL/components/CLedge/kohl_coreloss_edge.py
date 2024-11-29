from pyEELSMODEL.misc import hs_gdos as hsdos
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.database.Segger_Guzzinati_Kohl.download import download_file
import os
import h5py
from pyEELSMODEL import __file__


class KohlLossEdge(CoreLossEdge):
    """
    Coreloss edges which are calculated by Leonhard Segger, Giulio Guzzinati
    and Helmut Kohl https://zenodo.org/record/6599071#.Y3I1cnbMKUk

    """

    def __init__(self, specshape, A, E0, alpha, beta, element, edge, eshift=0,
                 q_steps=100, dir_path=None,fast=False):
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
        The type of edge. (K1, L3, M4, ...)

        eshift: float [eV]
        The shift of the onset energy with respect to the literature value.
        (default: 0)

        q_steps: int
        The number of q points taken into account for the integration over
        the momentum space. The larger the number of q_steps the more
        accurate the calculation. (default: 100)

        dir_path: string
            The filepath indicating where the GOS tables can be found.
            If None, the default path is used.

        fast: bool
        Use vectorized operations for the crossection calculation.

        Returns
        -------
        """
        if dir_path is None:
            dir_ = "/../pyEELSMODEL/database/Segger_Guzzinati_Kohl/"
            dir_path = os.path.dirname(
                os.path.dirname(__file__) + dir_
            )
        else:
            self.set_dir_path(dir_path)
        self.file = os.path.join(dir_path, 'Segger_Guzzinati_Kohl_1.5.0.gosh')
        if not os.path.exists(self.file):
            download_file(filename=self.file)
            

        self.set_dir_path(dir_path)
        super().__init__(specshape, A, E0, alpha, beta, element, edge,
                         eshift=eshift, q_steps=q_steps)

        self.set_gos_energy_q()
        self._fast = fast

    def set_gos_energy_q(self):
        if int(self.edge[-1]) == 1:
            edge = self.edge
        elif (int(self.edge[-1]) % 2 == 0):
            edge = self.edge + str(int(self.edge[-1]) + 1)
        else:
            edge = self.edge[0] + str(int(self.edge[-1]) - 1) + self.edge[-1]

        with h5py.File(self.file, 'r') as f:
            self.gos = f[self.element][edge]['data'][:]
            self.free_energies = f[self.element][edge]['free_energies'][:]
            self.q_axis = f[self.element][edge]['q'][:]

    def set_element(self, element):
        with h5py.File(self.file, 'r+') as f:
            elem_list = list(f.keys())
        if element in elem_list:
            self.element = element
        else:
            raise ValueError('Element you selected is not valid')

    def set_dir_path(self, path):
        self.dir_path = path
        self.file = os.path.join(self.dir_path,
                                 'Segger_Guzzinati_Kohl_1.5.0.gosh')

    # def set_onset_path(self, path):
    #     self.onset_path = path

    # def set_onset_energy(self):
    #     with h5py.File(self.onset_path, 'r+') as f:
    #         data = f[self.element][self.edge][:]
    #         self.onset_energy = data[0] + self.eshift
    #         self.prefactor = data[1]

    def _check_str_in_list(self, list, edge):
        for name in list:
            if name[0] == edge[0]:
                return True

        return False

    def set_edge(self, edge):
        """
        Checks if the given edge is valid and adds the directories of the
        :param edge:
        :return:
        """
        edge_list = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
                     'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
        if not isinstance(edge, str):
            raise TypeError('Edge should be a string: K1, L1, L2, L3, M2, M3,'
                            ' M4, M5, N4, N5', 'N6', 'N7')
        if edge in edge_list:
            self.edge = edge
        else:
            raise ValueError('Edge should be: K1, L1, L2, L3, M2, M3, M4, M5,'
                             ' N4, N5', 'N6', 'N7')

    # def set_edge(self, edge):
    #     """
    #     Checks if the given edge is valid and adds the directories of the
    #     :param edge:
    #     :return:
    #     """
    #     edge_list = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
    #                  'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
    #
    #     if not isinstance(edge, str):
    #         raise TypeError('Edge should be a string: K1, L1, L2, L3, M2,'
    #                         ' M3,'
    #                         ' M4, M5, N4, N5', 'N6', 'N7')
    #     if edge in edge_list:
    #         self.edge = edge
    #     else:
    #         raise ValueError('Edge should be: K1, L1, L2, L3, M2, M3, M4,'
    #                          ' M5,'
    #                          ' N4, N5', 'N6', 'N7')

    def calculate_cross_section(self):
        """
        Calculates the cross section in barns (1e-28 m^2)


        """
        ek = self.onset_energy
        E0 = self.parameters[1].getvalue()
        alpha = self.parameters[3].getvalue()
        beta = self.parameters[2].getvalue()
        e_axis = self.free_energies
        q_axis = self.q_axis
        gos = self.gos

        prf = 1e28 * self.prefactor  # prefactor and convert to barns

        if self._fast:
            css = prf * hsdos.dsigma_dE_from_GOSarray_FastKohl(self.energy_axis,
                                                  e_axis + ek, ek, E0, beta,
                                                  alpha, q_axis, gos,
                                                  q_steps=100)

        else:
            css = prf * hsdos.dsigma_dE_from_GOSarray(self.energy_axis,
                                                      e_axis + ek, ek, E0, beta,
                                                      alpha, q_axis, gos,
                                                      q_steps=100,
                                                      swap_axes=True)

        return css
