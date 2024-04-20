from pyEELSMODEL.misc import hs_gdos as hsdos
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
import os
import h5py
from pyEELSMODEL import __file__


class ZezhongCoreLossEdge(CoreLossEdge):
    """
    Coreloss edges which are calculated by Zezhong Zhang.
    https://zenodo.org/records/7729585

    """

    def __init__(self, specshape, A, E0, alpha, beta, element, edge, eshift=0,
                 q_steps=100, dir_path=None):
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
              The shift of the onset energy with respect to the literature
              value.
            (default: 0)

        q_steps: int
             The number of q points taken into account for the integration
             over the momentum space. The larger the number of q_steps the more
             accurate the calculation. (default: 100)

        Returns
        -------
        """
        if dir_path is None:
            self.dir_path = os.path.dirname(
                os.path.dirname(__file__) + "/../pyEELSMODEL/database/Zhang/"
            )

        else:
            self.set_dir_path(dir_path
                              )

        if edge == 'K':
            edge = 'K1'

        super().__init__(specshape, A, E0, alpha, beta, element, edge,
                         eshift=eshift, q_steps=q_steps)
        self.set_gos_energy_q()
        self.swap_axes = False
        # an approximation for the GOS which does not do the full integration
        self.use_approx = False

    def set_gos_energy_q(self):
        with h5py.File(self.dir_path_element, 'r') as f:
            self.gos = f[self.element][self.edge]['data'][:]
            self.free_energies = f[self.element][self.edge]['free_energy'][:][
                                 :]  # two dimensional array
            self.q_axis = f[self.element][self.edge]['q'][
                          :] / 1e-10  # in  [1/m]

    def set_element(self, element):
        file = os.path.join(self.dir_path, element + '.hdf5')
        isExist = os.path.exists(file)
        if not isExist:
            raise ValueError('Element you selected is not valid')
        self.element = element
        self.dir_path_element = file  # the filename to have easy access

    def set_dir_path(self, path):
        self.dir_path = path

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
            raise TypeError(
                'Edge should be a string: K1, L1, L2, L3, M2, M3, M4, M5, N4, '
                'N5,N6, N7')
        if edge in edge_list:
            self.edge = edge
        else:
            raise ValueError(
                'Edge should be: K1, L1, L2, L3, M2, M3, M4, M5, N4, N5',
                'N6, N7')

    def calculate_cross_section(self):
        ek = self.onset_energy
        print('hll')
        E0 = self.parameters[1].getvalue()
        alpha = self.parameters[3].getvalue()
        beta = self.parameters[2].getvalue()
        # e_axis = self.free_energies[:,0]
        e_axis = self.free_energies

        q_axis = self.q_axis
        gos = self.gos

        pref = 1e28 * self.prefactor

        if self.use_approx:
            cross_section = pref * hsdos.dsigma_dE_from_GOSarray_approx(
                self.energy_axis, e_axis + ek, E0, beta, gos)
        else:
            # cs = hsdos.dsigma_dE_from_GOSarray(self.energy_axis, e_axis + ek,
            #                                    ek+e_axis[0], E0, beta, alpha, q_axis,
            #                                    gos, q_steps=self.q_steps,
            #                                    swap_axes=self.swap_axes)

            cs = hsdos.dsigma_dE_from_GOSarray_bound(self.energy_axis, e_axis, ek,
                                                  E0, beta, alpha, q_axis, gos,
                                                  q_steps=100)

            cross_section = pref * cs

        return cross_section
