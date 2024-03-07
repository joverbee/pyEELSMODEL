from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge


class DummyEdge(CoreLossEdge):
    """
    DummyEdge is a first approximation of the edge. This assumes each edge has
    a different onset energy but the powerlaw value is can be chosen. Note that
    the parameters E0, alpha, beta do not influence the dummy edge model.
    """

    def __init__(self, specshape, A, E0=300e3, alpha=1e-9, beta=10e-3,
                 element=None, edge=None, eshift=0, r=3):
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
            The type of edge. (K1, L1, L2, L3, M1, etc.)

        eshift: float [eV]
            The shift of the onset energy with respect to the literature value.
            (default: 0)
        r: float
            The powerlaw of the dummyedge. (default: 3)

        Returns
        -------
        """

        super().__init__(specshape, A, E0, alpha, beta, element, edge,
                         eshift=eshift)
        self.r = r

    @classmethod
    def edge_by_onset(self, specshape, A, onset):
        """
        Class method to make a dummy edge where the edge onset energy is
        specified. In the list of elements not every edge is included so it
        would be of interest to have some freedom if needed.

        :param specshape:
        :param A:
        :param onset:
        :return:
        """
        d = DummyEdge(specshape, A, 1, 1, 1, 'C', 'K')
        d.onset_energy = onset
        return d

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
        edge_list = ['K1', 'L1', 'L2', 'L3', 'M2', 'M3', 'M4', 'M5', 'N4',
                     'N5']

        elem_list = self.get_elements()

        if not isinstance(edge, str):
            raise TypeError(
                'Edge should be a string: K1, L1, L2, L3, M2, M3, M4, M5, N4,'
                ' N5')
        if edge in edge_list:
            if self._check_str_in_list(elem_list, edge):
                self.edge = edge
            else:
                raise ValueError(r'The element {} does not'
                                 r' have a {} edge'.format(self.element, edge))

        else:
            raise ValueError(
                'Edge should be: K1, L1, L2, L3, M2, M3, M4, M5, N4, N5')

    def calculate_cross_section(self):
        cross_section = self.prefactor * self.energy_axis ** (-self.r)
        scale = 1 / cross_section[self.get_energy_index(self.onset_energy)]
        cross_section = cross_section * scale
        boolean = self.energy_axis < self.onset_energy
        cross_section[boolean] = 0
        return cross_section
