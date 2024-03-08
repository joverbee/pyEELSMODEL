from pyEELSMODEL.misc import hydrogen_gdos as hdos
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge


class HydrogenicCoreLossEdge(CoreLossEdge):
    """
      Calculates the core loss cross-section using the hydrogenic wave
      function. The cross-sections can be calculated for the K and L edges.

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

        q_steps: int
        The number of q points taken into account for the integration over
        the momentum space. The larger the number of q_steps the more
        accurate the calculation. (default: 100)

        Returns
        -------
        """

        super().__init__(specshape, A, E0, alpha, beta, element, edge,
                         eshift=eshift, q_steps=q_steps)

    def set_onset_energy(self):
        if self.edge == 'L':
            self.onset_energy = hdos.get_IE3()[self.Z - 13] + self.eshift

            if self.Z == 6:
                self.onset_energy = 5 + self.eshift

        elif self.edge == 'K':
            self.onset_energy = hdos.get_EK()[self.Z - 1] + self.eshift

    def set_edge(self, edge):
        """
        Checks if it is possible to calculate the given edge. If so, then
        this edge is set as an attribute.

        Parameters
        ----------
        edge : string
              The edge which needs to calculated.

        """
        if not isinstance(edge, str):
            raise TypeError('Edge should be a string: K or L')

        if edge == 'K' or edge == 'L':
            if edge == 'L':
                if (self.Z < 13):
                    # added the oxgyen L edge out of curiosity
                    raise ValueError('No L edge for this element')

                elif self.Z > 36:
                    raise ValueError('Hydrogenic L edge not supported for '
                                     'this element (Z>36)')
                else:
                    self.edge = edge
            else:
                self.edge = edge
        else:
            raise ValueError('Edge should be K or L for hydrogenic method')

    def calculate_cross_section(self):
        """
        Calculates the cross section using the hydrogenic wavefunctions.

        Returns
        -------
        cross_section: 1d numpy array [barn]
            The calculated cross section using the hydrogenic wavefunctions.

        """
        csc = 1e28 * hdos.dsigma_dE_hydrogenic(self.energy_axis, self.Z,
                                               self.onset_energy,
                                               self.parameters[1].getvalue(),
                                               self.parameters[2].getvalue(),
                                               self.parameters[3].getvalue(),
                                               shell=self.edge,
                                               q_steps=self.q_steps)

        return csc
