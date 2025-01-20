"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import numpy as np
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
from pyEELSMODEL.core.spectrum import Spectrum
import logging

logger = logging.getLogger(__name__)


class FixedPattern(Component):
    """
    Fixed pattern as component. This component can be used when an experimental
    edge can get extracted from experimental data.

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model

    spectrum: Spectrum
        The spectrum from which to extract the fixed pattern.

    A : float
        The amplitude of fixed pattern compared to the data of
        the spectrum. (default: 1)

    scale: float (> 0)
        A scale value for changing the energy axis. This can be used
        when another dispersion is used between the fixed pattern and
        the experimental data.
        (default: 1)

    shift: float [eV]
        Shifting the data from the spectrum with the given value.
        (default: 0)

    Returns
    -------

    """

    def __init__(self, specshape, spectrum, A=1, scale=1, shift=0, name=' '):
        super().__init__(specshape)
        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(-np.inf, np.inf)
        p1.sethasgradient(True)
        self._addparameter(p1)

        p2 = Parameter('scale', scale, changeallowed=False)
        self._addparameter(p2)

        p3 = Parameter('shift', shift, changeallowed=False)
        self._addparameter(p3)

        self.set_fixeddata(spectrum)

        self._spectrum = Spectrum(self.get_spectrumshape(),
                                  data=self.fixeddata)

        self.setdisplayname(name)
        self.setdescription("Fixed pattern of " + name)
        self._setname(name)

    def set_fixeddata(self, spectrum):
        """
        It can happen that the fixed pattern does not have the same energy axis
        as the spectrum to which we want to fit. Hence a interpolation is
        performed to match the fixedpattern data with the energy axis of the
        model.

        Parameters
        ----------
        spectrum: Spectrum
            The spectrum from where the data is interpolated on the
            other energy axis.

        Returns
        -------
        int_spec.data: numpy 1d array
            The raw data of the interpolated spectrum on the energy axis
            given by the spectrum shape of the fixed pattern.

        """
        int_spec = spectrum.interp_to_other_energy_axis(self)
        self.fixeddata = int_spec.data[:]
        # return int_spec.data[:]

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p3 = self.parameters[2]

        if p1.ischanged() or p2.ischanged() or p3.ischanged():
            A = p1.getvalue()
            scale = p2.getvalue()
            shift = p3.getvalue()
            self.data = self.modified_fixedpattern(self.fixeddata, A, scale,
                                                   shift)
        self.setunchanged()  # put parameters to unchanged

    def modified_fixedpattern(self, fixedpattern, A, scale, shift):
        """
        Recalculates the fixed pattern. The amplitude,

        Parameters
        ----------
        fixedpattern: numpy array (1D)
            The unmodified fixed pattern data.
        A: float
            The amplitude of the fixed pattern.
        scale: float
            The scale to change the energy axis
        shift: float
            The shift of the energy axis

        Returns
        -------
        res: numpy array (1D)
            The recalculated data from the fixed pattern.


        """
        if (scale == 1) and (shift == 0):
            res = A * fixedpattern
        else:
            res = A * self._spectrum.rescale_spectrum(scale, shift).data
        return res

    def getgradient(self, parameter):
        pA = self.parameters[0]
        if parameter == pA:
            self.gradient[0] = self.fixeddata
            return self.gradient[0]
        else:
            return None
