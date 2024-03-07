import numpy as np
import logging
from pyEELSMODEL.operators.aligns.align import Align

logger = logging.getLogger(__name__)


class FastAlignZeroLoss(Align):
    """
    FastAlignZeroLoss is a class which aligns the zero loss peak and other
    spectra can be added which also need to be aligned. It calculates the
    maximum value of the spectrum.
    """
    def __init__(self, multispectrum, other_spectra=None, signal_range=None,
                 cropping=True):
        """
        Parameters
        ----------
        multispectrum: Multispectrum
            The spectrum from which the background should be removed
        other_spectra: List of Multispectra
            The other multispectra which can be aligned with the same
            parameters
        signal_range: tuple
            Indicates the region on which the zero loss fitting should be
            performed.
        cropping: bool
            Indicates if the edges of the signal are cropped out when aligning
            the zero loss peak.
        """

        super().__init__(multispectrum, other_spectra, cropping, signal_range,
                         zero_index=0)
        self.method = 'ZLP Index Maximum'

    def determine_fast_shift(self):
        """
        Sets the index_shift and shift attribute by finding the maximum value
        in each spectrum.

        Returns
        -------
        None.

        """
        ind0 = self.multispectrum.get_energy_index(self.signal_range[0])
        ind1 = self.multispectrum.get_energy_index(self.signal_range[1])
        co = np.argmax(self.multispectrum.multidata[:, :, ind0:ind1], axis=2)
        shift = -1 * (co + ind0)
        self.shift = self.multispectrum.energy_axis[-1 * shift]
        self.index_shift = shift

    def perform_alignment(self):
        """
        Performs the alignment procedure by first calculating the shift and
        then applying it to the spectra. The roll method is chosen to perform
        the alignment.

        Returns
        -------
        None.

        """
        self.determine_fast_shift()
        self.fast_align()
        # self.align()
