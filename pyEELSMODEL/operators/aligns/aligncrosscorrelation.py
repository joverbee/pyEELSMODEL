# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
from scipy import interpolate
import logging
from pyEELSMODEL.operators.aligns.align import Align
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AlignCrossCorrelation(Align):
    """
    Aligns the dataset using the maximum from the cross correlation of the
    given spectra with the reference. At this point the reference is taken as
    the spectrum which is set in the multispectrum. This alignment method will
    only work if an edge or zero loss is the same over the
    entire scan.


    """
    def __init__(self, multispectrum, other_spectra=[], signal_range=None,
                 cropping=False, interp=1, subpixel=False, is_zlp=False):
        """
        Parameters
        ----------
        multispectrum: Multispectrum
            The spectrum from which the background should be removed
        other_spectra: List of Multispectra
            The other multispectra which can be aligned with the same
            parameters
        signal_range: tuple
            Indicates the region on which the cross correlation should be
            performed.
            (default: None)
        cropping: bool
            Indicates if the edges of the signal are cropped out when aligning
             the zero loss peak.
            (default: False)
        interp: int >1
            The interpolation used for subpixel accuracy (default: 1)
        is_zlp: boolean
            Inidicates if the aligned spectrum is a zero loss peak. If so,
            the offset is modified at the end to put maximum value at zero
            energy
        """

        super().__init__(multispectrum, other_spectra, cropping,
                         signal_range=signal_range, zero_index=None)
        self._reference = np.copy(self.multispectrum.data)
        # normalization of reference
        self._reference = self._reference - self._reference.mean()
        self.interp = interp
        # also other interpolation kinds can be used.
        self.interpolation_kind = 'linear'
        self.subpixel = subpixel
        self.method = 'Cross correlation'
        self.is_zlp = is_zlp

    @property
    def reference(self):
        """
        Reference spectrum used in the cross correlation
        """
        return self._reference

    @reference.setter
    def reference(self, ref):
        """
        Sets the reference data which is also normalized to use in the cross
        correlation
        """
        ind0 = self.multispectrum.get_energy_index(self.signal_range[0])
        ind1 = self.multispectrum.get_energy_index(self.signal_range[1])
        if self.subpixel:
            ref_ = ref[ind0:ind1] - ref[ind0:ind1].mean()
            self._reference = self.interpolate_int(ref_, self.interp)
        else:
            self._reference = ref[ind0:ind1] - ref[ind0:ind1].mean()

    @property
    def correlationmap(self):
        return self._correlationmap

    @correlationmap.setter
    def correlationmap(self, correlationmap):
        self._correlationmap = correlationmap

    @property
    def interp(self):
        return self._interp

    @interp.setter
    def interp(self, index):
        if index < 1 or not isinstance(index, int):
            raise ValueError(r'The given interpolation step should be an '
                             r'integer larger than 0')
        self._interp = index

    @property
    def subpixel(self):
        return self._subpixel

    @subpixel.setter
    def subpixel(self, b):
        self._subpixel = b
        ref = np.copy(self._reference)
        self.reference = ref

    def interpolate_int(self, data, index):
        """
        Function which interpolates the data with the number of index points.
        So if index is two, the interpolated data will contain twice as many
        points.

        Parameters
        -------
        data:

        index:

        Returns
        -------
        None.
        """
        old_axes = np.linspace(0, 1, data.size)
        new_axes = np.linspace(0, 1, data.size * index + 1 - index)
        f = interpolate.interp1d(old_axes, data, kind=self.interpolation_kind)
        return f(new_axes)

    def determine_shift(self):
        """
        Sets the index_shift and shift attribute by finding the maximum value
        of the cross correlation between each spectrum and the reference
        spectrum. Reference spectrum is fixed and should be chosen beforehand.

        Returns
        -------
        None.

        """
        ind0 = self.multispectrum.get_energy_index(self.signal_range[0])
        ind1 = self.multispectrum.get_energy_index(self.signal_range[1])
        shape = (self.multispectrum.xsize, self.multispectrum.ysize)
        shift = np.zeros(shape)
        correlation_map = np.zeros(shape)

        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            avg = self.multispectrum.multidata[islice][ind0:ind1].mean()
            signal = self.multispectrum.multidata[islice][ind0:ind1] - avg
            if self.subpixel:
                signal = self.interpolate_int(signal, self.interp)
            # croscor = np.correlate(self.reference[ind0:ind1], signal, 'full')

            croscor = np.correlate(self.reference, signal, 'full')

            shift[islice] = np.argmax(croscor) - self.reference.size + 1
            correlation_map[islice] = np.max(croscor)

        self.index_shift = shift.astype('int')
        if self.subpixel:
            self.shift = -1 * self.index_shift *\
                         self.multispectrum.dispersion/self.interp
        else:
            self.shift = -1*self.index_shift*self.multispectrum.dispersion
        self.correlationmap = correlation_map

    def perform_alignment(self):
        """
        Performs the alignment procedure by first calculating the shift and
        then applying it to the spectra. The roll method is chosen to perform
        the alignment.

        Returns
        -------
        None.

        """
        self.determine_shift()
        self.align()

        if self.is_zlp:
            zlp_pos = self.aligned.energy_axis[
                np.argmax(self.aligned.mean().data)]

            self.aligned.offset -= zlp_pos

            for spec in self.aligned_others:
                spec.offset -= zlp_pos

        # if self.subpixel:
        #     self.align()
        # else:
        #     self.fast_align()
