# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import logging
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape


logger = logging.getLogger(__name__)


class Deconvolution(Operator):
    """
    Parten class for the deconvolution methods.
    """
    def __init__(self, spectrum, llspectrum):
        """
        Parameters
        ----------
        spectrum: Spectrum or MultiSpectrum
            The spectrum which needs to be deconvolved. Best practice is to
            remove the background before deconvolving.
        llspectrum: Spectrum or MultiSpectrum
            The low loss which is used for the deconvolution


        """
        self.spectrum = spectrum
        self.llspectrum = llspectrum

        if isinstance(spectrum, MultiSpectrum):
            self.spectrum.multidata = self.spectrum.multidata.astype(float)

        self.spectrum.data = self.spectrum.data.astype(float)
        self.llspectrum.data = self.llspectrum.data.astype(float)

        if self.spectrum.size > self.llspectrum.size:
            print('low loss spectrum has to have same size as model')
            print('the low loss will be zero padded to have the same size')
            llspectrum = self.padding(spectrum.get_spectrumshape(), llspectrum)
            self.llspectrum = llspectrum

    def padding(self, specshape, llspectrum):
        """
        Zero padds the low loss if the size is smaller than the core-loss

        Parameters
        ----------
        specshape: Spectrumshape
            The spectrum shape of the core-loss
        llspectrum: Spectrum or MultiSpectrum
            The low loss which is used for the deconvolution

        """

        size_dif = specshape.size - llspectrum.size
        before = size_dif//2
        after = size_dif - before
        print('padding is done')
        print(llspectrum)
        if type(llspectrum) is Spectrum:
            print('low loss is spectrum')
            pad_data = np.pad(llspectrum.data, pad_width=(before, after))
            noffset = llspectrum.offset - llspectrum.dispersion * before
            sph = Spectrumshape(llspectrum.dispersion, noffset, pad_data.size)
            s = Spectrum(sph, data=pad_data)

        elif type(llspectrum) is MultiSpectrum:
            print('low loss is multispectrum')
            pad_w = ((0, 0), (0, 0), (before, after))
            pad_data = np.pad(llspectrum.multidata, pad_width=pad_w)
            noffset = llspectrum.offset - llspectrum.dispersion * before
            sph = MultiSpectrumshape(llspectrum.dispersion,
                                     noffset,
                                     pad_data.shape[-1],
                                     llspectrum.xsize,
                                     llspectrum.ysize)
            s = MultiSpectrum(sph, data=pad_data)

        return s
