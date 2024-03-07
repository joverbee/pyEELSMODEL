# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from pyEELSMODEL.operators.deconvolutions.deconvolution import Deconvolution
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.core.spectrum import Spectrum
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fft_phys(data):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data)))


def ifft_phys(data):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data)))


class GaussianModifier(Deconvolution):
    """
    The gaussian modifier deconvolution as described in
    10.1016/j.ultramic.2009.06.010.
    """
    def __init__(self, spectrum, llspectrum, factor=None):
        """
        Parameters
        ----------
        spectrum: Spectrum or MultiSpectrum
            The spectrum which needs to be deconvolved. Best practice is to
            remove the background before deconvolving.
        llspectrum: Spectrum or MultiSpectrum
            The low loss which is used for the deconvolution
        factor: uint
            This determines the width of the gaussian used. The fhwm
            of the gaussian is factor * dispersion. If factor is None, the
            2*fwhm of the zero-loss peak is used.

        """
        super().__init__(spectrum, llspectrum)
        self.restored = None
        self.restored = None

        if factor is None:
            self.factor = 2 * llspectrum.get_numerical_fwhm() \
                          / llspectrum.dispersion
        else:
            self.factor = factor
        # debugging feature and to optimize gaussian filter
        self.plotting = False

    def gaussianmodifier(self, signal, psf, centre, fwhm, plotting=False):
        """
        The gaussian modfier algorithm.
        """
        g = Gaussian(self.spectrum.get_spectrumshape(), A=1, centre=centre,
                     fwhm=fwhm)
        g.calculate()

        gdata = g.data / g.data.sum()

        co = np.argmax(psf)
        rol_co = int(psf.size / 2) - co

        if plotting:
            plt.figure()
            plt.plot(np.abs(np.fft.rfft(gdata)), label='FFT Gaussian')
            plt.plot(np.abs(np.fft.rfft(psf)), label='FFT Low Loss')
            plt.legend()

        O_f = np.fft.rfft(gdata) * np.fft.rfft(signal) / np.fft.rfft(psf)

        O_E = np.roll((np.fft.irfft(O_f)), -rol_co)

        return O_E

    def multi_deconvolve(self):
        """
        Performs the deconvolution on the  multispectrum
        """

        shape = (self.spectrum.xsize, self.spectrum.ysize)
        ms = self.spectrum.copy()
        centre = self.spectrum.energy_axis[int(self.spectrum.size/2)]
        sigma = self.spectrum.dispersion * self.factor

        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            self.spectrum.setcurrentspectrum(index)
            self.llspectrum.setcurrentspectrum(index)
            psf = self.llspectrum.data / self.llspectrum.data.sum()

            restore = self.gaussianmodifier(self.spectrum.data, psf, centre,
                                            sigma, plotting=self.plotting)
            ms.multidata[islice] = restore

        return ms

    def deconvolve(self):
        centre = self.spectrum.energy_axis[int(self.spectrum.size/2)]
        fwhm = self.spectrum.dispersion*self.factor
        psf = self.llspectrum.data/self.llspectrum.data.sum()

        if type(self.spectrum) is MultiSpectrum:
            s = self.multi_deconvolve()
            self.restored = s
        else:
            restore = self.gaussianmodifier(self.spectrum.data, psf, centre,
                                            fwhm, plotting=self.plotting)
            s = Spectrum(self.spectrum.get_spectrumshape(), data=restore)
            self.restored = s
        return s
