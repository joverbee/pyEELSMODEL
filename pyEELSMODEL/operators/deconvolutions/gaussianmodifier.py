# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.operators.deconvolution import Deconvolution
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from tqdm import tqdm

logger = logging.getLogger(__name__)

def fft_phys(data):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data)))

def ifft_phys(data):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data)))

class GaussianModifier(Deconvolution):

    def __init__(self, spectrum, llspectrum, factor=4):

        super().__init__(spectrum, llspectrum)
        self.restored = None
        self.factor = factor
        self.plotting=False #debugging feature and to optimize gaussian filter


    def gaussianmodifier(self, signal, psf, centre, fwhm, plotting=False):
        g = Gaussian(self.spectrum.get_spectrumshape(), A=1, centre=centre, fwhm=fwhm)
        g.calculate()

        gdata = g.data/g.data.sum()

        co = np.argmax(psf)
        rol_co = int(psf.size/2) - co


        if plotting:
            plt.figure()
            plt.plot(np.abs(np.fft.rfft(gdata)),label='FFT Gaussian')
            plt.plot(np.abs(np.fft.rfft(psf)), label='FFT Low Loss')
            plt.legend()

        O_f = np.fft.rfft(gdata)*np.fft.rfft(signal)/np.fft.rfft(psf)
        # O_f = np.fft.fft(gdata)*np.fft.fft(signal)/np.fft.fft(psf)

        O_E = np.roll((np.fft.irfft(O_f)), -rol_co)
        # O_E = np.real(np.fft.ifft(O_f))


        # plt.figure()
        # plt.plot(np.angle(O_f))

        return O_E

    def multi_deconvolve(self):
        shape = (self.spectrum.xsize, self.spectrum.ysize)
        ms = self.spectrum.copy()
        centre = self.spectrum.energy_axis[int(self.spectrum.size/2)]
        sigma = self.spectrum.dispersion*self.factor


        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            self.spectrum.setcurrentspectrum(index)
            self.llspectrum.setcurrentspectrum(index)
            psf = self.llspectrum.data / self.llspectrum.data.sum()

            restore = self.gaussianmodifier(self.spectrum.data, psf, centre,sigma,plotting=self.plotting)
            ms.multidata[islice] = restore

        return ms

    def deconvolve(self):
        centre = self.spectrum.energy_axis[int(self.spectrum.size/2)]
        sigma = self.spectrum.dispersion*self.factor
        psf = self.llspectrum.data/self.llspectrum.data.sum()


        # if not isinstance(self.spectrum, MultiSpectrum):
        #     s = Spectrum(self.spectrum.get_spectrumshape(), data=restore)
        #     self.restored = s
        #     return s
        #
        # else:
        #     return restore

        if type(self.spectrum) is MultiSpectrum:
            s = self.multi_deconvolve()
            self.restored = s
        else:
            restore = self.gaussianmodifier(self.spectrum.data, psf, centre, sigma,
                                            plotting=self.plotting)
            s = Spectrum(self.spectrum.get_spectrumshape(), data=restore)
            self.restored = s
        return s

    def apply_nothing(self):
        print('how the hell does this work')
































