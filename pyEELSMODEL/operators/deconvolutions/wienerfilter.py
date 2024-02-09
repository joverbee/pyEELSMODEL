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

class WienerFilter(Deconvolution):

    def __init__(self, spectrum, llspectrum, iterations=1):

        super().__init__(spectrum, llspectrum)
        self.restored = None
        self.plotting=False #debugging feature and to optimize gaussian filter
        self.iterations = iterations

    def wienerfilter(self, signal, psf, iterations):
        O_r = signal.copy()
        O_f = fft_phys(O_r)
        I_f = fft_phys(signal)
        N_f = np.ones(O_r.size)*1e-12
        psf_f = fft_phys(psf)

        SNR = np.ones(psf.size)*100

        co = np.argmax(psf)
        rol_co = int(psf.size/2) - co

        for i in range(iterations):
            # Y_f = np.conjugate(psf_f)/(np.abs(psf_f)**2+N_f**2/np.abs(O_f)**2)
            Y_f = np.conjugate(psf_f)/(np.abs(psf_f)**2+1/SNR)

            O_f = Y_f*I_f
            plt.figure()
            plt.plot(N_f**2/np.abs(O_f)**2)
            plt.plot(np.abs(psf_f)**2)

        O_E = np.roll(np.real(ifft_phys(O_f)), -rol_co)

        return O_E



    def deconvolve(self):
        psf = self.llspectrum.data/self.llspectrum.data.sum()
        signal = self.spectrum.data/self.spectrum.data.sum()
        restore = self.wienerfilter(self.spectrum.data, psf, self.iterations)

        if not isinstance(self.spectrum, MultiSpectrum):
            s = Spectrum(self.spectrum.get_spectrumshape(), data=restore)
            self.restored = s
            return s

        else:
            return restore



    def apply_nothing(self):
        print('how the hell does this work')
































