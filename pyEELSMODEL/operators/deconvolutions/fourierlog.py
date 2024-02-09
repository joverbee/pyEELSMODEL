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

class FourierLog(Deconvolution):

    def __init__(self, spectrum, llspectrum, factor=4):

        super().__init__(spectrum, llspectrum)
        self.restored = None
        self.plotting=False #debugging feature and to optimize gaussian filter


    def fourierlog(self, signal, psf):
        z = np.fft.rfft(psf)
        j = np.fft.rfft(signal)
        result = np.fft.irfft(z*np.log(j/z))


        return result



    def deconvolve(self):
        psf = self.llspectrum.data/self.llspectrum.data.sum()
        signal = self.spectrum.data/self.spectrum.data.sum()
        restore = self.fourierlog(self.spectrum.data, self.llspectrum.data)

        if not isinstance(self.spectrum, MultiSpectrum):
            s = Spectrum(self.spectrum.get_spectrumshape(), data=restore)
            self.restored = s
            return s

        else:
            return restore



    def apply_nothing(self):
        print('how the hell does this work')
































