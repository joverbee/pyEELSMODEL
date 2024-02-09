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
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RichardsonLucyDeconvolution(Deconvolution):

    def __init__(self, spectrum, llspectrum, iterations=5):

        super().__init__(spectrum, llspectrum)
        # if isinstance(spectrum, MultiSpectrum):
        #     self.spectrum.multidata = self.spectrum.multidata.astype(float)
        #
        # self.spectrum.data = self.spectrum.data.astype(float)
        # self.llspectrum.data = self.llspectrum.data.astype(float)
        #
        # if self.spectrum.size > self.llspectrum.size:
        #     print('low loss spectrum has to have same size as model')
        #     print('the low loss will be zero padded to have the same size')
        #     llspectrum = self.padding(spectrum.get_spectrumshape(), llspectrum)
        #     self.llspectrum = llspectrum


        self.iterations = iterations
        self.restored = None

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, n_iter):
        self._iterations = n_iter

    def richardsonlucy(self,signal, psf, iterations):
        imax = psf.argmax()
        result = np.array(signal).copy()
        mimax = psf.size - 1 - imax

        for i in range(iterations):
            first = np.convolve(psf, result)[imax: imax + psf.size]
            result *= np.convolve(psf[::-1], signal /
                                  first)[mimax:mimax + psf.size]

            # plt.figure()
            # plt.plot(result)
            # plt.title(i)

        return result

    # def ISRA(self, signal, psf, iterations):
    #     imax = psf.argmax()
    #     result = np.array(signal).copy()
    #     mimax = psf.size - 1 - imax
    #     mode = 'full'
    #     for i in range(iterations):
    #         # first = np.convolve(psf, result, mode=mode)
    #         first = result.copy()
    #         teller =  np.convolve(signal, psf[::-1], mode=mode)
    #         noemer =  np.convolve(np.convolve(psf, first, mode=mode), psf[::-1], mode=mode)
    #         # update = teller/noemer
    #         # print(update.sum())
    #         # result *= first*update
    #         # print(result.sum())
    #
    #         # plt.figure()
    #         # plt.plot(result)
    #     return result

    def multi_deconvolve(self):
        shape = (self.spectrum.xsize, self.spectrum.ysize)
        ms = self.spectrum.copy()

        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            self.spectrum.setcurrentspectrum(index)
            self.llspectrum.setcurrentspectrum(index)
            restore = self.richardsonlucy(self.spectrum.data, self.llspectrum.data, self.iterations)
            ms.multidata[islice] = restore

        return ms

    def deconvolve(self):
        psf = self.llspectrum.data/self.llspectrum.data.sum()
        signal = self.spectrum.data/self.spectrum.data.sum()
        # restore = self.ISRA(signal, psf, self.iterations)

        if type(self.spectrum) is MultiSpectrum:
            s = self.multi_deconvolve()
            self.restored = s
        else:
            restore = self.richardsonlucy(self.spectrum.data, self.llspectrum.data, self.iterations)
            s = Spectrum(self.spectrum.get_spectrumshape(), data=restore)
            self.restored = s
        return s


    def apply_nothing(self):
        print('how the hell does this work')
































