import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from tqdm import tqdm
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape



class Splice(Operator):
    '''
    This class splines multiple spectra into each other. At this point the spectra should still overlap
    The average is taken of the multiple spectra at this point. Even though they have different noise
    properties
    '''
    def __init__(self, spectra, weights=None):
        self.spectra = spectra

        if isinstance(spectra[0], MultiSpectrum):
            self.check_multispectra_validity(spectra)
            self.has_multispectra = True
        else:
            self.has_multispectra = False


        self.weights = weights


        #todo check if the spectra which are given have the same x, y size and dispersion. Energy does not need to be the same

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights= np.ones(len(self.spectra))
        else:
            self._weights = weights

    # def normalize_weights(self):
    #     print('sum of weights is :'+str(self.weights.sum()))
    #     self.weights = len(self.spectra)*self.weights/self.weights.sum()


    def check_multispectra_validity(self, spectra):
        if len(spectra)<2:
            raise ValueError(r'Not sufficient amount of multispectra in the list')
        initx = spectra[0].xsize
        inity = spectra[0].ysize
        for spectrum in spectra[1:]:
            if (spectrum.xsize != initx) | (spectrum.ysize != inity):
                raise ValueError(r'The scan size of the multispectra are not the same')



    def get_new_energy_axis(self):
        for i in range(len(self.spectra)-1):
            if i == 0:
                offset = min(self.spectra[i].offset, self.spectra[i+1].offset)
                dispersion = min(self.spectra[i].dispersion, self.spectra[i + 1].dispersion)
                end_E = max(self.spectra[i].energy_axis[-1], self.spectra[i+1].energy_axis[-1])

            else:
                offset = min(offset, self.spectra[i+1].offset)
                dispersion = min(dispersion, self.spectra[i + 1].dispersion)
                end_E = max(end_E, self.spectra[i+1].energy_axis[-1])

        E = np.arange(offset, end_E, dispersion)
        return E


    def get_overlapping_region(self, ind0, ind1):
        """
        Find the overlapping region for the two spectra which are in the list with index ind0 and ind1.
        :param ind0:
        :param ind1:
        :return:
        """
        s0 = self.spectra[ind0]
        s1 = self.spectra[ind1]

        dispersion = min(s0.dispersion, s1.dispersion)
        offset = min(s0.offset, s1.offset)
        end_E = max(s0.energy_axis[-1], s1.energy_axis[-1])

        E = np.arange(offset, end_E, dispersion)
        data = np.zeros(E.size)
        data[:s0.size] = s0.data

        return E, data

    def _get_spectrum_from_energy_axis(self, energy_axis):
        """
        Function which makes a spectrum of the enery axis.
        This is needed in the way the function is implemented in spectrum
        :return:
        """
        sh = Spectrumshape(np.diff(energy_axis)[0], energy_axis[0], energy_axis.size)
        s = Spectrum(sh)
        return s, sh

    def _get_multispectrum_from_energy_axis(self, energy_axis):
        """
        Function which makes a spectrum of the enery axis.
        This is needed in the way the function is implemented in spectrum
        :return:
        """
        xsize = self.spectra[0].xsize
        ysize = self.spectra[0].ysize
        sh = MultiSpectrumshape(np.diff(energy_axis)[0], energy_axis[0],
                                energy_axis.size, xsize, ysize)
        s = MultiSpectrum(sh)
        return s, sh


    def splice_spectra(self):
        """
        Multiple spectra can be spliced together. This only works for spectra and not multispectra
        :return:
        """
        E = self.get_new_energy_axis()
        spectrum, sshape = self._get_spectrum_from_energy_axis(E)
        ndata = self._weigth_interpolate(spectrum)
        s = Spectrum(sshape, data=ndata)
        return s

    def _weigth_interpolate(self, spectrum):
        ndata = np.zeros((len(self.spectra), spectrum.size))
        weight_array = np.zeros((len(self.spectra), spectrum.size))
        for ii, spec in enumerate(self.spectra):
            intspec = spec.interp_to_other_energy_axis(spectrum, constant_values=(np.nan, np.nan))
            boolean = np.invert(np.isnan(intspec.data))
            ndata[ii, boolean] = intspec.data[boolean]
            weight_array[ii, boolean] = self.weights[ii]

        res = (ndata * weight_array).sum(0)/weight_array.sum(0)

        return res

    def _avg_interpolate(self, spectrum):
        ndata = np.zeros(spectrum.size)
        weigth_array = np.zeros(ndata.size)
        for ii, spec in enumerate(self.spectra):
            intspec = spec.interp_to_other_energy_axis(spectrum, constant_values=(np.nan, np.nan))
            boolean = np.invert(np.isnan(intspec.data))
            ndata[boolean] += intspec.data[boolean]
            weigth_array[boolean] = 1

        return ndata/weigth_array



    def splice_multispectra(self):
        E = self.get_new_energy_axis()
        multispectrum, sshape = self._get_multispectrum_from_energy_axis(E)
        shape = (multispectrum.xsize, multispectrum.ysize)
        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            for spec in self.spectra:
                spec.setcurrentspectrum(index)
            multispectrum.multidata[islice] = self._weigth_interpolate(multispectrum)

        return multispectrum







