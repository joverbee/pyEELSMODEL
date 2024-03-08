import numpy as np
from tqdm import tqdm
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape


class Splice(Operator):
    """
    This class splines multiple spectra into each other. At this point the
    spectra should still overlap.
    The splicing first calculates a new energy axis which conveys each
    spectrum added to the spectra list.
    Then each spectrum is interpolated to the new axis and the weighted
    average is calcualted with the information on the acquisition time.
    This assumes that each spectrum is acquired with the
    same current.
    It is not needed for the dispersion to be the same. When this happens it
    uses the highest dispersion (lowest value) for the interpolation.


    """
    def __init__(self, spectra, acq_times=None):
        """
        Initializes the Splice object.

        Parameters
        ----------
        spectra: list of Spectrums or MultiSpectrums
            The list containing the spectra which need to be spliced together.
            If Multispectrums are used then it is important that the xsize and
            ysize are the same.
        acq_times: list
            A list of the acquisition times. The acquistion times do not need
            to be in absolute units (such as seconds) but the ratios should be
            conserved.


        """
        self.spectra = spectra

        if isinstance(spectra[0], MultiSpectrum):
            self.check_multispectra_validity(spectra)
            self.has_multispectra = True
        else:
            self.has_multispectra = False

        self.acq_times = acq_times
        self.weights = self.acq_times

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights = np.ones(len(self.spectra))
        else:
            self._weights = weights

    # def normalize_weights(self):
    #     print('sum of weights is :'+str(self.weights.sum()))
    #     self.weights = len(self.spectra)*self.weights/self.weights.sum()

    def check_multispectra_validity(self, spectra):
        if len(spectra) < 2:
            raise ValueError(r'Not sufficient amount of multispectra'
                             r' in the list')
        initx = spectra[0].xsize
        inity = spectra[0].ysize
        for spectrum in spectra[1:]:
            if (spectrum.xsize != initx) | (spectrum.ysize != inity):
                raise ValueError(r'The scan size of the multispectra '
                                 r'are not the same')

    def get_new_energy_axis(self):
        """
        Determines the lowest energy, highest energy and highest dispersion
        from the list of spectra provided. From this information it
        calculates a new energy axis E

        Returns
        -------
        E: 1d numpy array
            The new energy axis on which each spectrum will be interpolated.

        """

        for i in range(len(self.spectra)-1):
            if i == 0:
                offset = min(self.spectra[i].offset,
                             self.spectra[i+1].offset)
                dispersion = min(self.spectra[i].dispersion,
                                 self.spectra[i + 1].dispersion)
                end_E = max(self.spectra[i].energy_axis[-1],
                            self.spectra[i+1].energy_axis[-1])

            else:
                offset = min(offset, self.spectra[i+1].offset)
                dispersion = min(dispersion, self.spectra[i + 1].dispersion)
                end_E = max(end_E, self.spectra[i+1].energy_axis[-1])

        E = np.arange(offset, end_E, dispersion)
        return E

    def _get_spectrum_from_energy_axis(self, energy_axis):
        """
        Function which makes a spectrum of the enery axis.
        This is needed in the way the function is implemented in spectrum

        """
        sh = Spectrumshape(np.diff(energy_axis)[0], energy_axis[0],
                           energy_axis.size)
        s = Spectrum(sh)
        return s, sh

    def _get_multispectrum_from_energy_axis(self, energy_axis):
        """
        Function which makes a spectrum of the enery axis.
        This is needed in the way the function is implemented in spectrum
        """
        xsize = self.spectra[0].xsize
        ysize = self.spectra[0].ysize
        sh = MultiSpectrumshape(np.diff(energy_axis)[0], energy_axis[0],
                                energy_axis.size, xsize, ysize)
        s = MultiSpectrum(sh)
        return s, sh

    def splice_spectra(self):
        """
        Multiple spectra can be spliced together. This only works for spectra
        and not multispectra
        """
        E = self.get_new_energy_axis()
        spectrum, sshape = self._get_spectrum_from_energy_axis(E)
        ndata = self._weigth_interpolate(spectrum)
        s = Spectrum(sshape, data=ndata)
        return s

    def _weigth_interpolate(self, s):
        ndata = np.zeros((len(self.spectra), s.size))
        # weight_array = np.zeros((len(self.spectra), spectrum.size))
        for ii, spec in enumerate(self.spectra):
            intspec = spec.interp_to_other_energy_axis(s, (np.nan, np.nan))
            boolean = np.invert(np.isnan(intspec.data))
            ndata[ii, boolean] = intspec.data[boolean]/self.acq_times[ii]
            # weight_array[ii, boolean] = self.weights[ii]

        res = (ndata * self.weight_array).sum(0)/self.weight_array.sum(0)

        return res

    def _calculate_weight_array(self, s):

        weight_array = np.zeros((len(self.spectra), s.size))
        for ii, spec in enumerate(self.spectra):
            intspec = spec.interp_to_other_energy_axis(s, (np.nan, np.nan))
            boolean = np.invert(np.isnan(intspec.data))
            weight_array[ii, boolean] = self.weights[ii]

        self.weight_array = weight_array

    def splice_multispectra(self):
        E = self.get_new_energy_axis()
        multispectrum, sshape = self._get_multispectrum_from_energy_axis(E)
        self._calculate_weight_array(multispectrum)
        shape = (multispectrum.xsize, multispectrum.ysize)
        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            for spec in self.spectra:
                spec.setcurrentspectrum(index)
            multispectrum.multidata[islice] =\
                self._weigth_interpolate(multispectrum)

        return multispectrum
