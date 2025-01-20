from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
import numpy as np


class Mscatter(Component):
    """
    A convolutor component can take the input of another component and
    apply a convolution operator to it and store the results of that operation
    in its own spectrum data. The model will treat this as special and first
    calculate the individual components that have canconvolute set, sum those
    and then apply these as input to a convolutor component (typically
    only 1 in a model). After this those components are added which have
    canconvolute=False (eg a background component)
    """

    def __init__(self, specshape, llspectrum):
        """

        Parameters
        ---------
        specshape: Spectrumshape
            The spectrum shape on the spectrum it will be used, not the low
            loss spectrum
        llspectrum: Spectrum or MultiSpectrum
            The spectrum or multispectrum which is used to convolve the rest
            of the components with.

        """
        super().__init__(specshape)

        n_acc = 8  # sometimes round error make that disperion is not the same
        if np.round(self.dispersion, n_acc) != np.round(llspectrum.dispersion,
                                                        n_acc):
            raise ValueError('low loss spectrum has to have same '
                             'dispersion as model')

        if self.size < llspectrum.size:
            raise ValueError('size of ll spectrum is larger than the spectrum'
                             ' on which the analysis will be performed')

        elif self.size > llspectrum.size:
            # print('low loss spectrum has to have same size as model')
            # print('the low loss will be zero padded to have the same size')
            llspectrum = self.padding(specshape, llspectrum)

        self._setcanconvolute(False)  # meaningless in this case
        self._setshifter(False)
        self._isconvolutor = True
        self.llspectrum = llspectrum

    def padding(self, specshape, llspectrum):
        """
        Zero pads the low loss spectrum to have the same size as the spectrum
        shape.
        Parameters
        ---------
        specshape: Spectrumshape
            The spectrum shape on the spectrum it will be used, not the low
            loss spectrum
        llspectrum: Spectrum or MultiSpectrum
            The spectrum or multispectrum which is used to convolve the rest
            of the components with.
        """
        size_dif = specshape.size - llspectrum.size
        before = size_dif // 2
        after = size_dif - before
        # print('padding is done')
        # print(llspectrum)
        if type(llspectrum) is Spectrum:
            # print('low loss is spectrum')
            pad_data = np.pad(llspectrum.data, pad_width=(before, after))
            noffset = llspectrum.offset - llspectrum.dispersion * before
            sph = Spectrumshape(llspectrum.dispersion, noffset, pad_data.size)
            s = Spectrum(sph, data=pad_data)

        elif type(llspectrum) is MultiSpectrum:
            # print('low loss is multispectrum')
            pad_data = np.pad(llspectrum.multidata,
                              pad_width=((0, 0), (0, 0), (before, after)))
            noffset = llspectrum.offset - llspectrum.dispersion * before
            sph = MultiSpectrumshape(llspectrum.dispersion, noffset,
                                     pad_data.shape[-1],
                                     llspectrum.xsize, llspectrum.ysize)
            s = MultiSpectrum(sph, data=pad_data)

        return s

    def save(self, fh):
        # save the details of this component to file in such a way
        # that you can bring it to the same state later

        # save the file location of the ll spectrum to open it later
        # when reloading the model

        return

    def load(self, fh):
        # load this component from saved detaile

        # get filename
        # open the ll spectrum if not already open

        # self.llspectrum=...
        return

    def calculate(self):
        print(
            'if this statement is printed, it means that no '
            'calculate function exists for other MScatter components')
