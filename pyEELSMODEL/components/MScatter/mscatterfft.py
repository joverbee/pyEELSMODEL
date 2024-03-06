from pyEELSMODEL.components.MScatter.mscatter import Mscatter
import numpy as np


class MscatterFFT(Mscatter):
    """
    Mutiple scattering using FFT (e.g. to concolve model with LL spectrum)
    """

    def __init__(self, specshape, llspectrum, use_padding=True):
        """
        Parameters
        ----------
        specshape: Spectrumshape
            The spectrum shape on the spectrum it will be used, not the low
            loss spectrum
        llspectrum: Spectrum or MultiSpectrum
            The spectrum or multispectrum which is used to convolve the rest
            of the components with.
        use_padding: bool
            Indicates if both spectrum data and low loss are zero padded to
            reducde artifacts coming from the FFT. If True, the calculations
            take longer but are more precise. (default: True)

        """
        super().__init__(specshape, llspectrum)
        self._setname("Multiple scattering (FFT)")
        self.setdescription("Convolution of the HL spectrum with LL using fast"
                            " fourier transform convolution.\nThis simulates"
                            " the effect of multiple scattering\nwhich is an "
                            "important effect in experimental spectra ")

        self.padding = llspectrum.size
        self.use_padding = use_padding

    def calculate(self):
        if self.use_padding:
            self.calculate_w_padding()
        else:
            self.calculate_raw()

    def calculate_raw(self):
        fmodel = np.fft.rfft(self.data)  # real fourier transform the model
        self.llspectrum.normalise()
        fll = np.fft.rfft(
            self.llspectrum.data)  # real fourier transform the ll spectrum
        # shift zl peak position to 0!
        # need to compensate for zl peak not being at pix 0
        zlindex = self.llspectrum.getmaxindex()
        self.data = np.roll(np.fft.irfft(fmodel * fll), -zlindex)

    def calculate_w_padding(self):
        """
        Function which adds the zero padding to remove the intensity of the
        end of the model to come into the beginning of the model.
        """

        # real fourier transform the model
        fmodel = np.fft.fft(np.pad(self.data,
                                   pad_width=(self.padding, self.padding)))

        self.llspectrum.normalise()
        llpad = np.pad(self.llspectrum.data,
                       pad_width=(self.padding, self.padding))
        fll = np.fft.fft(llpad)  # real fourier transform the ll spectrum

        # conv = np.real(np.fft.ifft(fmodel*fll)[self.padding:-self.padding])
        conv = np.real(np.fft.ifft(fmodel * fll))

        # shift zl peak position to 0!
        # need to compensate for zl peak not being at pix 0
        zlindex = np.argmax(llpad)
        self.data = np.roll(conv, -zlindex)[self.padding:-self.padding]
