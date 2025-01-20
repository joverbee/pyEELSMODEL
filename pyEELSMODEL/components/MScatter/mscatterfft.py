from pyEELSMODEL.components.MScatter.mscatter import Mscatter
import numpy as np


class MscatterFFT(Mscatter):
    """
    Mutiple scattering using FFT (e.g. to concolve model with LL spectrum)
    """

    def __init__(self, specshape, llspectrum, use_padding=True,padding_mode_data="constant",padding_mode_llspectrum="constant"):
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

        padding_mode_data: string
        Padding mode to use on data, see the different modes in np.pad. (Recommended: "edge")

        padding_mode_llspectrum: string
        Padding mode to use on llspectrum, see the different modes in np.pad. (Recommended: "constant")


        """
        super().__init__(specshape, llspectrum)
        self._setname("Multiple scattering (FFT)")
        self.setdescription("Convolution of the HL spectrum with LL using fast"
                            " fourier transform convolution.\nThis simulates"
                            " the effect of multiple scattering\nwhich is an "
                            "important effect in experimental spectra ")
        self.padding_mode_data=padding_mode_data
        self.padding_mode_llspectrum = padding_mode_llspectrum
        self.padding = llspectrum.size
        self.use_padding = use_padding
        # some cache to not recalculate fourier transfrom when not needed
        # for instance in the calculation of convolved A matrix
        self.new_ll = True

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        if self.use_padding:
            self.calculate_w_padding()
        else:
            self.calculate_raw()

    def calculate_raw(self):
        fmodel = np.fft.rfft(self.data)  # real fourier transform the model
        if self.new_ll:
            self.llspectrum.normalise()
            fll = np.fft.rfft(
                self.llspectrum.data)  # real fourier transform the ll spectrum
            # shift zl peak position to 0!
            # need to compensate for zl peak not being at pix 0
            self.zlindex = self.llspectrum.getmaxindex()
            self.fll = fll
        self.data = np.roll(np.fft.irfft(fmodel * self.fll), -self.zlindex)

    def calculate_w_padding(self):
        """
        Function which adds the zero padding to remove the intensity of the
        end of the model to come into the beginning of the model.
        """
        pds = (self.padding, self.padding)
        # real fourier transform the model
        fmodel = np.fft.rfft(np.pad(self.data, pad_width=pds,mode = self.padding_mode_data))
        if self.new_ll:
            self.llspectrum.normalise()
            llpad = np.pad(self.llspectrum.data, pad_width=pds,mode = self.padding_mode_llspectrum)
            fll = np.fft.rfft(llpad)  # real fourier transform the ll spectrum
            self.fll = fll
            self.zlindex = np.argmax(llpad)

        # conv = np.real(np.fft.ifft(fmodel*fll)[self.padding:-self.padding])
        # conv = np.real(np.fft.ifft(fmodel * self.fll))
        conv = np.fft.irfft(fmodel * self.fll)

        # shift zl peak position to 0!
        # need to compensate for zl peak not being at pix 0
        self.data = np.roll(conv, -self.zlindex)[self.padding:-self.padding]

    def calculate_A_matrix(self, A_matrix):
        """
        Convolution via matrices instead of for loop.
        This function is not faster than the regular convolution and is not
        used in pyEELSMODEL.
        """
        dpad = ((self.padding, self.padding), (0, 0))
        A_matrix_pad = np.pad(A_matrix, pad_width=dpad)
        fftA = np.fft.fft(A_matrix_pad, axis=0)

        self.llspectrum.normalise()
        llpad = np.pad(self.llspectrum.data,
                       pad_width=(self.padding, self.padding))
        self.zlindex = np.argmax(llpad)

        fll = np.fft.fft(llpad)  # real fourier transform the ll spectrum
        fll_matrix = np.tile(fll[:, np.newaxis], (1, A_matrix.shape[1]))

        conv = np.real(np.fft.ifft(fftA * fll_matrix, axis=0))
        conv_A = np.roll(conv, -self.zlindex,
                         axis=0)[self.padding:-self.padding, :]
        return conv_A
