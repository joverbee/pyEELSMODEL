import logging
import numpy as np
import matplotlib.pyplot as plt

from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.operators.zlpremoval import ZLPRemoval
from pyEELSMODEL.core.spectrum import Spectrum
from pyEELSMODEL.core.multispectrum import MultiSpectrum

logger = logging.getLogger(__name__)


class ThicknessEstimator(Operator):
    """
    Calculates the thickness of the material.
    """
    def __init__(self, spectrum, model_type='Lorentzian', signal_range=None):
        self.spectrum = spectrum
        self.model_type = model_type
        self.signal_range = signal_range

    def log_ratio_method(self):
        """
        Calculates the inelastic mean free path by esitmating the zero loss
        peak. The estimated zlp is stored in the .zlp attribute. The inelastic
        mean free path is stored in the .tlambda attribute
        """
        if self.model_type == 'Mirrored':
            zlpremoval = ZLPRemoval(self.spectrum, self.signal_range)
            zlpremoval.mirrored_zlp()
            self.zlpremoval = zlpremoval
            self.zlp = self.zlpremoval.mirror_zlp

        elif self.model_type == 'Vacuum':
            sub = self.spectrum.get_interval(self.signal_range)
            int_vac = self.vacuum.get_interval(self.signal_range).integrate()
            int_sub = sub.integrate()
            fc1 = self.vacuum.data[np.newaxis, np.newaxis, :]
            fc2 = int_sub[:, :, np.newaxis]
            zlp_data = int_vac**(-1) * fc1 * fc2

            self.zlp = MultiSpectrum.from_numpy(zlp_data,
                                                self.spectrum.energy_axis)

        else:
            zlpremoval = ZLPRemoval(self.spectrum, self.signal_range,
                                    self.model_type)
            zlpremoval.fit_zlp()
            self.zlpremoval = zlpremoval
            self.zlp = self.zlpremoval.zlp

        It = self.spectrum.multidata.sum(2)  # total integral
        Izlp = self.zlp.multidata.sum(2)  # zero loss integral
        self.tlambda = np.log(It / Izlp)

    def extract_vacuum(self, threshold, show_inelastic=True):
        """
        It can happen that vacuum is present and this can be used to
        estimate the thickness. The vacuum spectrum is stored as the
        .vacuum attribute

        Parameters
        ----------
        threshold : 0 < float < 1
            The number of times the spectrum needs to be upsampled
        show_inelastic: boolean
            Indicates whether the inelastic signal used for vacuum
            determination is used.
        """
        fwhm = self.spectrum.get_numerical_fwhm()
        print('FWHM is {} eV'.format(fwhm))
        print('Integral for inelastic signal start at {} eV'.format(5*fwhm))
        Ei = 3*fwhm
        Ef = self.spectrum.energy_axis[-1]
        inelastic = self.spectrum.integrate((Ei, Ef))

        co = int(threshold * inelastic.size)
        thres = np.sort(inelastic.flatten())[co]
        boolean = inelastic < thres

        avg = self.spectrum.multidata[boolean].mean(0)
        s = Spectrum.from_numpy(avg, self.spectrum.energy_axis)
        self.vacuum = s

        if show_inelastic:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(inelastic)
            ax[1].imshow(boolean)
            ax[2].plot(self.vacuum.energy_axis, self.vacuum.data,
                       label='vacuum')
            ax[2].plot(self.spectrum.energy_axis, self.spectrum.mean().data,
                       label='average')
            ax[2].legend()

    # def zlp_integral(self):
    #     """
    #     Calculates the integral of the ZLP. If an analytical
    #     function is used for the fitting, this expression can
    #     used. Else a numerical summation is used. Note that
    #     a better implementation of the integral can be developed.
    #     """
    #     if type(self.spectrum) is MultiSpectrum:
    #         A = self.zlpremoval.fitter.coeff_matrix[:, :, 0]
    #         width = self.zlpremoval.fitter.coeff_matrix[:, :, 2] #fwhm
    #     elif type(self.spectrum) is Spectrum:
    #         A = self.zlpremoval.fitter.coeff[0]
    #         width = self.zlpremoval.fitter.coeff[2]
    #
    #     if self.model_type == 'Gaussian':
    #         sigma = np.abs(width) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    #         zlp_integral = np.sqrt(2) * A * sigma
    #
    #     elif self.model_type == 'Lorentzian':
    #         a = 1
    #     elif self.model_type == 'Voigt':
    #         a = 1
    #     elif self.model_type == 'Mirrored':
    #         zlp = self.zlpremoval.mirror_zlp.sum()
    #         zlp_integral = zlp.integrate()
