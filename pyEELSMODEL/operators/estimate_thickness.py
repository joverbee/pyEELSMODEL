import logging
import numpy as np

from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum
from pyEELSMODEL.operators.zlpremoval import ZLPRemoval



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
        Calculates the
        """
        if self.model_type=='Mirrored':
            zlpremoval = ZLPRemoval(self.spectrum, self.signal_range)
            zlpremoval.mirrored_zlp()
        else:
            zlpremoval = ZLPRemoval(self.spectrum, self.signal_range, self.model_type)
            zlpremoval.fit_zlp()

        self.zlpremoval = zlpremoval
        # zlp_int = self.zlp_integral()

        if self.model_type == 'Mirrored':
            inelastic_sig = zlpremoval.mirror_inelastic.integrate((5., 2000.))

            self.tlambda = (inelastic_sig / zlpremoval.mirror_zlp.multidata.sum(2))

        else:
            inelastic_sig = zlpremoval.inelastic.integrate((5., 2000.))
            self.tlambda = (inelastic_sig/zlpremoval.zlp.multidata.sum(2))


    def zlp_integral(self):
        """
        Calculates the integral of the ZLP. If an analytical
        function is used for the fitting, this expression can
        used. Else a numerical summation is used. Note that
        a better implementation of the integral can be developed.
        """
        if type(self.spectrum) is MultiSpectrum:
            A = self.zlpremoval.fitter.coeff_matrix[:, :, 0]
            width = self.zlpremoval.fitter.coeff_matrix[:, :, 2] #fwhm
        elif type(self.spectrum) is Spectrum:
            A = self.zlpremoval.fitter.coeff[0]
            width = self.zlpremoval.fitter.coeff[2]

        if self.model_type == 'Gaussian':
            sigma = np.abs(width)/(2.0*np.sqrt(2.0*np.log(2.0)))
            zlp_integral = np.sqrt(2) * A * sigma

        elif self.model_type == 'Lorentzian':
            a=1
        elif self.model_type == 'Voigt':
            a=1
        elif self.model_type == 'Mirrored':
            zlp = self.zlpremoval.mirror_zlp
            zlp_integral = zlp.integrate()






