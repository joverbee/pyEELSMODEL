from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.lorentzian import Lorentzian
from pyEELSMODEL.components.voigt import Voigt
from pyEELSMODEL.fitters.lsqfitter import LSQFitter
from pyEELSMODEL.operators.multispectrumvisualizer import \
    MultiSpectrumVisualizer
import logging
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class ZLPRemoval(Operator):
    """
    Removes the ZLP from the spectrum by fitting the appropriate function
    to it
    """

    def __init__(self, spectrum, signal_range=None, model_type='Gaussian'):
        self.spectrum = spectrum
        self.signal_range = signal_range
        self.model_type = model_type
        self.fwhm_boundaries = (self.spectrum.dispersion/2,
                                40*self.spectrum.dispersion)

        self.set_indices()
        self.make_zeroloss_model()
        self.set_start_parameters()

        self.start_parameters = None
        self.fitter = None
        self.model_signal = None

        self.inelastic = None
        self.zlp = None
        self.mirror_zlp = None
        self.mirror_inelastic = None

        self.mirror_ind = 50

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, spectrum):
        self._spectrum = spectrum

    @property
    def signal_range(self):
        return self._signal_range

    @signal_range.setter
    def signal_range(self, signal_range):
        if signal_range is None:
            # todo a smarter guess of the signal range would be good
            signal_range = (self.spectrum.offset,
                            self.spectrum.energy_axis[-1])
        self._signal_range = signal_range

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, model_type):
        self._model_type = model_type

    @property
    def model(self):
        """
        The model used for the fitting
        """
        return self._model

    @model.setter
    def model(self, m0):
        """
        Sets the attribute model to the given model.
        """
        self._model = m0

    @property
    def fitter(self):
        """
        The fitter used in the spectrum (now it is the LSQ since
        it fast and stable)
        """
        return self._fitter

    @fitter.setter
    def fitter(self, fit):
        self._fitter = fit

    def make_zeroloss_model(self):
        """
        Creates a model for the zero loss peak, this depends on which
        model_type is chosen when creating the background object. The model is
        stored in the model attribute
        """
        specshape = self.spectrum.get_spectrumshape()
        m0 = Model(specshape)
        if self.model_type == 'Gaussian':
            comp = Gaussian(specshape, A=1, centre=0, fwhm=1)
        elif self.model_type == 'Lorentzian':
            comp = Lorentzian(specshape, A=1, centre=0, fwhm=1)
        elif self.model_type == 'Voigt':
            comp = Voigt(specshape, A=1, centre=1, gamma=1, sigma=1)
        else:
            print('model type not included in the list of possibilities')

        comp.parameters[1].setboundaries(self.signal_range[0],
                                         self.signal_range[1], force=True)
        comp.parameters[2].setboundaries(self.fwhm_boundaries[0],
                                         self.fwhm_boundaries[1])
        m0.addcomponent(comp)

        self.model = m0

    def estimate_start_param(self):
        start_param = np.zeros(self.model.getnumparameters())
        ind0 = self.indices[0]
        ind1 = self.indices[1]

        start_param[0] = np.max(self.spectrum.data[ind0:ind1])
        co_max = np.argmax(self.spectrum.data[ind0:ind1])+ind0
        start_param[1] = self.spectrum.energy_axis[int(co_max)]
        start_param[2:] = self.spectrum.dispersion*10
        self.start_param = start_param

    def estimate_start_param_multi(self):
        """
        Estimates the starting value of the fitting for a multispectrum when
        having a gaussian or lorentzian model to fit.
        """
        # The lorentzian and gaussian model both have three parameters.
        start_param = np.zeros((self.spectrum.xsize, self.spectrum.ysize,
                                self.model.getnumparameters()))
        ind0 = self.indices[0]
        ind1 = self.indices[1]

        start_param[:, :, 0] = np.max(self.spectrum.multidata, axis=2)

        # this should be zero in principle
        co = np.argmax(self.spectrum.multidata[:, :, ind0:ind1], axis=2)
        shift = (co + ind0)
        start_param[:, :, 1] = self.spectrum.energy_axis[shift]

        # if voigt this is still  fine
        # start_param[:, :, 2:] = 4*self.spectrum.dispersion
        start_param[:, :, 2:] = 1

        self.start_param_multi = start_param

    def set_start_parameters(self):
        if isinstance(self.spectrum, MultiSpectrum):
            self.estimate_start_param_multi()
        else:
            self.estimate_start_param()

    def set_indices(self):
        """
        Calculates the indices used which are excluded in the fit.
        These indices are also used to determine a first guess of the
        background model. The result is stored in the indices attribute

        """
        ind1 = [self.spectrum.get_energy_index(self.signal_range[0]),
                self.spectrum.get_energy_index(self.signal_range[1])]
        self.indices = ind1

    def include_areas(self):
        """
        Sets the exlude of the spectrum such that only the integration
        range is taken into account.
        """
        self.spectrum.set_include_region(self.indices[0], self.indices[1])

    def fit_zlp(self):
        prev_exclude = self.spectrum.exclude[:]
        self.spectrum.exclude = np.ones(self.spectrum.size, dtype=bool)
        self.include_areas()

        fit = LSQFitter(self.spectrum, self.model, use_bounds=True)
        if isinstance(self.spectrum, MultiSpectrum):
            fit.multi_fit(start_param=self.start_param_multi)
            self.fitter = fit
            self.zlp = fit.model_to_multispectrum()
        else:
            for ii, param in enumerate(self.model.getfreeparameters()):
                print(self.start_param[ii])
                param.setvalue(self.start_param[ii])

            fit.perform_fit()
            self.fitter = fit
            self.zlp = Spectrum(self.spectrum.get_spectrumshape(),
                                data=self.model.data)

        self.spectrum.exclude = prev_exclude
        self.inelastic = self.spectrum - self.zlp

    def mirrored_zlp(self):
        ind0 = self.indices[0]
        ind1 = self.indices[1]
        print(isinstance(self.spectrum, MultiSpectrum))
        if isinstance(self.spectrum, MultiSpectrum):
            shift = (np.argmax(self.spectrum.multidata[:, :, ind0:ind1],
                               axis=2)+ind0)
            shape = (self.spectrum.xsize, self.spectrum.ysize)
            ndata = np.zeros((self.spectrum.xsize, self.spectrum.ysize,
                              self.spectrum.size))
            for index in (np.ndindex(shape)):
                isl = np.s_[index]
                ind = max(0, shift[isl]-self.mirror_ind)
                res = self.spectrum.multidata[isl[0], isl[1], ind:shift[isl]+1]

                rest_size = int(self.spectrum.size - 2*res.size - ind + 1)
                ndata[isl] = np.concatenate((np.zeros(ind),
                                             res, np.flip(res[:-1]),
                                             np.zeros(rest_size)))

            mshape = MultiSpectrumshape(self.spectrum.dispersion,
                                        self.spectrum.offset,
                                        self.spectrum.size,
                                        self.spectrum.xsize,
                                        self.spectrum.ysize)
            self.mirror_zlp = MultiSpectrum(mshape, data=ndata)
            self.mirror_inelastic = self.spectrum - self.mirror_zlp

        else:
            shift = np.argmax(self.spectrum.data[ind0:ind1])+ind0
            ind = max(0, shift - self.mirror_ind)
            print(ind)
            res = self.spectrum.data[ind:shift+1]
            rest_size = int(self.spectrum.size - 2*res.size - ind + 1)
            ndata = np.concatenate((np.zeros(ind),
                                    res, np.flip(res[:-1]),
                                    np.zeros(rest_size)))
            self.mirror_zlp = Spectrum(self.spectrum.get_spectrumshape(),
                                       data=ndata)
            self.mirror_inelastic = self.spectrum - self.mirror_zlp

    def show_fit_result(self):
        if isinstance(self.spectrum, MultiSpectrum):
            print('show the multispectrum')
            MultiSpectrumVisualizer([self.spectrum, self.zlp],
                                    labels=['Experimental', 'Fitted ZLP'])
        else:
            self.fitter.plot()

    def show_mirror_result(self):
        if isinstance(self.spectrum, MultiSpectrum):
            print('show the multispectrum')
            MultiSpectrumVisualizer([self.spectrum, self.zlp],
                                    labels=['Experimental', 'Fitted ZLP'])
        else:
            fig, ax = plt.subplots()
            ax.plot(self.spectrum.energy_axis, self.spectrum.data,
                    color='black', marker='o')
            ax.plot(self.spectrum.energy_axis, self.mirror_zlp.data)
