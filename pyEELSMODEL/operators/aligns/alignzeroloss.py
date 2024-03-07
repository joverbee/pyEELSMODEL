# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import logging

from pyEELSMODEL.core.model import Model

from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.lorentzian import Lorentzian

from pyEELSMODEL.fitters.lsqfitter import LSQFitter
from pyEELSMODEL.operators.aligns.align import Align

logger = logging.getLogger(__name__)


class AlignZeroLoss(Align):
    """
    AlignZeroLoss is a class which aligns the zero loss peak and other spectra
    can be added which also need to be aligned.
    The zero loss is fitted to a model (Gaussian or Lorentzian) and this
    parameter will be used to align the zero loss. Note that some interpolation
    step is used to shift the data. This could introduce differences
    in noise properties when wanting to perform the most accurate statistics
    on it.
    """
    def __init__(self, multispectrum, other_spectra=None,
                 signal_range=None, model_type='Gaussian', cropping=False,
                 use_bounds=True):
        """

        Parameters
        ----------
        multispectrum: Multispectrum
            The spectrum from which the background should be removed
        other_spectra: List of Multispectra
            The other multispectra which can be aligned with the same
            parameters. (default: None)
        signal_range: tuple
            Indicates the region on which the zero loss fitting should be
            performed.
            (default: None)
        model_type: string
            Selecting which model is used to fit the zero loss peak
             (default: 'Gaussian')
        cropping: bool
            Indicates if the edges of the signal are cropped out when aligning
            the zero loss peak.
            (default: False)
        use_bounds: bool
            Use the boundaries for the fitting. This could make the fit more
            stable.
            (default: True)
        """

        super().__init__(multispectrum, other_spectra, cropping,
                         signal_range=signal_range)

        self.model_type = model_type
        self.use_bounds = use_bounds

        self.aligned = None  # attribute which stores the aligned multispectrum
        self.aligned_others = []  # attribute
        self.start_parameters = None
        self._fitter = None
        self.model = None
        self.make_zeroloss_model()
        self.method = 'Model based fitting'

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
        The fitter used in the spectrum (now it is the LSQ since it fast
        and stable)
        """
        return self._fitter

    @fitter.setter
    def fitter(self, fit):
        self._fitter = fit

    def estimate_start_parameters(self):
        """
        Estimates the starting values for every scan position to fit the zero
        loss peak. The A value is found by taking the maximum.
        The centre is chosen as the coordinate at which the maximum occurs
        Sigma is taken as a fixed value of 1. (better guess can be tried but it
         seems to be stable)
        It stores the result is the start_parameters attribute.
        """
        # The lorentzian and gaussian model both have three parameters.
        start_param = \
            np.zeros((self.multispectrum.xsize, self.multispectrum.ysize, 3))

        start_param[:, :, 0] = np.max(self.multispectrum.multidata, axis=2)

        # this should be zero in principle
        ind0 = self.multispectrum.get_energy_index(self.signal_range[0])
        ind1 = self.multispectrum.get_energy_index(self.signal_range[1])
        co = np.argmax(self.multispectrum.multidata[:, :, ind0:ind1], axis=2)
        shift = (co + ind0)
        start_param[:, :, 1] = self.multispectrum.energy_axis[shift]

        # fwhm is guess from the width of the summed spectrum.
        # this says that the width does not change over the scan
        # (which is normally true)
        # zl = self.fast_aligned.multidata.sum((0,1))
        # coor = np.argwhere(zl>0.5*np.max(zl))

        start_param[:, :, 2] = 1

        self.start_parameters = start_param

    def set_indices(self):
        """
        Calculates the indices used which are excluded in the fit.
        These indices are also used to determine a first guess of the
        background model. The result is stored in the indices attribute

        """

        ind1 = [self.multispectrum.get_energy_index(self.signal_range[0]),
                self.multispectrum.get_energy_index(self.signal_range[1])]
        self.indices = ind1

    def make_zeroloss_model(self):
        """
        Creates a model for the zero loss peak, this depends on which
        model_type is chosen when creating the background object. The model
        is stored in the model attribute
        """
        specshape = self.multispectrum.get_spectrumshape()
        m0 = Model(specshape)
        if self.model_type == 'Gaussian':
            comp = Gaussian(specshape, A=1, centre=0, fwhm=1)
        elif self.model_type == 'Lorentzian':
            comp = Lorentzian(specshape, A=1, centre=0, fwhm=1)
        else:
            print('model type not included in the list of possibilities')

        comp.parameters[1].setboundaries(self.signal_range[0],
                                         self.signal_range[1], force=True)

        m0.addcomponent(comp)
        self.model = m0

    def include_areas(self):
        """
        Sets the exlude of the spectrum such that only the integration
        range is taken into account.
        """
        self.multispectrum.set_include_region(self.indices[0], self.indices[1])

    def calculate_model(self):
        """
        Fits the model to the data. The fitting region is resetted as it
        initially was after the fit.
        """

        prev_exclude = self.multispectrum.exclude[:]
        self.multispectrum.exclude = \
            np.ones(self.multispectrum.size, dtype=bool)
        self.set_indices()
        self.include_areas()

        # fit = MLFitter(self.spectrum, self.get_model())
        fit = LSQFitter(self.multispectrum, self.model, use_bounds=True)

        # if the start parameters are not yet calculated do it before the
        # model fitting
        if self.start_parameters is None:
            print('Estimates the parameters for the fitting procedure')
            self.estimate_start_parameters()

        fit.multi_fit(start_param=self.start_parameters)
        self.fitter = fit

        self.multispectrum.exclude = prev_exclude
        self.shift = fit.coeff_matrix[:, :, 1]

    def perform_alignment(self):
        """
        Performs the alignment procedure by first calculating the shift and
        then applying
        it to the spectra.

        Returns
        -------
        None.

        """
        self.calculate_model()
        self.align()
