# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.core.spectrum import Spectrum

from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.fast_background import FastBG2
from pyEELSMODEL.components.polynomial import Polynomial
from pyEELSMODEL.components.exponential import Exponential
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import\
    ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.fitters.lsqfitter import LSQFitter
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.fitters.minimizefitter import MinimizeFitter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BackgroundRemoval(Operator):
    """
    BackgroundRemoval is a class which needs a spectrum and signal range.
    This class provides a workflow on going extracting the elemental map
    of one edge.
    """

    def __init__(self, spectrum, signal_range, model_type='Powerlaw',
                 linear_fitting=False, order=2,
                 non_linear_fitter='LSQ', r_values=(2, 3)):
        """
        Initiates the BackgroundRemoval object

        Parameters
        ----------
        spectrum: Spectrum
            The spectrum from which the background should be removed
        signal_range: tuple
            Indicates the region on which the background fitting should be
            performed. This object can perform the fitting on two area and
            then two tuples should be given.
        model_type: str
            Indicates which model is used to perform the background fitting.
            At this point Powerlaw, FastBG, Exponential en Polynomial is
            implemented. (default: Powerlaw)
        linear_fitting: bool
            Indicates if a linear fitting procedure is used to determine the
            background. This can only work if the model only includes linear
            parameters such as for FastBG or Polynomial.
            (default: False)
        order: int
            If the model_type is 'Polynomial', the order value indicates the
            order of the polynomial. (default:2)
        non_linear_fitter: 'ML' or 'LSQ'
            The type of nonlinear fitter used in the fit. (default: LSQ)
        r_values: tuple of ints
            When the background model is the FastBG, the r values can be
            chosen. (default: (2,3))

        """

        self.spectrum = spectrum
        self.signal_range = signal_range
        self.model_type = model_type
        self.do_linear_fitting = linear_fitting
        self.non_linear_fitter = non_linear_fitter
        self.order = order
        self._r_values = r_values

        self.set_indices()
        self.make_background_model()
        self.set_fit_type()
        self.progress = 0
        self.result = None
        self._fast_fit = None
        self._multi_fast_fit = None
        self.multi_model_signal = None
        self.result = None

    @property
    def signal_range(self):
        return self._signal_range

    @signal_range.setter
    def signal_range(self, signal_range):
        """
        Set the signal range and first validates if the given signal_range is
        valid. The signal range can have two regions such that a interpolation
        on the background can be performed.

        Parameters
        ----------
        signal_range : tuple
            The signal range over which to fit the background.
        """
        # check if the signal range contains one or two ranges
        if any(isinstance(i, tuple) for i in signal_range):
            self.two_area = True
        else:
            self.two_area = False

        if not self.two_area:
            if signal_range[0] < self.spectrum.energy_axis[0] or \
                    signal_range[1] > self.spectrum.energy_axis[-1]:
                raise ValueError(r'Invalid signal range')
            if signal_range[0] >= signal_range[1]:
                raise ValueError(r'First number of signal range'
                                 r' must be smaller than the second value')

        else:
            for tup in signal_range:
                if tup[0] < self.spectrum.energy_axis[0] or \
                        tup[1] > self.spectrum.energy_axis[-1]:
                    raise ValueError(r'Invalid signal range')
                if tup[0] >= tup[1]:
                    raise ValueError(r'First number of signal range must'
                                     r' be smaller than the second value')

            if signal_range[0][1] > signal_range[1][0]:
                raise ValueError(r'The end energy of the first energy range '
                                 r'is larger than the beginning of the second')

        self._signal_range = signal_range

    @property
    def model_type(self):
        return self._model_type

    def getprogress(self):
        # returns an int from 0 to 100 indicating the progress in a multifit
        return self.fitter.progress

    def stop(self):
        # stop the fitting process
        self.fitter.stop = True

    @model_type.setter
    def model_type(self, model_type):
        model_list = ['Powerlaw', 'FastBG', 'Polynomial', 'Exponential']
        if model_type not in model_list:
            raise ValueError(r'Invalid model type')

        self._model_type = model_type

    @property
    def do_linear_fitting(self):
        return self._do_linear_fitting

    @do_linear_fitting.setter
    def do_linear_fitting(self, bool):
        if bool and (self.model_type != 'FastBG' and
                     self.model_type != 'Polynomial'):
            raise ValueError(r'This model type cannot perform a linear fit '
                             r'over the background. Set linear fitting to '
                             r'false if you want to use this type of'
                             r' background model')

        self._do_linear_fitting = bool

    @property
    def non_linear_fitter(self):
        return self._non_linear_fitter

    @non_linear_fitter.setter
    def non_linear_fitter(self, non_linear_fitter):
        non_list = ['LSQ', 'ML']
        if non_linear_fitter not in non_list:
            raise ValueError(r'Given non linear fitter is not implemented')

        self._non_linear_fitter = non_linear_fitter

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        if type(order) is not int:
            raise TypeError(r'Order should be integer')
        if order < 0:
            raise ValueError(r'Order should be a positive integer')

        self._order = order

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m0):
        self._model = m0

    @property
    def fitter(self):
        return self._fitter

    @fitter.setter
    def fitter(self, fit):
        self._fitter = fit

    def set_indices(self):
        """
        Calculates the indices used which are excluded in the fit.
        These indices are also used to determine a first guess of the
        background model.
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.two_area:
            ind1 = [self.spectrum.get_energy_index(self.signal_range[0][0]),
                    self.spectrum.get_energy_index(self.signal_range[0][1])]
            ind2 = [self.spectrum.get_energy_index(self.signal_range[1][0]),
                    self.spectrum.get_energy_index(self.signal_range[1][1])]
            self.indices = [ind1, ind2]
        else:
            ind1 = [self.spectrum.get_energy_index(self.signal_range[0]),
                    self.spectrum.get_energy_index(self.signal_range[1])]
            self.indices = ind1

    def make_background_model(self):
        """
        Creates a model for the background, this depends on which model_type
        is chosen when creating the background object.
        The model will be set and given as an attribute to this object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        specshape = self.spectrum.get_spectrumshape()
        m0 = Model(specshape)
        if self.model_type == 'Powerlaw':
            comp = PowerLaw(specshape, A=1, r=3)
        elif self.model_type == 'FastBG':
            comp = FastBG2(specshape, A1=1, r1=self._r_values[0], A2=1,
                           r2=self._r_values[1])
        elif self.model_type == 'Polynomial':
            comp = Polynomial(specshape, order=self.order)
        elif self.model_type == 'Exponential':
            comp = Exponential(specshape, A=1, b=-0.001)
        m0.addcomponent(comp)
        self.model = m0

    @property
    def fast_fit(self):
        return self._fast_fit

    @fast_fit.setter
    def fast_fit(self, param):
        self._fast_fit = param

    @property
    def multi_fast_fit(self):
        return self._multi_fast_fit

    @multi_fast_fit.setter
    def multi_fast_fit(self, param):
        self._multi_fast_fit = param

    def determine_fast_fit_parameters(self):
        """
        Performs the autofit over one spectrum, if the parameters need to be
        determined for a Multispectrum then the
        determine_multi_fast_fit_parameters() should be used.
        This is used as starting parameter for the more advanced fitting.
        Fast fitting only works for the powerlaw and exponential background
        models
        """
        if self.two_area:
            indic = [self.indices[0][0], self.indices[0][1]]
        else:
            indic = [self.indices[0], self.indices[1]]

        if self.model_type == 'Powerlaw' or self.model_type == 'Exponential':
            self.model.getcomponents()[0].autofit(self.spectrum,
                                                  indic[0],
                                                  indic[1])

            self.fast_fit = ([self.model.getfreeparameters()[0].getvalue(),
                              self.model.getfreeparameters()[1].getvalue()])

    def determine_multi_fast_fit_parameters(self):
        """
        Performs the autofit over the entire multispectrum.
        This is used as starting parameter for the more advanced fitting.
        Fast fitting only works for the powerlaw and exponential background
        models
        """
        print(isinstance(self.spectrum, MultiSpectrum))
        if not isinstance(self.spectrum, MultiSpectrum):
            raise ValueError(r'Spectrum used is no MultiSpectrum')

        if self.two_area:
            indic = [self.indices[0][0], self.indices[0][1]]
        else:
            indic = [self.indices[0], self.indices[1]]

        if self.model_type == 'Powerlaw' or self.model_type == 'Exponential':
            fast_matrix = np.empty((self.spectrum.xsize,
                                    self.spectrum.ysize, 2))
            for index in tqdm(np.ndindex(fast_matrix.shape[:-1])):
                islice = np.s_[index]
                self.spectrum.setcurrentspectrum(index)
                self.model.getcomponents()[0].autofit(self.spectrum,
                                                      indic[0], indic[1])
                A = self.model.getfreeparameters()[0].getvalue()
                r = self.model.getfreeparameters()[1].getvalue()
                fast_matrix[islice + (0,)] = A
                fast_matrix[islice + (1,)] = r

            self.multi_fast_fit = fast_matrix

        else:
            print('This model type does not support a fast fit method. '
                  'Only works for Powerlaw or Exponential')

    def include_areas(self):
        """
        Sets the exlude of the spectrum such that only the integration
        range is taken into account. It depends on the type of integration,
        it can be two area or just one area before the edge onset.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.spectrum.exclude = np.ones(self.spectrum.size, dtype=bool)
        if self.two_area:
            self.spectrum.set_include_region(self.indices[0][0],
                                             self.indices[0][1])
            self.spectrum.set_include_region(self.indices[1][0],
                                             self.indices[1][1])
        else:
            self.spectrum.set_include_region(self.indices[0], self.indices[1])

    def set_fit_type(self):
        """
        Set the fitter type for the background subtraction
        """
        if self.do_linear_fitting:
            fit = LinearFitter(self.spectrum, self.model)
        else:
            if self.non_linear_fitter == 'ML':
                fit = MinimizeFitter(self.spectrum, self.model)
                fit.method = 'SLSQP'
                # fit.estimator = fit.lsq_function

            elif self.non_linear_fitter == 'LSQ':
                fit = LSQFitter(self.spectrum, self.model,
                                use_bounds=True, method='trf')
        self.fitter = fit

    def calculate(self):
        """
        Returns the background subtracted spectrum. The exclude of the
        spectrum will be set the same as it was before this function was
        called. A least squares approach is used to fit the background.

        Returns
        -------
        self.results: Spectrum
            The background subtracted spectrum

        """
        prev_exclude = self.spectrum.exclude[:]
        self.set_indices()
        self.include_areas()
        # if self.fast_fit is None:
        #     logger.info('Perform the fast fitting to estimate parameters')
        self.determine_fast_fit_parameters()
        self.fitter.perform_fit()
        self.fitter.set_fit_values()
        self.spectrum.exclude = prev_exclude  # make the exclude is reset

        # if return_raw:
        #     return self.model.data
        # else:
        data = self.spectrum.data - self.model.data
        sback = Spectrum(self.spectrum.get_spectrumshape(), data=data)
        self.result = sback  # maybe get and setter?
        return self.result

    def calculate_multi(self):
        """
        Returns the background subtracted multispectrum. The exclude of the
        spectrum will be set the same as it was before this function was
        called.

        Returns
        -------
        self.results: MultiSpectrum
            The background subtracted multispectrum

        """

        prev_exclude = self.spectrum.exclude[:]
        self.set_indices()
        self.include_areas()

        if self.multi_fast_fit is None:
            logger.info('Perform the fast fitting to estimate parameters')
            self.determine_multi_fast_fit_parameters()

        self.fitter.multi_fit(start_param=self.multi_fast_fit)
        self.multi_model_signal = self.fitter.model_to_multispectrum()

        self.spectrum.exclude = prev_exclude  # make the exclude is reset
        multi_back = self.spectrum.copy()
        multi_back.multidata = self.spectrum.multidata \
            - self.get_background_model().multidata
        multi_back.setcurrentspectrum((0, 0))

        self.result = multi_back
        return self.result

    def get_background_model(self):
        return self.multi_model_signal

    def get_fitting_parameters(self):
        """
        Returns the fitting parameters
        :return:
        """
        return self.fitter.coeff_matrix

    def fast_calculate_multi(self):
        """
        Uses the parameters estimated from the autofit to remove the background
        this only works if the model_type is Powerlaw or Exponential

        Returns
        -------
        self.fast_result: MultiSpectrum
            The background subtracted multispectrum using the fast fitting
            parameters.

        """

        prev_exclude = self.spectrum.exclude[:]
        self.set_indices()
        self.include_areas()

        if self.multi_fast_fit is None:
            logger.info('Determination of fast fit paramters'
                        ' is peformed first')
            self.determine_multi_fast_fit_parameters()

        fast_back = self.spectrum.copy()  # background subtracted
        rem_sig = self.spectrum.copy()  # calculated background
        ndata = np.empty((self.spectrum.xsize,
                          self.spectrum.ysize, self.spectrum.size))
        rem_dat = np.empty(ndata.shape)
        for index in tqdm(np.ndindex(fast_back.multidata.shape[:-1])):
            islice = np.s_[index]

            A = self.multi_fast_fit[islice+(0,)]
            r = self.multi_fast_fit[islice+(1,)]

            self.model.getfreeparameters()[0].setvalue(A)
            self.model.getfreeparameters()[1].setvalue(r)
            self.model.calculate()
            ndata[islice] = self.spectrum.multidata[islice] - self.model.data
            rem_dat[islice] = self.model.data

        fast_back.multidata = ndata
        rem_sig.multidata = rem_dat
        self.spectrum.exclude = prev_exclude  # make the exclude is reset
        self.multi_model_signal = rem_sig
        self.fast_result = fast_back
        return fast_back

    def show_fit_result(self, index=(0, 0), integration_range=None,
                        use_mean=False):
        """
        Shows the background fitting result. Is usefull for showing the result
        of the procedure to identify problems or being able to optimize the
        signal_range.

        Parameters
        ----------
        index: tuple
            If a multispectrum is used, then the index inputted will be used
            for the visualization
        integration_range: tuple
            Input the expected integration range to get the elemental
            abundance. Just used for visualization purpose.
        use_mean: boolean
            If a multispectrum is used, the average results can be shown which
            could be useful when wanting to decrease the noise level in the
            individual spectrum.

        Returns
        -------
        fig: Figure
            The created figure
        """
        prev_exclude = self.spectrum.exclude[:]
        self.set_indices()
        self.include_areas()

        if isinstance(self.spectrum, MultiSpectrum):
            self.fitter.setcurrentfit(index)

        fig, ax = plt.subplots()
        if use_mean:
            ax.plot(self.spectrum.energy_axis, self.spectrum.mean().data,
                    color='black', label='Data')
            ax.plot(self.spectrum.energy_axis,
                    self.multi_model_signal.mean().data, color='red',
                    label='Background model')

        else:
            ax.plot(self.spectrum.energy_axis, self.spectrum.data,
                    color='black', label='Data')
            ax.plot(self.spectrum.energy_axis, self.model.data,
                    color='red', label='Background model')

        ax.fill_between(self.spectrum.energy_axis, 0,
                        1.1*self.spectrum.data.max(),
                        where=np.invert(self.spectrum.exclude), color='green',
                        alpha=0.5, label='Background region')
        ax.set_xlabel(r'Energy Loss [eV]')
        ax.set_ylabel(r'Counts')
        if integration_range is not None:
            boolean = (integration_range[0] <= self.spectrum.energy_axis) \
                      & (integration_range[1] >= self.spectrum.energy_axis)
            ax.fill_between(self.spectrum.energy_axis, self.model.data,
                            self.spectrum.data, where=boolean,
                            color='blue', alpha=0.5,
                            label='Integration region')
        ax.legend()
        ax.set_ylim([0, 1.1*self.spectrum.data.max()])
        if isinstance(self.spectrum, MultiSpectrum):
            ax.set_title(self.spectrum.currentspectrumid)

        self.spectrum.exclude = prev_exclude  # make the exclude is reset
        return fig

    def quantify_from_edge(self, integration_range, element, edge, E0, alpha,
                           beta, ll=None):
        """
        Calculates the abundance of the element using the background removed
        edge. The integration window should be added together with information
        on which element, and the experimental parameters. A low loss could
        be provided to get a more accurate result.


        Parameters
        ----------
        integration_range: tuple
            Input the expected integration range to get the elemental
            abundance. Just used for visualization purpose.
        element: string
            The element of interest
        edge: string
            The edge being used
        E0: float
            acceleration voltage [V]
        alpha: float
            convergence angle [rad]
        beta: float
            collection angle [rad]
        ll: Spectrum or MultiSpectrum
            The low loss of the used core-loss spectrum which is used to
            convolve the theoretical core-loss edges and have a more accurate
            result.

        Returns
        -------
        abundance: float, numpy 2d array, None
            The calculated abundance using the background removal method.
            A float is returned for a spectrum and a 2d numpy array when a
            multispectrum is used. None is returned when this function is
            called before any background removal has been applied.

        """

        specshape = self.spectrum.get_spectrumshape()
        comp_elements = []
        comp_elements.append(ZezhongCoreLossEdgeCombined(specshape, 1, E0,
                                                         alpha, beta, element,
                                                         edge))

        if isinstance(ll, MultiSpectrum):
            shape = (self.spectrum.xsize, self.spectrum.ysize)
            calc_int = np.zeros(shape)

            for index in tqdm(np.ndindex(shape)):
                islice = np.s_[index]
                ll0 = ll[islice[0], islice[1], :]
                llcomp = MscatterFFT(specshape, ll0)
                mod = Model(specshape, components=comp_elements+[llcomp])
                mod.calculate()
                calc_int[islice] = mod.integrate(integration_range)

        else:
            if ll is not None:
                comp_elements.append(MscatterFFT(specshape, ll))

            mod = Model(specshape, components=comp_elements)
            mod.calculate()

            if isinstance(self.spectrum, MultiSpectrum):
                shape = (self.spectrum.xsize, self.spectrum.ysize)
                calc_int = np.zeros(shape)
                calc_int[:, :] = mod.integrate(integration_range)
            else:
                calc_int = mod.integrate(integration_range)

            if self.result is None:
                print('uses the fast calculated background fit')
                return None

        exp_int = self.result.integrate(integration_range)
        abundance = exp_int / calc_int
        return abundance
