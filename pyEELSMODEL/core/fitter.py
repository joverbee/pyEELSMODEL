# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import logging
import time
from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.core.spectrum import Spectrum
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


class Fitter:
    """
    Fitter class from which all actual fitters can be derived.

    many things are in common for all fitters like:
        -calculate goodness of fit (actual implementation can be overridden eg
        difference between weighted least square or lsq or poisson specific)
        -do iterations until certain goodness of fit is reached
        -calculate degrees of freedom
        -calculate crlb
        -
    """

    def __init__(self, spectrum, model):
        """
        Initialises a Fitter instance.

        Parameters
        ----------
        spectrum : Spectrum or Multispectrum
            The experimental data used which needs to be fitted
        model : Model
            The model used to fit the experimental data.

        Returns
        -------
        An instance of a Fitter.

        """
        self.spectrum = spectrum
        self.model = model
        self.fittertype = 'fitter base class'
        self.status = 'status info'
        self.progress = 0
        self.stop = False
        self.set_start_parameters()
        self.set_bounds()

        if self.model.hasconvolutor():
            # no analytical gradient when using a convolution
            self.usegradients = False
        else:
            self.usegradients = True

        self.information_matrix = None
        self.covariance_matrix = None
        self.coeff_matrix = None
        self.coeff = None
        self.error = None
        self.error_matrix = None
        self.LR = None
        self.LR_matrix = None
        self.getcurrentfit = None
        if isinstance(self.spectrum, MultiSpectrum):
            # set the current spectrum to the first one
            self.spectrum.setcurrentspectrum(spectrum.currentspectrumid)
            self.getcurrentfit = spectrum.currentspectrumid

        self.elapsed_time = None  # time it took to do the multifit

    @property
    def fittertype(self):
        return self._fittertype

    @fittertype.setter
    def fittertype(self, fittertype):
        self._fittertype = fittertype

    # def getprogress(self):
    #     # returns an int from 0 to 100 indicating the progress in a multifit
    #     return self.progress

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def usegradients(self):
        return self._usegradients

    @usegradients.setter
    def usegradients(self, b):
        # If convolution is performed, the analytical gradients are not used
        if self.model.hasconvolutor():
            logger.warning(
                'cannot use analytical gradients since a '
                'convolutor is inside model')
            self._usegradients = False

        else:
            self._usegradients = b

        # todo better implementation, will not work since jacobian
        # is not defined in the fitter.

        try:
            self.jacobian = self._jacobian
        except Exception:
            logger.info(r'No jacobian defined for this fitter type')

    @property
    def information_matrix(self):
        return self._information_matrix

    @information_matrix.setter
    def information_matrix(self, information_matrix):
        self._information_matrix = information_matrix

    @property
    def covariance_matrix(self):
        return self._covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(self, covariance_matrix):
        self._covariance_matrix = covariance_matrix

    def getlabelist(self):
        label_list = []
        for param in self.model.getfreeparameters():
            comp = self.model.getcomponentbyparameter(param)
            label_list.append(comp.name + ': ' + param.name)
        return label_list

    def show_covariance_matrix(self):
        """
        Shows the covariance matrix and adds the labels of the different
        components.

        Returns
        -------
        fig: Figure
        """

        self.set_information_matrix()
        label_list = self.getlabelist()

        fig, ax = plt.subplots()
        ax.imshow(np.abs(self.covariance_matrix))
        ax.set_xticks(np.arange(len(label_list)))
        ax.set_yticks(np.arange(len(label_list)))
        ax.set_xticklabels(label_list, rotation=45, ha="right")
        ax.set_yticklabels(label_list, rotation=45)
        fig.set_tight_layout(True)
        return fig

    def show_variance(self):
        """
        Shows the variance and adds the labels of the different components.

        Returns
        -------
        fig: Figure
        """
        label_list = self.getlabelist()
        print(label_list)
        fig, ax = plt.subplots()
        ax.plot(self.covariance_matrix.diagonal())
        ax.set_xticks(np.arange(len(label_list)))
        ax.set_xticklabels(label_list, rotation=45, ha="right")
        fig.set_tight_layout(True)
        return fig

    def show_coefficients(self, index=(0, 0)):
        if isinstance(self.spectrum, MultiSpectrum):
            self.setcurrentfit(index=index)

        label_list = self.getlabelist()

        fig, ax = plt.subplots()
        x = np.arange(self.coeff.size)
        width = x[1] - x[0]
        ax.bar(x, self.coeff, width=width)
        ax.set_xticks(np.arange(len(label_list)))
        ax.set_xticklabels(label_list, rotation=45, ha="right")
        fig.set_tight_layout(True)

    def getdoresidual(self):
        return self.residual

    def setminstep(self, x):
        self.minstep = x

    def setmaxstep(self, x):
        self.maxstep = x

    def setfraction(self, x):
        self.fraction = x

    def setusegradients(self, b):
        self.usegradients = b

    def setdoresidual(self, b):
        self.residual = b

    def settolerance(self, t):
        self.tolerance = t

    def setnmax(self, n):
        self.nmax = n

    def gettolerance(self):
        return self.tolerance

    def createmodelinfo(self):

        return 0

    def CRLB(self, par, index=None):
        """
        Returns the CRLB bound of the parameter par. If the spectrum is a
        multispectrum, the results for the different spectra can be modified by
        inputting another index.

        Parameters
        ----------
        par: Parameter
            A parameter which is used in the model to fit the data
        index: tuple
            The index of the spectrum from which we want the CRLB. Only works
            when the given spectrum is a multispectrum
            (default: use the current index).

        Returns
        -------
        CRLB: float
            The CRLB for the given parameter and spectrum.

        """
        if isinstance(self.spectrum, MultiSpectrum):
            if index is not None:
                self.setcurrentfit(index=index)

        ii = self.get_param_index(par)

        if ii is None:
            print("This parameter is not fitted, check if this is not free or"
                  " non existing in model")
            return None

        if self.covariance_matrix is None:
            self.set_information_matrix()

        CRLB = np.sqrt(self.covariance_matrix[ii, ii])
        return CRLB

    def CRLB_map(self, par):
        """
        Returns the CRLB bound of the parameter par. For each scanned point.
        This only works when a multispectrum is fitted.

        Parameters
        ----------
        par: Parameter
            A parameter which is used in the model to fit the data

        Returns
        -------
        crlb_map: numpy array (2D)
            The CRLB map the given parameter.

        """

        if not isinstance(self.spectrum, MultiSpectrum):
            raise TypeError(r'Mapping not possible since given spectrum is no'
                            r' multispectrum')

        shape = (self.spectrum.xsize, self.spectrum.ysize)
        crlb_map = np.zeros(shape)
        for index in tqdm(np.ndindex(shape),total=np.prod(shape),leave=True,position=0):
            islice = np.s_[index]
            ii = (islice[0], islice[1])
            crlb = self.CRLB(par, index=ii)
            crlb_map[islice] = crlb

        return crlb_map

    def CRLB_ratio(self, par1, par2, index=(0, 0)):
        """
        Returns the CRLB bound of the ratio between par1 and par2 (par1/par2).
        The results for the different spectra can be modified by inputting
        another index.

        Parameters
        ----------
        par1: Parameter
            A parameter which is used in the model to fit the data
        par2: Parameter
            A parameter which is used in the model to fit the data
        index: tuple
            The index of the spectrum from which we want the CRLB. Only works
            when the given spectrum is a multispectrum (default: (0,0)).

        Returns
        -------
        error: float
            The CRLB bound calculated for the ratio between two parameters.

        """

        if isinstance(self.spectrum, MultiSpectrum):
            self.setcurrentfit(self, index=index)

        ratio = par1.getvalue() / par2.getvalue()
        index1 = self.get_param_index(par1)
        index2 = self.get_param_index(par2)
        if index1 is None or index2 is None:
            print("This parameter is not fitted, check if this is not free or "
                  "non existing in model")
            return None

        if self.covariance_matrix is None:
            self.set_information_matrix()

        rel_er1 = self.covariance_matrix[index1, index1] / par1.getvalue() ** 2
        rel_er2 = self.covariance_matrix[index2, index2] / par2.getvalue() ** 2
        cov_er = 2 * self.covariance_matrix[index1, index2] / \
            (par1.getvalue() * par2.getvalue())
        error = ratio * np.sqrt(rel_er1 + rel_er2 - cov_er)
        return error

    def get_param_index(self, param):
        """
        Returns the index of the parameter in the free parameter list. This
        function is used to ask the fitted value for which parameter. If the
        given is not in the list it returns None

        Parameters
        ----------
        param: Parameter
            The parameter from which we want to get the index

        Returns
        -------
        index: int or None
            The index which corresponds to the given parameter in the fit.

        """
        for index, parameter in enumerate(self.model.getfreeparameters()):
            if parameter is param:
                return index
        return None

    def degreesoffreedom(self):
        """
        Returns the number of degrees of freedom in the fitter. This depends
        on the number of free parameters in the model and the number of fitted
        points in the spectrum
        """
        dof = self.spectrum.getnonexcludedpoints() \
            - self.model.getnumfreeparameters()
        return dof

    def LRtestconfidence(self, index=(0, 0)):
        """
        Confidence test using the likelyhood ratio of Poisson noise. See
        publication EELSmodel for explanation on this

        """
        if self.LR is None:
            logger.info('LR needs to be calculated')
            self.likelihood_ratio(index=index)

        dgf = self.degreesoffreedom()
        confidence = stats.chi2.cdf(self.LR, dgf)
        return confidence

    def multi_LRtestconficence(self):
        shape = (self.spectrum.xsize, self.spectrum.ysize)

        self.confidence_map = np.zeros(shape)

        for index in tqdm(np.ndindex(shape),total=np.prod(shape),leave=True,position=0):
            islice = np.s_[index]
            self.confidence_map[islice] = self.LRtestconfidence(index)

    def LRtestconfidence_string(self):
        confidence = 100 * self.LRtestconfidence()
        print('The model is accepted at a confidence levet of: {conf} percent'
              .format(conf=confidence))

    def getstatus(self):
        return 'status string'

    def dolintrick(self, b):
        # wether to do a special lin/nonlin trick for the linear parameters
        self.dolintrick = b

    def getdolintrick(self):
        return self.dolintrick

    def candolintrick(self):
        return self.candolin

    def _setcandolin(self, b):
        self.candolin = b

    def set_information_matrix(self):
        """
        Calculates the fischer information matrix followed by formula (16)
        in https://doi.org/10.1016/j.ultramic.2004.06.004
        Used for Poisson noise

        """
        self.model.calculate()  # calculate the newest model

        # do not fully understand why **(-1) but it seems to fit
        ndata = self.spectrum.pppc ** (-1) * np.copy(self.model.data)
        # ndata[ndata < 1.] = 1. #small values will blow up lambda_m

        lambda_m = 1 / np.sqrt(np.tile(ndata[np.invert(self.spectrum.exclude)],
                                       (self.model.getnumfreeparameters(), 1)))
        deriv_matrix = lambda_m * self.calculate_derivmatrix()
        self.information_matrix = np.dot(deriv_matrix,
                                         np.transpose(deriv_matrix))

        try:
            self.covariance_matrix = np.linalg.inv(self.information_matrix)
        except Exception:
            # todo solve this issue since it can happen that two rows are zero
            print(
                'Information matrix is Singular, can happen if two components'
                ' are exactly zero')

    def calculate_derivmatrix(self):
        """
        Calculates the derivative matrix where each row is a derivative of the
        model spectrum to a free parameter. The number of columns depends on
        the size of the spectrum.

        Returns
        -------
        deriv_matrix: numpy array (2D)
            The derivative matrix for each fitted parameter.
        """
        deriv_matrix = np.zeros((self.model.getnumfreeparameters(),
                                 self.spectrum.getnonexcludedpoints()))
        for index, param in enumerate(self.model.getfreeparameters()):
            deriv_matrix[index] = self.partial_derivative(param)
        return deriv_matrix

    def partial_derivative(self, parameter, fraction=0.001):
        """
        Calculates the partial derivative of the parameter. It uses the
        use_gradient attribute to check if the gradient is calculated
        analytically or numerically.

        Parameters
        ----------
        parameter: Parameter
            The parameter from which the partial derivative will be
            calculated.
        fraction: float
            The fraction of the step the parameter does to calculate
            the numerical gradient. (default: 0.001)

        Returns
        -------
        parial: numpy array (1D)
            The partial derivative for the given parameter.
        """

        if not self.usegradients:
            partial = self.numerical_partial_derivative(parameter,
                                                        fraction=fraction)
        else:
            component = self.model.getcomponentbyparameter(parameter)
            partial = component.getgradient(parameter)
            if partial is None:
                logging.info(r'No analytical gradient is implemented for'
                             r' this parameter. It will use a numerical '
                             r'calculation to get the result')
                partial = self.numerical_partial_derivative(parameter,
                                                            fraction=fraction)

        if partial is None:
            logging.warning(r'No partial derivative can be determined')
            return None
        else:
            return partial[np.invert(self.spectrum.exclude)]

    def numerical_partial_derivative(self, parameter, fraction=0.001):
        """
        Calculates the numerical partial derivative of the parameter.

        Parameters
        ----------
        parameter: Parameter
            The parameter from which the numerical derivative will be
            calculated.
        fraction: float
            The fraction of the step the parameter does to calculate
            the numerical gradient (default: 0.001)

        Returns
        -------
        """

        minstep = 1e-99
        maxstep = 1e99
        component = self.model.getcomponentbyparameter(parameter)
        if component is None:
            return None
        if component._ismultiplier or component._isconvolutor \
                or component._isshifter:
            return None
        delt = np.abs(parameter.getvalue() * fraction)
        delt = max(delt, minstep)
        delt = min(delt, maxstep)

        orig_par = parameter.getvalue()
        self.model.calculate()
        orig_mod = self.model.data[:]

        parameter.setvalue(orig_par + delt)
        self.model.calculate()
        up_mod = self.model.data[:]
        parameter.setvalue(orig_par)
        partial = (up_mod - orig_mod) / delt
        return partial

    def set_start_parameters(self):
        """
        Gets the start parameters of the model as initial parameters for the
        fitting. This function is used when starting values need to be
        added to the fit.

        Returns
        -------
        start_param: List
            A list containing the starting values for the fitting

        """
        start_param = []
        for param in self.model.getfreeparameters():
            start_param.append(param.getvalue())
        self._start_param = start_param

    def get_start_parameters(self):
        return self._start_param

    def set_fit_values(self):
        """
        Function which set the fitting values to the model.
        """
        if self.coeff is None:
            logger.info('No fit has been performed hence the values cannot '
                        'be set.')
            return None

        for ii, param in enumerate(self.model.getfreeparameters()):
            # if setvalue does not work, then it returns False
            if not param.setvalue(self.coeff[ii]):
                logger.info(param.name + r' value cannot be set')
                param.setvalue(0)
        self.model.calculate()

    def multi_fit(self, start_param=None):
        """
        Performs the selected fit over each spectrum in the multispectrum.

        Parameters
        ----------
        start_param: list
            List of starting parameters for the fit. If None, the values
            of the model are used. The shape of the start_param should be
            of size (xsize, ysize, numberoffreeparamters). The starting
            parameters are only needed for the non-linear models.

        Returns
        -------

        """

        start = time.time()

        # iterate over the multispectrum
        if not isinstance(self.spectrum, MultiSpectrum):
            raise ValueError(r'Spectrum should be a multispectrum')

        # the convolutor used in the model
        conv = self.model.getconvolutor()

        shape = (self.spectrum.xsize, self.spectrum.ysize)
        self.progress = 0
        self.stop = False
        initialindex = self.spectrum.currentspectrumid

        step = 100 / (self.spectrum.xsize * self.spectrum.ysize)
        for index in tqdm(np.ndindex(shape),total=np.prod(shape),leave=True,position=0):
            islice = np.s_[index]
            self.spectrum.setcurrentspectrum(index)

            # if the low loss spectrum is a multispectrum, the index needs to
            # be changed
            if conv is not None:
                if isinstance(conv.llspectrum, MultiSpectrum):
                    conv.llspectrum.setcurrentspectrum(index)

            # need to be sure that the order of the free parameters is the same
            # as the input start_param
            if start_param is not None:
                self._start_param = start_param[islice]

            # small functionality which should give the ability to use "nearby"
            # solution to have a better initial guess
            # not well tested since mainly linear fitters are used and this
            # does not need any starting parameters
            # if use_prev:
            #     self.set_start_parameters()

            if self.stop:
                # restore to state that we started from
                self.spectrum.setcurrentspectrum(initialindex)
                if conv is not None:
                    if isinstance(conv.llspectrum, MultiSpectrum):
                        conv.llspectrum.setcurrentspectrum(initialindex)
                return

            self.perform_fit()

            # initialized the result matrix at first iteration of the fit.
            if (islice[0] == 0) and (islice[1] == 0):
                self.coeff_matrix = np.zeros((shape[0], shape[1],
                                              self.coeff.size))
                self.error_matrix = np.zeros((shape[0], shape[1]))

            self.coeff_matrix[islice] = self.coeff
            self.error_matrix[islice] = self.error

            self.progress = self.progress + step

        stop = time.time()
        self.elapsed_time = stop - start  # calculates the calculation time

    def set_bounds(self):
        """
        Sets the boundaries for the fitting. It uses the boundaries from
        each free parameter in the model to get these values.
        """
        bounds = []
        for param in self.model.getfreeparameters():
            bounds.append((param.getlowerbound(), param.getupperbound()))
        self._bounds = bounds

    def get_bounds(self):
        return self._bounds

    def model_to_multispectrum(self):
        """
        After the fitting of the multispectrum with the given model, the
        resulting model at every probe position is calculated which will be a
        multispectrum.

        Returns
        -------
        sig: MultiSpectrum
            The fitted multispectrum at each probe position.
        """
        conv = self.model.getconvolutor()

        sig = self.spectrum.copy()
        m = self.model
        shape = (sig.xsize, sig.ysize)
        for index in tqdm(np.ndindex(shape),total=np.prod(shape),leave=True,position=0):
            islice = np.s_[index]

            if conv is not None:
                if isinstance(conv.llspectrum, MultiSpectrum):
                    conv.llspectrum.setcurrentspectrum(index)

            for ii, param in enumerate(m.getfreeparameters()):
                param.setvalue(self.coeff_matrix[islice][ii])
            m.calculate()
            sig.multidata[islice] = m.data

        return sig

    def model_to_multispectrum_without_comps(self, comps):
        """
        Set the parameter values to zero for the components in comps and
        calculates the fitted model. This can be used to remove the background
        from the fitted result or to only have the background.

        Parameters
        ----------
        comps: list of Components
            List of components which should not be added to the calcualted
            model.

        Returns
        -------
        sig: MultiSpectrum
            The fitted multispectrum at each probe position without the given
            components in comp.
        """
        #suppress the components you don't want
        for comp in comps:
            comp.setsuppress(True)
        self.model.setchanged(True)
        sig=self.model_to_multispectrum()
        
        #release the suppress to not confuse things
        for comp in comps:
            comp.setsuppress(False)

        return sig
    
    def model_to_multispectrum_with_comps(self, comps):
        """
        Only uses the components in comps to calculate the fitted model.
        Similar to model_to_multispectrum_without_comps but the exact opposite
        Note that this only works 1st level components and not with subcomponents owned by these 1st level components

        Parameters
        ----------
        comps: list of Components
            List of components are only used in the calculation of the fit.

        Returns
        -------
        sig: MultiSpectrum
            The fitted multispectrum at each probe position only have the given
             components in comps.
        """
        ncomps = []
        for comp in self.model.components:
            if comp in comps:
                pass
            else:
                ncomps.append(comp)

        s = self.model_to_multispectrum_without_comps(ncomps)
        return s

    def get_experimental_edge(self, component, percentile, plotting=False,
                              other_spectra=[]):
        """
        Calculate the proposed edge when using a multispectrum. The percentile
        takes the spectra having the highest content of these edges.


        :return:
        """
        if not isinstance(self.spectrum, MultiSpectrum):
            raise ValueError(r'Inputted spectrum should be a multispectrum')

        if self.coeff_matrix is None:
            raise ValueError(r'Fit is not yet performed, this should be done '
                             r'before getting'
                             r'proposed experimental edge')

        sig_w_comp = self.model_to_multispectrum_without_comps([component])

        index = self.get_param_index(component.parameters[0])
        print(index)
        int_map = self.coeff_matrix[:, :, index]
        co = int(percentile * int_map.size)
        thres = np.sort(int_map.flatten())[co]
        boolean = int_map > thres

        if plotting:
            plt.figure()
            plt.imshow(boolean, cmap='gray')
            plt.title('The mask used to determine the average ')

        avg_ = self.spectrum.multidata[boolean, :]

        res = np.sum(avg_ - sig_w_comp.multidata[boolean, :], axis=0)
        spec = Spectrum(self.spectrum.get_spectrumshape(), data=res)
        avg = Spectrum(self.spectrum.get_spectrumshape(),
                       data=avg_.sum(axis=0))

        other = []
        for ospec in other_spectra:
            avg_ = ospec.multidata[boolean, :]
            other.append(
                Spectrum(ospec.get_spectrumshape(), data=avg_.sum(axis=0)))

        return spec, avg, other

    def plot(self, index=None, externalplt=None, non_components=[], **kwargs):
        """
        Plots the fitter

        Parameters
        ----------
        externalplt : matplotlib reference
              A reference to an external matplotlib reference, if None we use
              our own matplotlib and create a new figure.

        Returns
        ----------
        fig: Figure
        """
        tempplt = plt
        if isinstance(externalplt, plt.Figure):
            tempplt = externalplt
        elif isinstance(externalplt, plt.Axes):
            tempplt = externalplt
            print('axes as input')
        else:
            # create our own figure
            fig = plt.figure()
            plt.title('Fit')
        # show components if visible
        # self.model.calculate()
        if isinstance(self.spectrum, MultiSpectrum) and index is not None:
            self.setcurrentfit(index=index)

        conv = self.model.getconvolutor()
        if conv is not None:
            if isinstance(conv.llspectrum, MultiSpectrum):
                conv.llspectrum.setcurrentspectrum(index)

        self.set_fit_values()
        tempplt.plot(self.spectrum.energy_axis, self.spectrum.data,
                     color='black', label='Experiment', linestyle='dotted')
        for comp in self.model.components:
            if not (comp in non_components):
                tempplt.plot(self.model.energy_axis, comp.data,
                             label=comp.name,
                             **kwargs)

            # print(comp.name)
            # print(np.sum(comp.data))
            # print('- - - - - ')

        tempplt.plot(self.model.energy_axis, self.model.data, label='Model',
                     **kwargs)  # and the total spectrum

        tempplt.fill_between(self.spectrum.energy_axis, 0,
                             self.spectrum.data.max(),
                             where=self.spectrum.exclude, color='green',
                             alpha=0.5)

        if isinstance(externalplt, plt.Axes):
            tempplt.set_xlabel(r'Energy Loss [eV]')
            tempplt.set_ylabel('Counts')
        else:
            tempplt.xlabel(r'Energy Loss [eV]')
            tempplt.ylabel('Counts')
        tempplt.legend()
        if isinstance(self.spectrum, MultiSpectrum):
            tempplt.title(self.spectrum.currentspectrumid)

        if externalplt is None:
            return fig
        else:
            return None

    def likelihood_ratio(self, index=(0, 0)):
        """
        Defined likelihood ratio from: 10.1016/j.ultramic.2004.06.004
        The index will be used when using a multispectrum.

        Parameters
        ----------
        index : tuple
              A tuple showing the index of the used spectrum for the likelihood
              ratio calculation.

        Returns
        ----------
        None

        """
        if isinstance(self.spectrum, MultiSpectrum):
            self.setcurrentfit(index=index)

        data = self.spectrum.data[np.invert(self.spectrum.exclude)].copy()
        data[data <= 0] = 1
        model = self.model.data[np.invert(self.spectrum.exclude)]

        LR = 2 * np.sum(-data + data * np.log(data / model) + model)
        self.LR = LR

    def multi_likelihood_ratio(self):
        shape = (self.spectrum.xsize, self.spectrum.ysize)

        self.LR_map = np.zeros(shape)

        for index in tqdm(np.ndindex(shape),total=np.prod(shape),leave=True,position=0):
            islice = np.s_[index]
            self.likelihood_ratio(index)
            self.LR_map[islice] = self.LR

    def setcurrentfit(self, index=(0, 0)):
        """
        Set the fit to the used index. It will recalculate the model and
        information matrix for the used index. If the index has not changed, it
        will not perform any calculation to reduce the calculation time.

        Parameters
        ----------
        index : tuple
              A tuple showing the index of the used spectrum.

        Returns
        ----------
        None

        """
        if not isinstance(self.spectrum, MultiSpectrum):
            print('Spectrum is no MultiSpectrum hence this method has no use')
            return None

        if index == self.getcurrentfit and self.information_matrix is not None:
            pass
        else:
            self.getcurrentfit = index
            self.spectrum.setcurrentspectrum(index)
            self.coeff = self.coeff_matrix[index[0], index[1]]
            self.set_fit_values()
            self.set_information_matrix()

    def show_map_result(self, comp_elements):
        """
        Shows the maps for the given components in the list. Note that it only
        uses the first free parameter of the component. Hence this is mainly
        used for visualization of the elements maps where the components of
        the coreloss edges have only one parameter which needs to be fitted.


        Parameters
        ----------
        comp_elements : list
              List containing the components used in the model which have been
              fitted.

        Returns
        ----------
        fig: matplotlib figure
            The figure which is created. This way you can save it or modify it
            if needed.
        maps: numpy array
            Contains all the different maps where the indices are the same as
            the list of components.
        names: list
            List containing the name for each map. These returns are mainly
            used for modifying the images if more advanced figures need to be
            created for publications or reports.

        """
        maps = np.zeros((len(comp_elements), self.spectrum.xsize,
                         self.spectrum.ysize))
        names = np.zeros(len(comp_elements), dtype='object')

        # todo ugly way of discriminating between 1 or multiple components
        if len(comp_elements) == 1:
            fig, ax = plt.subplots(1, len(comp_elements), figsize=(8, 3))
            for i in range(len(comp_elements)):
                index = self.get_param_index(comp_elements[i].parameters[0])
                ax.imshow(self.coeff_matrix[:, :, index], cmap='inferno')
                ax.set_title(comp_elements[i].name)
                ax.set_xticks([])
                ax.set_yticks([])
                maps[i] = self.coeff_matrix[:, :, index]
                names[i] = comp_elements[i].name

        else:
            fig, ax = plt.subplots(1, len(comp_elements), figsize=(8, 3))
            for i in range(len(comp_elements)):
                index = self.get_param_index(comp_elements[i].parameters[0])
                ax[i].imshow(self.coeff_matrix[:, :, index], cmap='inferno')
                ax[i].set_title(comp_elements[i].name)
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                maps[i] = self.coeff_matrix[:, :, index]
                names[i] = comp_elements[i].name

        return fig, maps, names

    def get_map_results(self, comp_elements):
        """
        Calculates the maps of the elements.

        Parameters
        ----------
        comp_elements : list
              List containing the components used in the model which have been
              fitted.

        Returns
        ----------
        maps: numpy array
            Contains all the different maps where the indices are the same as
            the list of components.
        names: list
            List containing the name for each map. These returns are mainly
            used for modifying the images if more advanced figures need to be
            created for publications or reports.
        """
        maps = np.zeros((len(comp_elements), self.spectrum.xsize,
                         self.spectrum.ysize))
        names = np.zeros(len(comp_elements), dtype='object')
        for i in range(len(comp_elements)):
            index = self.get_param_index(comp_elements[i].parameters[0])
            maps[i] = self.coeff_matrix[:, :, index]
            names[i] = comp_elements[i].name
        return maps, names

    def get_jump_ratio_map(self, component, onset_energy, interval=10):
        """
        Returns a map with the jump ratio between the background and edge
        calculated by integrating over interval region.
        This is an experimental function which needs to be more tested and
        explored.
        """

        sig_bkg = self.model_to_multispectrum_without_comps([component])
        sig_comp = self.model_to_multispectrum_with_comps([component])

        index = sig_bkg.get_energy_index(onset_energy)

        bkg_int = sig_bkg.multidata[:, :, index:index + interval].sum(2)
        comp_int = sig_comp.multidata[:, :, index:index + interval].sum(2)

        jump_ratio = comp_int / bkg_int

        return jump_ratio, comp_int, bkg_int, sig_bkg, sig_comp

    def get_map(self, param):
        """
        Returns the fitted coefficients of the parameter param. This only
        work if a multispectrum is used to fit.

        Parameters
        ----------
        param : Parameter
              A parameter which is optimized and from which the resulting
              map is requested.

        Returns
        ----------
        param_map: 2d numpy array
            The coefficient map of the given parameters

        """
        index = self.get_param_index(param)
        if index is None:
            print('Parameter is not optimized during the fit')
            return None
        param_map = self.coeff_matrix[:, :, index]
        return param_map
