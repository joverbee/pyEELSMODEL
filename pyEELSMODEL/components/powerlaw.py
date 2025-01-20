"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PowerLaw(Component):
    """
    A power law background model

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model

    A : float
        Amplitude of the powerlaw. Defined such that the offset
        energy has the value of the amplitude A.

    r: float
        The exponent value of the powerlaw.

    Returns
    -------
    """

    def __init__(self, specshape, A, r):
        super().__init__(specshape)
        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(0, 1e20)
        p1.sethasgradient(True)
        self._addparameter(p1)

        p2 = Parameter('r', r)
        p2.setlinear(False)
        p2.setboundaries(0.0, 1e20)
        p2.sethasgradient(True)
        self._addparameter(p2)
        # don't convolute the background it only gives problems and
        # adds no extra physics
        self._setcanconvolute(False)
        self.Estart = specshape.offset

        self._setname('Powerlaw')

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        if p1.ischanged() or p2.ischanged():
            A = p1.getvalue()
            r = p2.getvalue()
            self.data = self.powerlaw(A, r)
        self.setunchanged()  # put parameters to unchanged

    def powerlaw(self, A, r):
        return A * (self.energy_axis / self.Estart) ** (-r)

    def getgradient(self, parameter):
        """calculate the analytical partial derivative wrt parameter
        returns a reference to the gradient
        """
        if not parameter.hasgradient:
            return None
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        A = p1.getvalue()
        r = p2.getvalue()
        E = self.energy_axis
        mask = (E > 0) * (r > 0)  # only meaningfull when E>0 and r>0
        if parameter == p1:
            self.gradient[0] = mask * self.powerlaw(1, r)
            return self.gradient[0]
        elif parameter == p2:
            self.gradient[1] = mask * self.powerlaw(A, r) * np.log(
                self.Estart / E)

            return self.gradient[1]
        else:
            # throw Componenterr::bad_index()
            return None

    def autofit_(self, spectrum, istart, istop):
        """
        Performs a first guess of the parameters from the Powerlaw fit.
        It checks if the obtained values are reasonable and will apply
        them to the components parameters. The procedure is explained
        in Electron Energy Loss Spectroscopy (Third Edition) of Egerton
        in ...

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum data from which an estimate is done on the background
        istart: int
            Index from where to start
        istop: int
            Index indicating the end of the fitting region

        Returns
        -------
        None.

        """

        # powerlaw fit to autodetermine the A and r parameter.
        if istart >= istop:
            # impossible to fit
            A = 0.0
            r = 3.0
            return
        N = istop - istart
        E = self.energy_axis

        # added to remove pixel values smaller than 1 since they give a
        # problem with the log
        ndat = np.copy(spectrum.data[istart:istop])
        # ndat[ndat<1.] = 1
        boolean = ndat >= 1

        x = np.log(E[istart:istop][boolean])
        N = x.size
        #

        y = np.log(ndat[boolean])
        xy = (x * y).sum()
        sx = x.sum()
        sy = y.sum()
        sxsq = (x ** 2).sum()

        b = (N * xy - sx * sy) / (N * sxsq - sx ** 2.0)
        a = sy / N - b * sx / N
        r = -b
        A = np.exp(a) * E[0] ** -r  # correct for referencing A to E[0]

        # print(A)
        # print(r)

        # todo modified the checks for the fast background measurement
        # sanity check
        if (r < 0.0) or (r > 10):
            r = 3.0
            A = spectrum.data[0]
            logger.warning('Powerlaw autofit resulted in r<1, ignoring output')
        if (A < 0.0):
            r = 3.0
            A = spectrum.data[0]
            logger.warning('Powerlaw autofit resulted in A<0, ignoring output')

        # logger.info('PowerLaw autofit resulted in A=%e, r=%e',A,r)
        # set the parameters
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p1.setvalue(A)
        p2.setvalue(r)
        self.calculate()

    def autofit(self, spectrum, istart, istop):
        """
        Performs a first guess of the parameters from the Powerlaw fit.
        It checks if the obtained values are resonable and will apply
        them to the components parameters. The procedure is explained
        in Electron Energy Loss Spectroscopy (Third Edition) of Egerton
        in ...

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum data from which an estimate is done on the background
        istart: int
            Index from where to start
        istop: int
            Index indicating the end of the fitting region

        Returns
        -------
        None.

        """

        # powerlaw fit to autodetermine the A and r parameter.
        if istart >= istop:
            # impossible to fit
            A = 0.0
            r = 3.0
            return
        N = istop - istart
        E = self.energy_axis

        # added to remove pixel values smaller than 1 since they give a
        # problem with the log
        ndat = np.copy(spectrum.data[istart:istop])
        # ndat[ndat<1.] = 1

        boolean = ndat > 1.
        x = np.log(E[istart:istop][boolean])
        N = x.size

        y = np.log(ndat[boolean])
        sx = x.sum()

        D = np.nansum(x ** 2 / y) * np.nansum(1 / y) - np.nansum(x / y) ** 2
        r = -1 * (sx * np.nansum(1 / y) - N * np.nansum(x / y)) / D
        logA = (N * np.nansum(x ** 2 / y) - sx * np.nansum(x / y)) / D
        A = np.exp(logA) * E[0] ** -r

        # print('A: ' + str(A))
        # print('r: ' + str(r))

        # todo modified the checks for the fast background measurement
        # sanity check
        # if (r < 0.0) or (r > 10):
        #     r = 3.0
        #     A = spectrum.data[0]
        #     logger.warning('Powerlaw autofit resulted in r<1, 3
        #     ignoring output')
        # if (A < 0.0):
        #     r = 3.0
        #     A = spectrum.data[0]
        #     logger.warning('Powerlaw autofit resulted in A<0,
        #     ignoring output')

        # logger.info('PowerLaw autofit resulted in A=%e, r=%e',A,r)
        # set the parameters
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        b1 = p1.setvalue(A)
        b2 = p2.setvalue(r)
        self.calculate()
        return b1 and b2

    def autofit1(self, spectrum, istart, istop):
        """
        Performs a first guess of the parameters from the Powerlaw fit.
        It checks if the obtained values are resonable and will apply
        them to the components parameters. The procedure is explained
        in Electron Energy Loss Spectroscopy (Third Edition) of Egerton
        in ...

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum data from which an estimate is done on the background
        istart: int
            Index from where to start
        istop: int
            Index indicating the end of the fitting region

        Returns
        -------
        None.

        """

        # powerlaw fit to autodetermine the A and r parameter.
        if istart >= istop:
            # impossible to fit
            A = 0.0
            r = 3.0
            return
        E = self.energy_axis

        x = np.log(E[istart:istop])

        ndat = np.copy(spectrum.data[istart:istop])
        ndat[ndat < 1.] = 1

        # w = 1/np.sqrt(ndat)
        # todo have a proper description of the weights for linear fitter
        # w = 1/ndat
        w = np.ones(ndat.size)

        y = np.log(ndat)
        xw = (w * x).sum() / w.sum()
        yw = (w * y).sum() / w.sum()
        b = np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw) ** 2)
        a = yw - b * xw
        r = -b
        A = np.exp(a) * E[0] ** -r  # correct for referencing A to E[0]

        # todo modified the checks for the fast background measurement
        # sanity check
        # if (r < 0.0) or (r > 10):
        #     r = 3.0
        #     A = spectrum.data[0]
        #     logger.warning('Powerlaw autofit resulted in r<1,
        #     ignoring output')
        # if (A < 0.0):
        #     r = 3.0
        #     A = spectrum.data[0]
        #     logger.warning('Powerlaw autofit resulted in A<0,
        #     ignoring output')

        # logger.info('PowerLaw autofit resulted in A=%e, r=%e',A,r)
        # set the parameters
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p1.setvalue(A)
        p2.setvalue(r)
        self.calculate()

    def autofit_matrix(self, spectrum, istart, istop):
        """
        Weighted linear least squares where the variance is given by
         the square root of the
        signal

        :param spectrum:
        :param istart:
        :param istop:
        :return:
        """
        if istart > istop:
            return

        subdata = spectrum.data[istart:istop]
        boolean = subdata >= 1.
        ln_y = np.log(subdata[boolean])
        W = np.diag(np.log(1 / (subdata[boolean])))
        A = np.zeros((boolean.sum(), 2))
        A[:, 0] = np.ones(boolean.sum())
        A[:, 1] = -np.log(
            self.energy_axis[istart:istop][boolean] / self.energy_axis[0])

        term1 = np.linalg.inv(np.dot(A.T, np.dot(W, A)))
        term2 = np.dot(A.T, np.dot(W, ln_y))
        x_ = np.dot(term1, term2)

        # x_, resi, rank, sing = np.linalg.lstsq(A, ln_y, rcond=-1)
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p1.setvalue(np.exp(x_[0]))
        p2.setvalue(x_[1])
        self.calculate()

    def autofit_twowindow(self, spectrum, istart, istop):
        """
        Performs a first guess of the parameters from the Powerlaw fit.
        It checks if the obtained values are resonable and will apply
        them to the components parameters. The procedure is explained
        in Electron Energy Loss Spectroscopy (Third Edition) of Egerton
        in ...

        This is the two-window technique. It is quick and dirty,
        and not as accurate as a proper least squares fit,
        but is has its uses.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum data from which an estimate is done on the background
        istart: int
            Index from where to start
        istop: int
            Index indicating the end of the fitting region

        Returns
        -------
        None.

        """

        # powerlaw fit to autodetermine the A and r parameter.
        if istart >= istop:
            # impossible to fit
            A = 0.0
            r = 3.0
            return

        y = np.copy(spectrum.data[istart:istop])  # istop = non-inclusive

        x0 = self.energy_axis[istart]
        x1 = self.energy_axis[istop - 1]  # istop - 1 is inclusive

        I0 = np.sum(y[:len(y) // 2])
        I1 = np.sum(y[len(y) // 2:])
        r = 2. * np.log(I0 / I1) / np.log(
            x1 / x0)  # the usual power-law exponent
        A = (I0 + I1) * (x0 * x1) ** (r / 2.) / (x1 - x0)
        A *= self.Estart ** (-r)

        # todo modified the checks for the fast background measurement
        # sanity check
        if (r < 0.0) or (r > 10):
            r = 3.0
            A = spectrum.data[0]
            logger.warning('Powerlaw autofit resulted in r<1, ignoring output')
        if (A < 0.0):
            r = 3.0
            A = spectrum.data[0]
            logger.warning('Powerlaw autofit resulted in A<0, ignoring output')

        # logger.info('PowerLaw autofit resulted in A=%e, r=%e',A,r)
        # set the parameters
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p1.setvalue(A)
        p2.setvalue(r)
        self.calculate()

    def autofit_areas(self, spectrum, istart, istop):
        """
        Performs a first guess of the parameters from the Powerlaw fit.
        It checks if the obtained values are resonable and will apply
        them to the components parameters. The procedure is explained
        in Electron Energy Loss Spectroscopy (Third Edition) of Egerton
        in Chapter 4.4.2. (ISBN: 978-1-4419-9583-4).

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum data from which an estimate is done on the background
        istart: int
            Index from where to start
        istop: int
            Index indicating the end of the fitting region

        Returns
        -------
        None.

        """
        boolean = (istart + istop) % 2 == 0
        if not boolean:
            istop -= 1

        if istart - istop == 0:
            istop = istart + 2

        if istart >= istop:
            # impossible to fit
            A = 0.0
            r = 3.0
            return

        imiddle = (istop + istart) // 2
        I1 = spectrum.data[istart:imiddle].sum()
        I2 = spectrum.data[imiddle:istop].sum()
        E = spectrum.energy_axis
        r = 2 * np.log(I1 / I2) / (np.log(E[istop] / E[istart]))
        k = 1 - r
        # Since E0 is used there is another normalization
        # in this case it is found experimentally instead of proven
        # (so can be wrong)
        A = (k * I2 / ((E[istop] / E[0]) ** k - (E[imiddle] / E[0]) ** k)) / E[
            0]
        # sanity check
        if (r < 1.0) or (r > 10):
            r = 3.0
            A = spectrum.data[0]
            logger.warning('Powerlaw autofit resulted in r<1, ignoring output')
        if (A < 0.0):
            r = 3.0
            A = spectrum.data[0]
            logger.warning('Powerlaw autofit resulted in A<0, ignoring output')

        # logger.info('PowerLaw autofit resulted in A=%e, r=%e',A,r)
        # set the parameters
        p1 = self.parameters[0]
        p2 = self.parameters[1]
        p1.setvalue(A)
        p2.setvalue(r)
        self.calculate()
