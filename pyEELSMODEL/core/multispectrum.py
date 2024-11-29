"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import h5py
from os.path import exists
import os
from tqdm import tqdm

from pyEELSMODEL.io_tools.dm_ncempy import dmReader
from pyEELSMODEL.io_tools.hdf5_io import load_h5py, load_hspy
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape

import copy
import logging

logger = logging.getLogger(__name__)


class MultiSpectrumshape:
    """
    MultiSpectrumshape is a class holding the main parameters of a spectrum in
    order to compactly create several spectra with the same size by providing
    an instance of this class.
    It holds:
        dispersion: float, dispersion in eV/pixel describing the energy scale
        offset: float, energy of the first pixel in the spectrum in eV
        size: int, the size of a spectrum in number of pixels (eg typ 1024)
        xsize: int number of spectra in x direction
        ysize: int number of spectra in y direction
    """

    def __init__(self, dispersion, offset, Esize, xsize, ysize):
        self.dispersion = dispersion
        self.offset = offset
        self.Esize = Esize
        self.xsize = xsize
        self.ysize = ysize
        self.spectrumshape = Spectrumshape(dispersion, offset, Esize)
        self._index = 0
        logger.debug('Multispectrumshape init')

    def getspectrumshape(self):
        return self.spectrumshape


class MultiSpectrumIterator:
    """MultiSpectrum Iterator class"""

    def __init__(self, multispectrum):
        # Team object reference
        self._multispectrum = multispectrum
        # member variable to keep track of current index
        self._index = 0
        self.size = multispectrum.xsize * multispectrum.ysize

    def __next__(self):
        if self._index < self.size:
            result = self._multispectrum[self._index]
            self._index += 1
            return result
        raise StopIteration


class MultiSpectrum(Spectrum):
    def __init__(self, multispectrumshape, data=None, acq_time=0):
        """
        Initialises a MultiSpectrum instance

        Parameters
        ----------
        spectrumshape : MultiSpectrumshape
            holds the basic parameters dispersion [eV/pixel], offset [eV] and
            size [pixels] to create the energy scale of the spectrum
        data : float or int, optional
            Holds the EELS intensity data with same size as spectrumshape.size.
            If None is given an empty spectrum is created with all zeros. The
            default is None.

        Raises
        ------
        ValueError
            if data does not have same size as in spectrumshape.size
        TypeError
            if data is neither int or float

        Returns
        -------
        An instance of a Spectrum.

        """
        spectrumshape = Spectrumshape(multispectrumshape.dispersion,
                                      multispectrumshape.offset,
                                      multispectrumshape.Esize)

        super().__init__(spectrumshape, acq_time=acq_time)

        self.xsize = multispectrumshape.xsize
        self.ysize = multispectrumshape.ysize
        self.currentspectrumid = (0, 0)

        if data is None:
            # create empy vector of data
            self.multidata = np.zeros((self.xsize, self.ysize, self.size))

        else:
            sh = (self.xsize, self.ysize, multispectrumshape.Esize)
            if data.shape != sh:
                raise ValueError('data needs to be same size as spectrum.')
            if not np.issubdtype(data.dtype, np.floating):
                raise TypeError('data needs to be convertible to float.')
            self.multidata = data

        self.setcurrentspectrum(self.currentspectrumid)
        self.setcurrentmeanspectrum(self.currentspectrumid, 1, 1)

        # Array which points to exclude
        self.exclude = np.zeros(self.size, dtype=bool)
        self.pppc = 1.0
        self.name = 'a multispectrum'

    @classmethod
    def load(cls, filename, flip_sign=False):
        ext = os.path.splitext(filename)[-1]
        if ext == '.hdf5':
            s = cls.load_hdf5(filename)

        elif ext == '.hspy':
            s = cls.load_hspy(filename)

        elif (ext == '.dm3') or (ext == '.dm4'):
            s = cls.load_dm(filename, flip_sign=flip_sign)

        else:
            raise ValueError(r'Extension is not valid')

        return s

    @classmethod
    def load_hdf5(cls, filename):
        """
        Loads the multispectrum from .hdf5 file

        Parameters
        ----------
        filename : string
              Filename containing the data.

        Returns
        ----------
        s: MultiSpectrum
            The multispectrum which is inside the filename

        """
        data, dispersion, offset, size = load_h5py(filename)
        specshape = MultiSpectrumshape(dispersion, offset, size, data.shape[0],
                                       data.shape[1])
        s = MultiSpectrum(specshape, data=data)

        # how to return alternate extra data like haadf map?
        return s

    @classmethod
    def load_hspy(cls, filename):
        """
        Loads hyperspy data file format.

        Parameters
        ----------
        filename : string
              Filename containing the data.

        Returns
        ----------
        s: MultiSpectrum
            The multispectrum which is inside the filename
        df: list
            List containing other acquired data

        """
        params, df, detector_type = load_hspy(filename)
        specs = []
        for param in params:
            specshape = MultiSpectrumshape(param[1], param[2], param[3],
                                           param[0].shape[0],
                                           param[0].shape[1])
            s = MultiSpectrum(specshape, data=param[0], acq_time=param[4])
            specs.append(s)
        return specs, df

    @classmethod
    def load_dm(cls, filename, flip_sign=False, dispersion=None):
        """
        Loading dm data

        Parameters
        ----------
        filename : string
              Filename containing the data.
        flip_sign: boolean
            Bug when loading in data where the energy axis starts negative.
            If wrong is set to True, then this bug is worked around.
        dispersion: float
            Sometimes the dispersion in dual EELS is different, this is a
            method to force it to be the same.

        Returns
        ----------
        s: MultiSpectrum
            The multispectrum which is inside the filename
        """
        dmfile = dmReader(filename)
        print(dmfile['data'].ndim)
        if dmfile['data'].ndim == 3:
            data = np.swapaxes(dmfile['data'], 0, 2)
            e_axis = dmfile['coords'][0]

        elif dmfile['data'].ndim == 2:
            data = dmfile['data'][:, np.newaxis, :]
            e_axis = dmfile['coords'][1]

        multi_specshape = MultiSpectrumshape(e_axis[1] - e_axis[0], e_axis[0],
                                             e_axis.size, data.shape[0],
                                             data.shape[1])
        ms = MultiSpectrum(multi_specshape, data=data)
        if dispersion is not None:
            ms.dispersion = dispersion

        # bug which is something that you can manually solve.
        if flip_sign:
            ms.offset = -1 * ms.offset

        return ms

    @classmethod
    def from_numpy(cls, data_array, energy_axis):
        """
        Creates an Multispectrum object from the data and energy axis using
        numpy arrays.
        Usefull when playing with different functions which are not (yet)
        integrated into pyEELSmodel functionalities.

        Parameters
        ----------
        data_array : numpy array (1D)
            The data of the EEL spectrum.
        energy_axis: numpy array (1D)
            The energy axis used [eV]

        Returns
        ----------
        s: MultiSpectrum
            The multispectrum which is inside the filename

        """
        dispersion = energy_axis[1] - energy_axis[0]
        offset = energy_axis[0]
        size = energy_axis.size
        xsize = data_array.shape[0]
        ysize = data_array.shape[1]
        specshape = MultiSpectrumshape(dispersion, offset, size, xsize, ysize)
        s = MultiSpectrum(specshape, data=data_array)
        return s

    @property
    def multidata(self):
        return self._multidata

    @multidata.setter
    def multidata(self, multidata):
        self._multidata = multidata
        self.setcurrentspectrum((0, 0))

    def indexOK(self, id):
        """
        Index is valid for the scan dimensions of the EELS map

        Parameters
        ----------
        id : tuple int
              Check if the given index is valid.

        Returns
        ----------
        bool:
            Returns True if the index is valid.

        """
        if (0 <= id[0] < self.xsize) and (0 <= id[1] < self.ysize):
            return True
        return False

    def get_multispectrumshape(self):
        sh = MultiSpectrumshape(self.dispersion, self.offset, self.size,
                                self.xsize, self.ysize)
        return sh

    def __getitem__(self, key):
        """
        [] operator, makes a copy of the multispectrum where key
        is the indexing used.

        Parameters
        ----------
        key : indexing
            Indexing of the multispectrum can be performed as is done with
            numpy arrays.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        s: MultiSpectrum/Spectrum
            The sliced multispectrum or spectrum
        TODO add some warining when slicing the energy loss spectrum because
        this is something we do not want
        """

        ndata = self.multidata[key]

        # check if there is an int in the key list and make sure to add this
        # dimension in the data
        for i, val in enumerate(key):
            if isinstance(val, int):
                ndata = np.expand_dims(ndata, axis=i)

        # check if the energy axis is sliced
        if ndata.shape[-1] != self.size:
            if key[-1].start is None:
                offset = self.offset
            else:
                offset = self.energy_axis[key[-1].start]
            if key[-1].step is None:
                dispersion = self.dispersion
            else:
                dispersion = self.dispersion * key[-1].step
            size = ndata.shape[-1]
        else:
            offset = self.offset
            dispersion = self.dispersion
            size = self.size

        # return a single spectrum and not multispectrum when only one spectrum
        # is selected
        if (ndata.shape[0] == 1) and (ndata.shape[1] == 1):
            ms = Spectrumshape(dispersion, offset, size)
            return Spectrum(ms, data=ndata[0, 0])
        else:
            ms = MultiSpectrumshape(dispersion, offset, size, ndata.shape[0],
                                    ndata.shape[1])
            return MultiSpectrum(ms, data=ndata)

    def sum(self, axis=(0, 1)):
        """
        Sum the real space coordinates with each other. When summed over
        both direction, a single spectrum is returned instead of a
        multispectrum

        Parameters
        ----------
        axis : tuple
            The axis over which the sum needs to be performed.

        Returns
        -------
        m: MultiSpectrum/Spectrum
            The summed multispectrum or spectrum
        """
        # Cannot sum over the energy axis only the real space coordinates
        if isinstance(axis, int):
            if axis == 2:
                raise ValueError(r'Cannot sum over the energy axis')
        else:
            for el in axis:
                if el == 2:
                    raise ValueError(r'Cannot sum over the energy axis')

        ndata = self.multidata.sum(axis=axis)

        if ndata.ndim == 1:
            ms = Spectrumshape(self.dispersion, self.offset, self.size)
            m = Spectrum(ms, data=ndata)

        else:
            ndata = np.expand_dims(ndata, axis)
            ms = MultiSpectrumshape(self.dispersion, self.offset, self.size,
                                    ndata.shape[0], ndata.shape[1])
            m = MultiSpectrum(multispectrumshape=ms, data=ndata)

        m.exclude = self.exclude[:]  # todo nicer way of copying the exclude
        return m

    def mean(self, axis=(0, 1)):
        """
        Mean the real space coordinates with each other. When summed over
        both direction, a single spectrum is returned instead of a
        multispectrum

        Parameters
        ----------
        axis : tuple
            The axis over which the mean needs to be performed.

        Returns
        -------
        s: MultiSpectrum/Spectrum
            The
            summed multispectrum or spectrum
        """
        if axis == (0, 1):
            n_spec = self.xsize * self.ysize
        elif axis == 0:
            n_spec = self.xsize
        elif axis == 1:
            n_spec = self.ysize

        s = self.sum(axis)
        s.data = s.data / n_spec
        return s

    def integrate(self, window=None, index_type=False):
        """
        Integrates the signal over the given energy window or
        the given index.

        Parameters
        ----------
        window : tuple
            The window over which to integrate. If index_type is True,
            the integration window is given with the index instead of energy.
            If window is None, the integral over the entire energy range is
            given.
        index_type: bool
            Indicates if the window should be interpreted as the indices or
            the energy of the spectrum

        Returns
        -------
        result : numpy array of float of sizex,sizey
            The value of the integration

        """
        if window is None:
            window = [0, self.size]
            index_type = True

        if index_type:
            result = self.multidata[:, :, window[0]:window[1]].sum(-1)
        else:
            ind0 = self.get_energy_index(window[0])
            ind1 = self.get_energy_index(window[1])
            result = self.multidata[:, :, ind0:ind1].sum(-1)
        return result

    def setcurrentspectrum(self, index):
        """
        Sets the current spectrum to the given index

        Parameters
        ----------
        index : tuple
            The index to which the multispectrum is set.

        Returns
        -------
        None

        """
        if self.indexOK(index):
            self.currentspectrumid = index
            self.data = self.multidata[index[0], index[1], :]
        else:
            raise IndexError('Index out of bounds multispectrum')

    def setcurrentmeanspectrum(self, index, width, height):
        """
        Sets the current mean spectrum to the given index using the proper
        width and height. Usefull for the GUI.

        Parameters
        ----------
        index : tuple
            The index to which the multispectrum is set.

        Returns
        -------
        None

        """
        if self.indexOK(index):
            xx = [index[0], index[0] + width]
            yy = [index[1], index[1] + height]
            ndata = self.multidata[xx[0]:xx[1], yy[0]: yy[1], :].mean((0, 1))
            self.meandata = ndata
        else:
            raise IndexError('Index out of bounds multispectrum')

    def __iter__(self):
        return MultiSpectrumIterator(self)

    def swap_xy(self):
        """
        Function which will swap the x and y coordinates of the scan
        and return this as a new multispectrum.

        :return:
        """
        sh = MultiSpectrumshape(self.dispersion, self.offset, self.size,
                                self.ysize, self.xsize)
        s = MultiSpectrum(sh, data=np.swapaxes(self.multidata, 0, 1))
        return s

    def copy(self):
        """
        Returns a copy of the object

        """
        return copy.deepcopy(self)

    def geteshift(self):
        """
        Get energy shift. This shift allows to move the whole energy scale up
        and down by calling seteshift(energy)

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.eshift

    def __mul__(self, other):
        """
        Multiply two spectra with each other. All properties of the self
        spectrum are copied, except the data is multiplied with the data of
        the other spectrum. If other is an int or float, the spectrum is
        multiplied by that number.

        Parameters
        ----------
        other : MultiSpectrum
            The multispectrum to multiply with.

        Raises
        ------
        ValueError
            In case both spectra don't have the same spectrumshape settings
            (in which case multiply makes no sense).
        TypeError
            In case other is neither a Spectrum, int or float.

        Returns
        -------
        s : Spectrum
            Returns a reference to a -new- spectrum that is the multiplication
            result.

        """
        if isinstance(other, MultiSpectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Dispersion, offset or size are not the same')

            else:
                s = self.copy()
                s.data = self.data * other.data
                s.multidata = self.multidata * other.multidata
                return s

        else:
            raise TypeError('Input should be spectrum object, float or int')

    def __add__(self, other):
        """
        Add two spectra. All properties of the self spectrum are
        copied, except the data is added with the data of the other spectrum.
        If other is an int or float, the number is added to that spectrum.

        Parameters
        ----------
        other : MultiSpectrum
            The multispectrum to add.

        Raises
        ------
        ValueError
            In case both spectra don't have the same spectrumshape settings
            (in which case adding makes no sense).
        TypeError
            In case other is neither a Spectrum, int or float.

        Returns
        -------
        s : MultiSpectrum
            Returns a reference to a -new- spectrum that is the addition
             result.

        """
        if isinstance(other, MultiSpectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Disperion, offet or size are not the same')

            else:
                s = self.copy()
                s.data = self.data + other.data
                s.multidata = self.multidata + other.multidata
                return s

        else:
            raise TypeError('Input should be spectrum object, float or int')

    def __sub__(self, other):
        """
        Subtract two spectra. All properties of the self spectrum are
        copied, except the data is added with the data of the other spectrum.
        If other is an int or float, the number is subtracted from that
        spectrum.

        Parameters
        ----------
        other : MultiSpectrum
            The multispectrum to subtract.

        Raises
        ------
        ValueError
            In case both spectra don't have the same spectrumshape settings
            (in which case subtracting makes no sense).
        TypeError
            In case other is neither a Spectrum, int or float.

        Returns
        -------
        s : Spectrum
            Returns a reference to a -new- spectrum that is the subtraction
            result.

        """
        if isinstance(other, MultiSpectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Dispersion, offset or size are not the same')

            else:
                s = self.copy()
                s.data = self.data - other.data
                s.multidata = self.multidata - other.multidata

                return s

        else:
            raise TypeError('Input should be MultiSpectrum object, float or '
                            'int')

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        """
        Divide two spectra by each other. All properties of the self spectrum
        are copied, except the data is divided with the data of the other
        spectrum. If other is an int or float, the spectrum is divided by that
        number.

        Parameters
        ----------
        other : MultiSpectrum
            The spectrum to divide by.

        Raises
        ------
        ValueError
            In case both spectra don't have the same spectrumshape settings
            (in which case divide makes no sense).
        TypeError
            In case other is neither a Spectrum, int or float.

        Returns
        -------
        s : Spectrum
            Returns a reference to a -new- spectrum that is the division
            result.

        """
        if isinstance(other, MultiSpectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Dispersion, offset or size are not the same')

            else:
                s = self.copy()
                s.data = self.data / other.data
                s.multidata = self.multidata / other.multidata

                return s

        else:
            raise TypeError('Input should be spectrum object, float or int')

    def get_interval(self, interval, even_size=True):
        """
        Parameters
        ----------
        interval : tuple
            Interval which has the start and end of the sub spectrum
        even_size: bool
            Makes the size of the returned multispectrum even to accomodate for
             a fast low loss convolution
        Raises
        ------
        ValueError
            In case both spectra don't have the same spectrumshape settings
            (in which case divide makes no sense).
        TypeError
            In case other is neither a Spectrum, int or float.

        Returns
        -------
        s : MultiSpectrum
            Returns a reference to a -new- spectrum that is the division
            result.

        """

        ind0 = self.get_energy_index(interval[0])
        ind1 = self.get_energy_index(interval[1])

        is_even = (ind1 - ind0) % 2
        if (is_even != 0) & even_size:
            print('Make the size of the energy spectrum even')
            ind1 -= 1

        ndata = self.multidata[:, :, ind0:ind1]
        noffset = self.offset + ind0 * self.dispersion
        sh = MultiSpectrumshape(self.dispersion, noffset, ndata.shape[2],
                                self.xsize, self.ysize)
        s = MultiSpectrum(sh, data=ndata)
        return s

    def _check_not_same_settings(self, other):
        """
        Checks if the other spectrum has the same spectrumshape properties as
        self.

        Parameters
        ----------
        other : Spectrum
            A spectrum to compare with.

        Returns
        -------
        bool
            True if both spectra have the same spectrumshape, False otherwise.

        """

        return (self.dispersion != other.dispersion) or (
                    self.offset != other.offset) or (
                           self.size != other.size) or (
                           self.xsize != other.xsize) or (
                           self.ysize != other.ysize)

    #   def _print_type_warning(self):
    #       print('Input should be spectrum object, float or int')

    def plot(self, index=None, externalplt=None, **kwargs):
        """
        Plot the spectrum

        Parameters
        ----------
        externalplt : matplotlib reference
              A reference to an external matplotlib reference, if None we use
              our own matplotlib and create a new figure.
        """
        tempplt = plt
        if isinstance(externalplt, plt.Figure):
            tempplt = externalplt
        else:
            # create our own figure
            plt.figure()

        if index is not None:
            self.setcurrentspectrum(index)

        tempplt.plot(self.energy_axis, self.data, **kwargs)
        tempplt.xlabel(r'Energy Loss [eV]')
        tempplt.ylabel('Counts')
        tempplt.title(self.currentspectrumid)

    def show_excluded_region(self, **kwargs):
        fig, ax = plt.subplots()
        ax.plot(self.energy_axis, self.mean().data, color='black')
        ax.fill_between(self.energy_axis, 0, self.mean().data.max(),
                        where=self.exclude, color='green', alpha=0.5)

    def erase(self):
        self.data = np.zeros(self.size)

    def gaussian_3d(self, X, Y, Z, sigma):
        """
        Computes the 3 dimensional gaussian where sigma can be non isotropic.

        Parameters
        ----------
        X : numpy array (3D)
            Meshgrid indicating the x coordinate of the mesh.
        Y : numpy array (3D)
            Meshgrid indicating the x coordinate of the mesh.
        Z : numpy array (3D)
            Meshgrid indicating the x coordinate of the mesh.
        sigma : tuple of floats
            Tuple of length 3 of the sigmas of each gaussian.
        Returns
        -------
        f: numpy array (3D)
            Three dimensional gaussian.

        """
        f = np.exp(
            -((X ** 2 / (2 * sigma[0] ** 2)) + (Y ** 2 / (2 * sigma[1] ** 2))
              + (Z ** 2 / (2 * sigma[2] ** 2))))
        return f

    def rebin(self, factor):
        """
        Rebins the dataset with given factor which contains three uints
        indicating the binning factor for each dimension. Rebinning is
        performed by convoluting the dataset with a tophat function of given
        size. This makes that some edge artifacts will remain.

        Parameters
        ----------
        factor : [a1, a2, a3]
            The three binning factors. The values of a1, a2 and a3 >= 1 and
            integer.

        Returns
        -------
        None.
        """
        # todo use fftconvolve and zero padding

        if (factor[0] == 1) & (factor[1] == 1) & (factor[2] == 1):
            return self

        top_hat = np.ones(factor)
        top_hat = top_hat / top_hat.sum()
        # note that this is emperical found and I do not know
        # if this is true for everything
        sind = (np.array(factor) - 1) // 2

        res = ndimage.convolve(self.multidata, top_hat, mode='constant',
                               cval=0.0)

        ndat = res[sind[0]::factor[0], sind[1]::factor[1], sind[2]::factor[2]]
        spc = MultiSpectrumshape(factor[2] * self.dispersion,
                                 self.offset + factor[2] * self.dispersion,
                                 ndat.shape[2], ndat.shape[0], ndat.shape[1])
        s = MultiSpectrum(spc, data=ndat)
        return s

    def gaussiansmooth(self, sigma, crop=False):
        """
        Uses the 3D FFT to compute a gaussian blurred EELS map.
        Note the this performs a 3d fft which makes that the computation needs
        a lot of RAM

        Parameters
        ----------
        sigma : [a1, a2, a3]
            The sigma value of the 3d gaussian.

        crop: boolean
            Indicates whether the filtered spectrum will be

        Returns
        -------
        cop: MultiSpectrum
            The gaussian smoothed multispectrum
        """

        fft_sig = np.pi ** 2 / np.array(sigma)  # The sigma for FFT Gaussian

        xax = np.arange(self.xsize) - self.xsize / 2
        yax = np.arange(self.ysize) - self.ysize / 2
        X, Y, E = np.meshgrid(yax, xax, self.energy_axis
                              - self.energy_axis[int(self.size / 2)])

        FFT_d = np.fft.fftshift(np.fft.fftn(self.multidata))
        gauss = self.gaussian_3d(X, Y, E, fft_sig)
        res = np.real(np.fft.ifftn(np.fft.ifftshift(FFT_d * gauss)))

        if crop:
            # TODO better implementation of cropping can be thought off
            crop = [int(np.ceil(4 * sig)) for sig in sigma]
            crop[2] = int(np.ceil(crop[2] / self.dispersion))
            ndata = res[crop[0]:-crop[0], crop[1]:-crop[1], crop[2]:-crop[2]]
            noffset = crop[2] * self.dispersion + self.offset
            ms = MultiSpectrumshape(self.dispersion, noffset, ndata.shape[2],
                                    ndata.shape[0], ndata.shape[1])
            cop = MultiSpectrum(ms, data=ndata)
        else:
            cop = self.copy()
            cop.multidata = res

        return cop

    def firstderivative(self):
        """
        Calculates the first derivative of the multispectrum with respect to
        the energy axis. At this point I do not see any reason on implementing
        this for the scanning dimensions.

        Returns
        -------
        s: MultiSpectrum
            The first derivative for the multispectrum
        """

        differential = np.zeros_like(self.multidata)
        differential[:, :, 0] = self.multidata[:, :, 1] \
            - self.multidata[:, :, 0]
        differential[:, :, -1] = self.multidata[:, :, -1] \
            - self.multidata[:, :, -2]
        differential[:, :, 1:-1] = .5 * self.multidata[:, :, 2:] \
            - .5 * self.multidata[:, :, 0:-2]
        differential /= self.dispersion

        nspec = MultiSpectrumshape(self.dispersion, self.offset, self.size,
                                   self.xsize, self.ysize)

        s = MultiSpectrum(nspec, data=differential)
        return s

    def secondderivative(self):
        """
        Calculates the second derivative of the multispectrum with respect to
        the energy axis.

        Returns
        -------
        s: MultiSpectrum
            The second derivative for the multispectrum
        """
        differential = np.zeros_like(self.multidata)
        differential[:, :, 0] = self.multidata[:, :, 0] \
            - 2. * self.multidata[:, :, 1] + self.multidata[:, :, 2]
        differential[:, :, -1] = self.multidata[:, :, -1] - 2. \
            * self.multidata[:, :, -2] + self.multidata[:, :, -3]
        differential[:, :, 1:-1] = self.multidata[:, :, 0:-2] - 2. \
            * self.multidata[:, :, 1:-1] + self.multidata[:, :, 2:]

        differential /= self.dispersion ** 2

        nspec = MultiSpectrumshape(self.dispersion, self.offset,
                                   differential.shape[2], self.xsize,
                                   self.ysize)
        s = MultiSpectrum(nspec, data=differential)
        return s

    def upsample(self, nn):
        """
        Upsamples the spectrum with factor nn.

        Parameters
        ----------
        nn : uint
            The number of times the spectrum needs to be upsampled

        Returns
        -------
        spec_up: MultiSpectrum
            The upsampled multispectrum


        """
        shape = (self.xsize, self.ysize)
        spc = Spectrumshape(self.dispersion / nn, self.offset,
                            nn * self.size)
        s = Spectrum(spc)
        up = np.zeros(shape + (spc.size,))

        for index in tqdm(np.ndindex(shape),total=np.prod(shape),leave=True,position=0):
            islice = np.s_[index]

            self.setcurrentspectrum(islice)
            int_spec = self.interp_to_other_energy_axis(s)
            up[islice] = int_spec.data
        spec_up = MultiSpectrum.from_numpy(up, int_spec.energy_axis)

        return spec_up

    def map_to_line(self):
        """
        Make a map into a line scan
        :return:
        """
        new_shape = (int(self.xsize * self.ysize), 1, self.size)
        reshaped = np.reshape(self.multidata, new_shape)
        nspec = MultiSpectrumshape(self.dispersion, self.offset, self.size,
                                   new_shape[0], 1)
        s = MultiSpectrum(nspec, data=reshaped)
        return s

    def to_logarithm(self):
        s = self.copy()
        s.multidata[s.multidata < 1] = 1
        s.multidata = np.log(s.multidata)
        return s

    def apply_dark_and_gain(self, gain_name, dark_name):
        s = self.apply_dark(dark_name)
        gain = np.load(gain_name, allow_pickle=True)
        s.multidata = gain * s.multidata

        return s

    def apply_dark(self, dark_name):
        s = self.copy()
        dark = np.load(dark_name, allow_pickle=True)
        s.multidata -= dark
        return s

    def save_hdf5(self, filename, metadata=None, overwrite=False):
        file_exists = exists(filename)
        if not overwrite and file_exists:
            logger.info(
                r'File already exists, set overwrite to True '
                r'if you want to overwrite existing file')
            print('not overwriting')
            return False

        if file_exists:
            os.remove(filename)

        f = h5py.File(filename, 'w')
        f.create_dataset('Data', data=self.multidata)
        d = f.create_group('Energy_axis')

        d.attrs['Dispersion'] = self.dispersion
        d.attrs['Offset'] = self.offset
        d.attrs['Size'] = self.size

        if metadata is not None:
            print('saving metadata')
            if 'alpha' in metadata:
                d.attrs['alpha'] = metadata['alpha']
            if 'beta' in metadata:
                d.attrs['beta'] = metadata['beta']
            if 'E0' in metadata:
                d.attrs['E0'] = metadata['E0']
            if 'elements' in metadata:
                d.attrs['elements'] = metadata['elements']
            if 'edges' in metadata:
                d.attrs['edges'] = metadata['edges']
            if 'Layers' in metadata:
                print('saving layers')
                f.create_dataset('Layers', data=metadata['Layers'])
            if 'Layernames' in metadata:
                print('saving layers')
                f.create_dataset('Layernames', data=metadata['Layernames'])
            if 'Layerlow' in metadata:
                print('saving layers')
                f.create_dataset('Layerlow', data=metadata['Layerlow'])
            if 'Layerhigh' in metadata:
                print('saving layers')
                f.create_dataset('Layerhigh', data=metadata['Layerhigh'])
            if 'Layervisible' in metadata:
                print('saving layers')
                f.create_dataset('Layervisible', data=metadata['Layervisible'])

        f.close()
        return True
