"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from os.path import exists
from scipy import interpolate
import os
from scipy.ndimage.filters import gaussian_filter

from pyEELSMODEL.io_tools.dm_ncempy import dmReader
from pyEELSMODEL.io_tools.hdf5_io import load_h5py
# from pyEELSMODEL import __file__  # when icon is used
# from PyQt5 import QtGui  # when icon is used

import copy
import logging

logger = logging.getLogger(__name__)


class Spectrumshape():
    """
    Spectrumshape is a class holding the main parameters of a spectrum in order
    to compactly create several spectra with the same size by providing an
    instance of this class
    It holds:
        dispersion: float, dispersion in eV/pixel describing the energy scale
        offset: float, energy of the first pixel in the spectrum in eV
        size: int, the size of a spectrum in number of pixels (eg typ 1024)
    """

    def __init__(self, dispersion, offset, size):
        self.dispersion = dispersion
        self.offset = offset
        self.size = size


class Spectrum:
    """
    Spectrum object which contains the experimental data.
    """

    def __init__(self, spectrumshape, data=None, acq_time=1, pppc=1):

        """
        Initialises a Spectrum instance

        Parameters
        ----------
        spectrumshape : Spectrumshape
            holds the basic parameters dispersion [eV/pixel], offset [eV] and
            size [pixels] to create the energy scale of the spectrum
        data : float or int, optional
            Holds the EELS intensity data with same size as spectrumshape.size.
            If None is given an empty spectrum is created with all zeros.
            (default is None).
        acq_time: float [s]
            The acquisition time for the spectrum. This attribute gives the
            ability to compare spectra with each when other acquisition
            times are used. (default: 1)
        pppc: primary particles per count, this is the value one electron
            generates.

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
        self._dispersion = spectrumshape.dispersion
        self._offset = spectrumshape.offset
        self.size = spectrumshape.size
        self.eshift = 0
        self._set_energy_axis()

        if data is None:
            self.data = np.zeros(self.size)

        else:
            if len(data) != self.size:
                raise ValueError('data needs to be same size as spectrum.')
            if type(float(data[0])) != float:
                raise TypeError('data needs to be convertible to float.')
            self.data = data
        # Array which points to exclude
        self.exclude = np.zeros(self.size, dtype=bool)
        self.pppc = pppc
        self.name = 'a spectrum'
        self.acq_time = acq_time

        logger.debug('Spectrum init succeeded')

    @classmethod
    def load(cls, filename, flip_sign=False, **kwargs):
        """
        Loads different types of data. The possible datatypes are: .hdf5,
        .hspy, .dm3/.dm4 and .msa.

        Parameters
        ----------
        filename : string
           String of the filename where the extention should be '.hdf5'
        flip_sign: boolean
            Indicates whether the offset value should be negative when loading
            a .dm file. (default: False)

        Returns
        -------
        s: Spectrum
            The spectrum which is contained in the filename

        """
        ext = os.path.splitext(filename)[-1]

        if ext == '.hdf5':
            s = cls.load_hdf5(filename, **kwargs)

        elif ext == '.hspy':
            s = cls.load_hspy(filename)

        elif (ext == '.dm3') or (ext == '.dm4'):
            s = cls.load_dm(filename, flip_sign)

        elif ext == '.msa':
            s = cls.loadmsa(filename)

        else:
            raise ValueError(r'Extension is not valid')

        return s

    @classmethod
    def load_hdf5(cls, filename, return_metadata=False):
        """
        Loads the hdf5 file.

        Parameters
        ----------
        filename : string
           String of the filename where the extention should be '.hdf5'

        Returns
        -------
        s: Spectrum
            The spectrum which is contained in the filename

        """
        data, dispersion, offset, size = load_h5py(filename)

        specshape = Spectrumshape(dispersion, offset, size)
        s = Spectrum(specshape, data=data)
        if return_metadata:
            return s,
        else:
            return s

    @classmethod
    def load_hspy(cls):
        pass

    @classmethod
    def from_numpy(cls, data_array, energy_axis):
        """
        Creates a Spectrum object from the data and energy axis using
        numpy arrays. Usefull when playing with different functions which are
        not (yet) integrated into pyEELSmodel functionalities.

        Parameters
        ----------
        data_array : numpy array (1D)
            The data of the EEL spectrum.
        energy_axis: numpy array (1D)
            The energy axis used [eV]

        Returns
        ----------
        s: Spectrum

        """
        dispersion = energy_axis[1] - energy_axis[0]
        offset = energy_axis[0]
        size = energy_axis.size
        specshape = Spectrumshape(dispersion, offset, size)
        s = Spectrum(specshape, data=data_array)
        return s

    def _set_energy_axis(self):
        """
        Prepares the energy axis
        Returns
        -------
        None.
        """
        self._energy_axis = np.arange(self.size, dtype='float') * \
            self._dispersion + self._offset + self.eshift

    @property
    def energy_axis(self):
        return self._energy_axis

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset
        self._set_energy_axis()

    @property
    def dispersion(self):
        return self._dispersion

    @dispersion.setter
    def dispersion(self, dispersion):
        self._dispersion = dispersion
        self._set_energy_axis()

    @property
    def acq_time(self):
        """
        The acquisition time used for spectra. Can be usefull when splicing
        multiple spectra together.

        """
        return self._acq_time

    @acq_time.setter
    def acq_time(self, acq_time):
        self._acq_time = acq_time

    def setname(self, name):
        """
        Set the name of a spectrum. Can be used when displaying the spectrum.
        Could hold e.g. the filename if loaded from a file.
        """
        self.name = name

    def copy(self):
        """
        Returns a copy of the object
        """
        return copy.deepcopy(self)

    @property
    def eshift(self):
        return self._eshift

    @eshift.setter
    def eshift(self, eshift):
        self._eshift = eshift
        self._set_energy_axis()

    def __mul__(self, other):
        """
        Multiply two spectra with each other. All properties of the self
        spectrum are copied, except the data is multiplied with the data of the
        other spectrum. If other is an int or float, the spectrum is multiplied
        by that number.

        Parameters
        ----------
        other : Spectrum
            The spectrum to multiply with.

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
        if isinstance(other, Spectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Dispersion, offet or size are not the same')
            else:
                s = self.copy()
                s.data = self.data * other.data
                return s

        elif type(other) is int or type(other) is float:
            s = self.copy()
            s.data = self.data * other
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
        other : Spectrum
            The spectrum to add.

        Raises
        ------
        ValueError
            In case both spectra don't have the same spectrumshape settings
            (in which case adding makes no sense).
        TypeError
            In case other is neither a Spectrum, int or float.

        Returns
        -------
        s : Spectrum
            Returns a reference to a -new- spectrum that is the addition
            result.

        """
        if isinstance(other, Spectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Disperion, offet or size are not the same')

            else:
                s = self.copy()
                s.data = self.data + other.data
                return s

        elif type(other) is int or type(other) is float:
            s = self.copy()
            s.data = self.data + other
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
        other : Spectrum
            The spectrum to subtract.

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
        if isinstance(other, Spectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Dispersion, offet or size are not the same')

            else:
                s = self.copy()
                s.data = self.data - other.data
                return s
        elif type(other) is int or type(other) is float:
            s = self.copy()
            s.data = self.data - other
            return s

        else:
            raise TypeError('Input should be spectrum object, float or int')

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
        other : Spectrum
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
        if isinstance(other, Spectrum):
            if self._check_not_same_settings(other):
                raise ValueError('Dispersion, offset or size are not the same')

            else:
                s = self.copy()
                s.data = self.data / other.data
                return s

        elif type(other) is int or type(other) is float:
            s = self.copy()
            s.data = self.data / other
            return s

        else:
            raise TypeError('Input should be spectrum object, float or int')

    def __getitem__(self, key):
        """
        [] operator, makes a copy of the spectrum where key
        is the indexing used. Slicing the energy axis is not possible and
        the get_interval method should be used.


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

        #TODO add some warining when slicing the energy loss spectrum
        """

        # check if the energy axis is sliced
        print(type(key))
        if type(key) is dict:
            if key.start is None:
                ind0 = 0
            else:
                ind0 = key.start
        elif type(key) is int:
            ind0 = key
        elif type(key) is slice:
            if key.start is None:
                ind0 = 0
            else:
                ind0 = key.start
        else:
            raise TypeError('__getitem__ key should be dictionary or int or '
                            'slice')
        ndata = self.data[key]
        noffset = self.offset + ind0 * self.dispersion
        sh = Spectrumshape(self.dispersion, noffset, ndata.size)
        s = Spectrum(sh, data=ndata)
        return s

    def get_interval(self, interval):
        """
        Returns a subspectrum of the given interval.

        Parameters
        ----------
        interval : tuple
            Interval which has the start and end of the sub spectrum [eV]

        Returns
        -------
        s : Spectrum
            Sub spectrum given by the interval.

        """

        ind0 = self.get_energy_index(interval[0])
        ind1 = self.get_energy_index(interval[1])
        s = self[ind0:ind1]
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
        bl0 = self.dispersion != other.dispersion
        bl1 = self.offset != other.offset
        bl2 = self.size != other.size

        return bl0 or bl1 or bl2

    def get_specparameters(self):
        """
        Returns a tuple containing dispersion, offset, size of spectrum.
        Consider using get_spectrumshape() instead which returns a class
        holding the same information but in a way it can more easily be used to
        construct a new spectrum with same basic properties

        Returns
        -------
        dispersion : float
            Energy dispersion in eV/pixel.
        offset : float
            Energy of the first bin in the spectrum in eV.
        size : int
            Number of spectral bins.

        """
        return (self.dispersion, self.offset, self.size)

    def get_spectrumshape(self):
        """
        Return a Spectrumshape object holding information on the dispersion,
        offset, size of the Spectrum.

        Returns
        -------
        shape : Spectrumshape
            A spectrumshape object.

        """
        shape = Spectrumshape(self.dispersion, self.offset, self.size)
        return shape

    def bad_index(self, index: int):
        """
        Check if index is in range otherwise raises a ValueError

        Parameters
        ----------
        index : int
            Index from 0..size into the spectral data.

        Raises
        ------
        ValueError
            If index <0 or index>self.size.

        Returns
        -------
        None.

        """
        if index < -1 or index > self.size:
            raise ValueError('Bad index value')

    def set_exclude_region_energy(self, Ei, Ef):
        """
        Sets the excluded region using the energy.

        Parameters
        ----------
        Ei: float
            The starting energy of the excluded region.
        Ef: float
            The end energy of the excluded region.

        """
        index_i = self.get_energy_index(Ei)
        index_f = self.get_energy_index(Ef)
        self.set_exclude_region(index_i, index_f)

    def set_include_region_energy(self, Ei, Ef):
        """
        Sets the included region using the energy.

        Parameters
        ----------
        Ei: float
            The starting energy of the excluded region.
        Ef: float
            The end energy of the excluded region.

        """
        index_i = self.get_energy_index(Ei)
        index_f = self.get_energy_index(Ef)
        self.set_include_region(index_i, index_f)

    def set_exclude_region(self, index_i, index_f):
        """
        Set a range of pixels from index_i to index_f (both inclusive) as
        'excluded'. This means the Fitter will not take into account these
        spectral regions. Note that you can call this function multiple times
        on different index ranges to define multiple areas of exclusion.

        Parameters
        ----------
        index_i: int
            starting index for the exclusion area.
        index_f: int
            end index for the exclusion area.

        Returns
        -------
        None.

        """
        self.bad_index(index_i)
        # if the start or end index are not inside the region, throw error
        self.bad_index(index_f)
        self.exclude[index_i:index_f] = True

    def set_include_region(self, index_i, index_f):
        """
        Set a range of pixels from index_i to index_f (both inclusive) as
        'included'. This means the Fitter will take into account these spectral
        regions. Note that you can call this function multiple times on
        different index ranges to define multiple areas of exclusion.

        Parameters
        ----------
        index_i : int
            starting index for the inclusion area.
        index_f : int
            end index for the inclusion area.

        Returns
        -------
        None.

        """
        self.bad_index(index_i)
        # if the start or end index are not inside the region, throw error
        self.bad_index(index_f)
        self.exclude[index_i:index_f] = False

    def reset_exclude_region(self, index_i=0, index_f=-1):
        """
        Reset a range of pixels from index_i to index_f (both inclusive) from
        being 'excluded'. This means the Fitter will take into account these
        spectral regions. Note that you can call this function multiple times
        on different index ranges to define multiple areas as being not
        excluded. Note that default behaviour is that all spectral bins are not
        excluded when creating a spectrum.

        Parameters
        ----------
        index_i : int
            starting index for the inclusion area.
        index_f : int
            end index for the inclusion area.

        Returns
        -------
        None.

        """
        self.bad_index(index_i)
        # if the start or end index are not inside the region, throw error
        self.bad_index(index_f)
        self.exclude[index_i:index_f] = False

    def getmaxindex(self):
        """
        Returns the index containing the highest value in the spectrum. In case
        of multiple pixels having the same maximum value, index to the first
        occurence will be returned.
        """
        return np.argmax(self.data)

    def getmaxenergy(self):
        co = self.getmaxindex()
        return self.energy_axis[co]

    def get_first_higher_then(self, x):
        """
        Get index of the first pixel having a value higher then x

        Parameters
        ----------
        x : float
            The value of which the pixel value should be larger.

        Returns
        -------
        int
            index of the first occurence of data>x.
        None
            if no value higher than x is found

        """
        # todo add type checking of x?

        boolean = self.data > x
        if boolean.any():
            return np.argmax(boolean)
        else:
            logger.warning('No larger value then %d in the data', x)
            return None

    def get_max(self):
        """
        Returns the maximum intensity value in the spectrum.

        Returns
        -------
        float or int
            Maximum value in the EELS dataset.

        """
        return self.data.max()

    def get_min(self):
        """
        Returns the minimum intensity value in the spectrum.

        Returns
        -------
        float or int
            Minimum value in the EELS dataset.

        """
        return self.data.min()

    @property
    def pppc(self):
        """
        Get 'primary particles per count=pppc'. This value defines the number
        of electrons per given EELS count and is used to include information on
        the detector step. This is important for scintillating detectors where
        multiple counts can be generated for a single incoming electron.
        Knowing this is essential to calculate the effect of Poisson noise for
        the fitter. This only makes sense for experimental data and should not
        be used in e.g. components.

        Returns
        -------
        float
            The pppc value.

        """

        return self._pppc

    @pppc.setter
    def pppc(self, p):
        """
        Set the 'primary particles per count=pppc' value. This value defines
        the number of electrons per given EELS count and is used to include
        information on the detector step. This is important for scintillating
        detectors where multiple counts can be generated for a single incoming
        electron. Based on this, the estimated standard deviation is
        pre-calculated for the data which is used by the fitter. For Component
        objects, the default is pppc=1 and should not be set.

        Parameters
        ----------
        p : float
            the pppc value obtained e.g. from a test script determining the
            noise properties of the detector.


        Raises
        ------
        ValueError
            when pppc <0

        Returns
        -------
        None.

        """
        if p < 0:
            raise ValueError('pppc should be >=0')
        self._pppc = p
        # self.data = self.data*p
        self.init_poisson_error()

    def init_poisson_error(self):
        """
        Calculate estimated standard deviation on the data assuming underlying
        Poisson statistics. Result can be obtained with get_error(). Any
        non-positive data is ignored and result in a standard deviation of 0

        Returns
        -------
        None.

        """
        mask = self.data > 0
        self.error = np.sqrt(
            self.pppc * self.data * mask)  # only makes sense for positive data

    def get_energy_index(self, E):
        """
        Return the first index where E>energy within the spectral range

        Parameters
        ----------
        E : float
            Energy to find in this spectrum [eV].

        Returns
        -------
        int
            Index of the closest energy bin.
            0 when below the energy onset of the spectrum
            self.size when above the energy range of the spectrum

        """
        if E < self.energy_axis[0]:
            return 0
        elif E >= self.energy_axis[-1]:
            return self.size
        else:
            co = np.argmin(np.abs(self.energy_axis - E))
            return co

    def plot(self, externalplt=None, use_e_axis=True, logscale=False,
             **kwargs):
        """
        Plot the spectrum

        Parameters
        ----------
        externalplt : matplotlib reference
              A reference to an external matplotlib reference, if None we use
              our own matplotlib and create a new figure.
        use_e_axis: bool
            Indicates if the x-axis in the energy axis or if the pixel value
            of the detector will be used to visualize the data.

        """
        tempplt = plt
        if isinstance(externalplt, plt.Figure):
            tempplt = externalplt
        else:
            # create our own figure
            plt.figure()
            # thismanager = plt.get_current_fig_manager()

            # dirname= os.path.dirname(__file__) + "/../pyEELSMODEL/images/"
            # icon_name = os.path.join(dirname, 'test_logo.ico')
            # thismanager.window.wm_iconbitmap(icon_name)
            # thismanager.window.setWindowIcon(QtGui.QIcon(icon_name))

        if use_e_axis:
            tempplt.plot(self.energy_axis, self.data, **kwargs)
            tempplt.xlabel(r'Energy Loss [eV]')
        else:
            tempplt.plot(self.data, **kwargs)

        tempplt.ylabel('Counts')
        if logscale:
            tempplt.yscale('log')

    def show_excluded_region(self, **kwargs):
        """
        Shows the points which are excluded from the fit.
        """

        fig, ax = plt.subplots()
        ax.plot(self.energy_axis, self.data, color='black')
        ax.fill_between(self.energy_axis, 0, self.data.max(),
                        where=self.exclude, color='green', alpha=0.5)

    def erase(self):
        self.data = np.zeros(self.size)

    def normalise(self, method='sum', window=None):
        """
        Normalizes the spectrum using the method identified. The data is over-
        written

        Parameters
        ----------
        method : string
            The normalization method which can be 'sum' or 'max'.
            (default: 'sum')
        window: tuple
            If the method is 'sum', an integration range can be added. If
            no window is provided the sum under the entire spectrum is taken.

        """
        if method == 'max':
            self.data = self.data / self.data.max()

        elif method == 'sum':
            if window is None:
                norm = np.sum(self.data)
            else:
                norm = self.integrate(window)
            self.data = self.data / norm

        else:
            print('method is not supported')

    def _getmsatag(self, line, tag):
        """
        Look for 'tag' in string 'line' and convert value after ':' to float
        return, return -1 if tag not found, return 0 if tag found but value
        couldnt be interpreted

        Parameters
        ----------
        line : TYPE
            DESCRIPTION.
        tag : TYPE
            DESCRIPTION.
        ref : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        val = 0
        # logger.debug('getmsatag on line=%s and looking for tag=%s',line,tag)
        if line.find(tag) != -1:
            logger.debug('item found %s', tag)
            items = line.split(':')
            try:
                val = float(items[-1])
            except (ValueError, IndexError):
                # if value cant be converted we still want True as
                # this is the case for #SPECTRUM which has no value
                val = 0
            return val
        return -1

    @classmethod
    def loadmsa(cls, filename):
        """
        Load single spectrum from an MSA file

        Parameters
        ----------
        filename : string
            The name of the .msa which needs to opened.

        Returns
        -------
        s: Spectrum
            The EEL spectrum in the file

        """
        logger.warning('this will change the size and contents of the current '
                       'spectrum')
        specdata = False
        i = 0
        specshape = Spectrumshape(1, 200, 1024)
        s = Spectrum(specshape)
        with open(filename, 'r') as f:
            for line in f:
                if not specdata:
                    val = s._getmsatag(line, '#NPOINTS')
                    if val != -1:
                        s.size = int(val)
                        s.erase()
                    val = s._getmsatag(line, '#XPERCHAN')
                    if val != -1:
                        s.dispersion = val
                    val = s._getmsatag(line, '#OFFSET')
                    if val != -1:
                        s.offset = val
                    if s._getmsatag(line, '#SPECTRUM') != -1:
                        specdata = True
                else:
                    items = line.split(',')
                    try:
                        first = float(items[0])
                    except (ValueError, IndexError):
                        pass
                    try:
                        second = float(items[-1])
                    except (ValueError, IndexError):
                        pass
                    logger.debug('reading data: %s,%s', first, second)
                    if i < s.size:
                        s.data[i] = float(second)
                    i += 1
        logger.info('MSA file loaded')
        return s

    @classmethod
    def load_dm(cls, filename, flip_sign=False):
        """
        Load single spectrum from an DM file. There seems to be a bug in the
        dm loader which gives a positive offset value while it should be
        negative. This happens mainly for the low loss spectra which contain
        the zero-loss peak. The flip_sign parameter takes care of this.

        Parameters
        ----------
        filename : string
            filename of the dm file
        flip_sign: boolean
            Indicates where the offset value should be set to negative.
            (default: False)

        Returns
        -------
        s: Spectrum

        """
        dmfile = dmReader(filename)
        data = dmfile['data'][:]
        e_axis = dmfile['coords'][0]
        dispersion = e_axis[1] - e_axis[0]
        offset = e_axis[0]
        if flip_sign:
            offset = -1 * offset
        size = e_axis.size
        spc = Spectrumshape(dispersion, offset, size)
        return Spectrum(spc, data)

    def save_hdf5(self, filename, metadata=None, overwrite=False):
        """
        Saves the spectrum as a hdf5 file. The structure of the file can easily
        be investigated via a hdfview software.

        Parameters
        ----------
        filename : string
            filename of the saved file.
        metadata: dictionary
            A dictionary containing E0, alpha, beta, elements and edges can be
            added to the hdf5 file. If None is given, nothing will be saved.
        overwrite: boolean
            Indicates if the file will be overwritten if it already exists.
            (default: False)
        Returns
        -------
        If the file saving workes, a True value is returned.
        """
        file_exists = exists(filename)
        if not overwrite and file_exists:
            logger.info(r'File already exists, set overwrite to True if you '
                        r'want to overwrite existing file')
            return False
        if file_exists:
            os.remove(filename)

        f = h5py.File(filename, 'w')
        f.create_dataset('Data', data=self.data)
        d = f.create_group('Energy_axis')

        d.attrs['Dispersion'] = self.dispersion
        d.attrs['Offset'] = self.offset
        d.attrs['Size'] = self.size

        if metadata is not None:
            d.attrs['alpha'] = metadata['alpha']
            d.attrs['beta'] = metadata['beta']
            d.attrs['E0'] = metadata['E0']
            d.attrs['elements'] = metadata['elements']
            d.attrs['edges'] = metadata['edges']

        f.close()
        return True

    def gaussiansmooth(self, sigma):
        """
        Return a new spectrum which is a gaussian smoothed version of this one.

        Parameters
        ----------
        sigma : float
            Standard deviation of the gaussian with which it will be
            convolved


        Returns
        -------
        result : Spectrum
            A smoothed spectrum version of the initial spectrum.

        """
        result = self.copy()
        result.data = gaussian_filter(self.data, sigma / self.dispersion)
        return result

    def getnonexcludedpoints(self):
        """
        Returns the number of included points in the spectrum, used for the
        degrees of freedom.
        """
        return np.size(self.exclude) - np.count_nonzero(self.exclude)

    def integrate(self, window=None, index_type=False):
        # todo add test for this function
        # todo add badindex check
        """
        Integrates the signal over the given energy window or the given index.

        Parameters
        ----------
        window : tuple
            The window over which to integrate. If index_type is True,
            the integration window is given with the index instead of energy
            If window is None, the integration is performed over the entire
            energy range.
        index_type: bool
            Indicates if the window should be interpreted as the indices or
            the energy of the spectrum

        Returns
        -------
        result : float
            The value of the integration

        """
        if window is None:
            window = [0, self.size]
            index_type = True

        if index_type:
            result = self.data[window[0]:window[1]].sum()
        else:
            ind0 = self.get_energy_index(window[0])
            ind1 = self.get_energy_index(window[1])
            result = self.data[ind0:ind1].sum()
        return result

    def interp_to_other_energy_axis(self, spectrum, constant_values=(0, 0)):
        """
        Interpolates the given spectrum to another given spectrum. The region
        which does not overlap with the energy axis of spectrum will be given a
        zero value. Other methods of padding can easily be incorporated using
        numpys pad functionality. The interpolation is linear and can also be
        modified using the interp function of scipy.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum which holds energy axis to which the spectrum needs to
             be interpolated.

        Returns
        -------
        int_sepc : Spectrum
            A new spectrum which is interpolated to the given spectrums energy
            axis.
        """

        disp = self.dispersion
        dstart = -1 * min(spectrum.offset - self.offset, 0)
        dend = max(spectrum.energy_axis[-1] - self.energy_axis[-1], 0)

        # elongate the energy axis such the entire area can be interpolated.
        if dstart == 0:
            before = []
        else:
            before = np.arange(np.ceil(dstart / disp)) * disp
            before += self.offset - before[-1] - disp

        if dend == 0:
            after = []
        else:
            after = np.arange(np.ceil(dend / disp)) * disp \
                    + self.energy_axis[-1] + disp

        E_ax = np.concatenate((before, self.energy_axis, after))
        ndata = np.pad(self.data, pad_width=(len(before), len(after)),
                       constant_values=constant_values)
        f = interpolate.interp1d(E_ax, ndata)
        int_spec = Spectrum(spectrum.get_spectrumshape(),
                            data=f(spectrum.energy_axis))
        return int_spec

    def rescale_spectrum(self, scale, shift):
        """
        Modifies the EELS spectrum by keeping the energy axis the same but
        shifting and rescaling the spectrum. First the shift is applied
        and then the scaling is modified.

        Parameters
        ----------
        scale : float
            The value with which the dispersion is multiplied.

        shift: float [eV]
            The shift of the spectrum on the energy axis.

        Returns
        -------
        int_spec : Spectrum
            A new spectrum has been shifted and rescaled but has the same
            energy axis.
        """
        cop = self.copy()
        cop.offset += shift
        cop.dispersion = scale * self.dispersion
        int_spec = cop.interp_to_other_energy_axis(self)
        return int_spec

    def padding(self, padding):
        """
        Zero padding the spectrum with padding size. This function is used
        when convolving the zero loss spectrum.

        Parameters
        ----------
        padding : uint
            The number of elements padded on the left and right

        Returns
        -------
        s : Spectrum object
            A new spectrum which is zero padded.
        """
        ndata = np.pad(self.data, (padding, padding))
        n_offset = self.offset - padding * self.dispersion
        hs = Spectrumshape(self.dispersion, n_offset, ndata.size)
        s = Spectrum(hs, data=ndata)
        s.set_exclude_region(0, padding + 1)
        s.set_exclude_region(self.size + padding - 1, s.size)
        return s

    def recalibrate_spectrum(self, coefficients):
        """
        Recalibrate the spectrum using the coefficient for changing the
        dispersion and offset. These coefficients can be retrieved by linear
        fitting the measured edge onset energies compared to the literature
        values.

        Parameters
        ----------
        coefficients : numpy array 1D
            This array has two values coming from a linear fit (y = ax + b).
            The first element is the slope of the fit (a) and the second
            element is the constant (b).

        """
        n_en_axis = coefficients[0] * self.energy_axis + coefficients[1]
        disp = n_en_axis[1] - n_en_axis[0]
        offset = n_en_axis[0]
        self.dispersion = disp
        self.offset = offset

    def firstderivative(self):
        """
        Calculates the second derivative of the spectrum.
        Returns
        -------
        s: Spectrum
            The first derivative for the spectrum
        """

        differential = np.zeros_like(self.data)
        differential[0] = self.data[0] - 2 * self.data[1]
        differential[-1] = self.data[-1] - 2 * self.data[-2]
        differential[1:-1] = self.data[0:-2] - 2 * self.data[1:-1] \
            + self.data[2:]

        nspec = Spectrumshape(self.dispersion, self.offset, self.size)
        s = Spectrum(nspec, data=differential)
        return s

    def secondderivative(self):
        """
        Calculates the second derivative of the spectrum.

        Returns
        -------
        s: Spectrum
            The second derivative for the spectrum
        """
        differential = np.zeros_like(self.data)
        differential[0] = self.data[0] - 2. * self.data[1] + self.data[2]
        differential[-1] = self.data[-1] - 2. * self.data[-2] + self.data[-3]
        differential[1:-1] = self.data[0:-2] - 2. * self.data[1:-1] \
            + self.data[2:]

        differential /= self.dispersion ** 2

        nspec = Spectrumshape(self.dispersion, self.offset, self.size)
        s = Spectrum(nspec, data=differential)
        return s

    def apply_dark_and_gain(self, gain_name, dark_name):
        """
        Apply dark and gain. The dark and gain can be saved as a numpy array
        and applied to the spectrum. This is done on the data of the spectrum
        itself and no new Spectrum object is created.
        It assumes that the gain and dark is stored as a .npy file

        Parameters
        ----------
        gain_name : string
            The filename where the gain is stored.

        dark_name: string
            The filename where the dark reference is stored

        """
        self.apply_dark(dark_name)
        gain = np.load(gain_name, allow_pickle=True)
        self.data = gain * self.data

    def apply_dark(self, dark_name):
        """
        Remove the dark reference from the detector. This is done on the data
        of the spectrum itself and no new Spectrum object is created.


        Parameters
        ----------
        dark_name: string
            The filename where the dark reference is stored

        """
        dark = np.load(dark_name, allow_pickle=True)
        self.data -= dark

    def get_numerical_fwhm(self):
        """
        Small function which calculates the fwhm of a spectrum numerically
        by subtracting halve of the maximum from the spectrum and finding the
        smallest values. Note that the zero loss peak should be present else
         the result will be garbage.

        Returns
        -------
        fwhm: float
            The full width halve maximum [eV], None is returned if the
            offset energy of the spectrum is bigger or equal to zero.

        """
        if self.offset >= 0:
            print('The energy axis is positive indicating that the zero loss'
                  'peak is not present')
            return None

        rs = np.abs(self.data - 0.5 * self.data.max())
        co = np.argsort(rs)
        fwhm = np.abs(self.energy_axis[co[1]] - self.energy_axis[co[0]])
        return fwhm
