import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from tqdm import tqdm
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.misc.pdf_maker import PDF
import logging
logger = logging.getLogger(__name__)


class Align(Operator):
    """
    This class is the super class of different types of alignment procedures.
    For instance the zero loss alignment using the maximum value or model based
    fitting with Gaussian/Lorentzian. Also other methods of alignment such as
    the cross correlation can be used as child class
    """
    def __init__(self, multispectrum, other_spectra, cropping,
                 signal_range=None, zero_index=None):
        self.multispectrum = multispectrum

        if other_spectra is None:
            self.other_spectra = []
        else:
            if self.check_valid_other_spectra(other_spectra):
                self.other_spectra = other_spectra
        self.update_multispectra()
        self.cropping = cropping
        self.signal_range = signal_range
        self.zero_index = zero_index
        self.shift = None
        self.index_shift = None
        self.aligned = None
        self.aligned_others = []
        self.method = None
        self.pdf_path = r'E:\eelsmodel_pdfs'  # directory to put the pdfs

    @property
    def shift(self):
        """
        The determined shift using a method which is given in the sub-class

        Returns
        -------
        self._shift: 2d numpy array
        """
        return self._shift

    @shift.setter
    def shift(self, shift):
        """
        Sets the shift
        Parameters
        ----------
        shift : 2d numpy array
            The shift calculated using the different methods

        Returns
        -------
        None.
        """

        self._shift = shift

    @property
    def index_shift(self):
        """
        The index shift, this will be used to align the spectra using the
        roll function

        Returns
        -------
        self._index_shift: 2d numpy array (int)
            The index shift.
        """
        return self._index_shift

    @index_shift.setter
    def index_shift(self, shift):
        """
        Sets the index shift

        Parameters
        ----------
        shift : 2d numpy array (int)
            The index shift calculated using the different methods

        Returns
        -------
        None.
        """
        self._index_shift = shift

    @property
    def zero_index(self):
        """
        The reference index for the alignment of the fast align.

        Returns
        -------
        self._zero_index: int
        """
        return self._zero_index

    @zero_index.setter
    def zero_index(self, E):
        """
        Sets the reference index with the used energy E (eV).

        Returns
        -------
        None
        """
        if E is None:
            self._zero_index = None
            return

        E0 = self.multispectrum.energy_axis[0]
        E1 = self.multispectrum.energy_axis[-1]

        if (E >= E1) or (E <= E0):

            raise ValueError(r'The given energy value is not inside the energy'
                             r' axis so it cannot, align on this energy. '
                             r'Please take a energy value inside the axis.')
        self._zero_index = self.multispectrum.get_energy_index(E)

    @property
    def signal_range(self):
        """
        The signal range used in the determination of the shift/index_shift

        Returns
         -------
        self._signal_range: tuple
        """
        return self._signal_range

    @signal_range.setter
    def signal_range(self, signal_range):
        """
        Sets the signal range in eV

        Parameters
        ----------
        signal_range : tuple
            A tuple containing the signal range which is used for the
            alignment. The energy range is given in eV.

        Returns
        -------
        None
        """

        if signal_range is None:
            signal_range = (self.multispectrum.energy_axis[0],
                            self.multispectrum.energy_axis[-1])

        if signal_range[1] < signal_range[0]:
            raise ValueError(r'Second energy in the signal range should be '
                             r'bigger then the first one')

        if signal_range[0] < self.multispectrum.energy_axis[0]:
            signal_range[0] = self.multispectrum.energy_axis[0]
            logger.warning(r'Begin energy is set to start of energy axis')

        if signal_range[1] > self.multispectrum.energy_axis[-1]:
            signal_range[1] = self.multispectrum.energy_axis[-1]
            logger.warning(r'End energy is set to end of energy axis')

        self._signal_range = signal_range

    def check_valid_other_spectra(self, other_spectra):
        """
        Checks whether the given other_spectra have the same x and y size
        which is necessary to have in order to correct the drift.

        Parameters
        ----------
        other_spectra : list of MultiSpectrum
            A list containing the multispectra which uses the same alignment.

        Returns
        -------
        bool: True
            Returns True if no error is thrown.

        """
        for spec in other_spectra:
            if (spec.xsize != self.multispectrum.xsize) or \
                    (spec.ysize != self.multispectrum.ysize):
                raise ValueError(r'Scan size of the other spectra is not '
                                 r'equal')
                return False

            # elif (spec.size != self.multispectrum.size) or \
            #         (spec.dispersion != self.multispectrum.dispersion):
            #     raise ValueError(r'Dispersion or offset is not the same for '
            #                      r'the other spectra')

        return True

    def update_multispectra(self):
        """
        If the zero loss is not centered around the zero energy then we cut
        too much of when cropping. This is resolved by doing a first quick
        update on the offset energy of the multispectrum by taking the median
        value of the energy at maximum intensity.
        """
        co = np.argmax(self.multispectrum.multidata[:, :], axis=2)
        zlp_pos = self.multispectrum.energy_axis[co]
        zlp_med = np.median(zlp_pos)

        self.multispectrum.offset -= zlp_med

        for spec in self.other_spectra:
            spec.offset -= zlp_med

    def fast_align(self):
        """
        Aligns the data using a roll without any interpolation.
        parameter. The new aligned spectrum is saved in the aligned attribute.
        The additional aligned spectra can be found in the aligned_others
        attribute.

        Returns
        -------
        None.
        """
        if self.index_shift is None:
            print('The shift value should first be calculated before applying'
                  ' it')
            return None

        # make the aligned other empty
        self.aligned_others = []

        shape = (self.multispectrum.xsize, self.multispectrum.ysize)
        if self.zero_index is None:
            shf = np.copy(self.index_shift)
        else:
            shf = self.index_shift + self.zero_index

        min_shift = np.nanmin(shf)
        max_shift = np.nanmax(shf)

        # if no shift is observed, then it will not try to apply something
        if min_shift == max_shift:
            print('dataset is already aligned')
            self.aligned = self.multispectrum.copy()
            self.aligned_others = self.other_spectra[:]
            return None

        if self.cropping:
            # set the offset to the maximum shift
            new_offset = self.multispectrum.offset + \
                         self.multispectrum.dispersion * max_shift

            # apply the offset to all the other spectra
            new_offset_list = []
            for spec in self.other_spectra:
                new_offset_list.append(spec.offset +
                                       self.multispectrum.dispersion *
                                       max_shift)

            # the maximum shift should be zero
            shf -= max_shift

            # the new size of the cropped spectrum
            new_size = self.multispectrum.size - (max_shift - min_shift)
            ind = [0, -1 * (max_shift - min_shift)]

            # no odd pixels in spectrum --> For the convolution
            if new_size % 2 == 1:
                new_size -= 1
                ind = [0, -1 * (max_shift - min_shift) - 1]

            ms = MultiSpectrumshape(self.multispectrum.dispersion, new_offset,
                                    new_size, shape[0], shape[1])
            self.aligned = MultiSpectrum(ms, data=np.zeros((shape[0], shape[1],
                                                            new_size)))

            # make new multispectra with the new size of the cropped spectra
            for index, spec in enumerate(self.other_spectra):
                ms = MultiSpectrumshape(self.multispectrum.dispersion,
                                        new_offset_list[index], new_size,
                                        shape[0], shape[1])

                nshape = (shape[0], shape[1], new_size)
                self.aligned_others.append(MultiSpectrum(ms, np.zeros(nshape)))

        else:
            # if no cropping is used, everything is a lot easier :)
            self.aligned = self.multispectrum.copy()

            for spectra in self.other_spectra:
                self.aligned_others.append(spectra.copy())
            ind = [0, None]

        # apply the shifts using the numpy roll function and cropping away
        # the proper regions.
        for index in (np.ndindex(shape)):
            islice = np.s_[index]
            self.aligned.multidata[islice] \
                = np.roll(self.multispectrum.multidata[islice],
                          shf[islice])[ind[0]:ind[1]]

            # apply the shift to every other spectrum
            for ii, spec in enumerate(self.aligned_others):
                spec.multidata[islice] \
                    = np.roll(self.other_spectra[ii].multidata[islice],
                              shf[islice])[ind[0]:ind[1]]

    def align(self):
        """
        Applies the calculated shift to the given multispectrum and other given
        multispectra. The new aligned spectrum is saved in the aligned
        attribute. The additional aligned spectra can be found in the
        aligned_others attribute.
        The difference between the 'align' or 'fast_align' function is that
        align uses the interpolation to shift a spectrum whereas the
        'fast_align' uses the indices to shift so no subpixel shifts are
        possible.

        Returns
        -------
        None.
        """
        if self.shift is None:
            print('The shift value should first be calculated before applying'
                  ' it')
            return None

        # make the aligned others list empty
        self.aligned_others = []

        # determine the minimal and maximum shift
        shape = (self.multispectrum.xsize, self.multispectrum.ysize)

        # some weird prefactor and naming to make everything work
        max_shift = -1*min(np.nanmin(self.shift), 0)
        min_shift = max(np.nanmax(self.shift), 0)

        disp = self.multispectrum.dispersion

        # the zero padded array to make sure that every spectrum can
        # interpolated
        if max_shift == 0:
            ar_before = []
        else:
            ar_before = np.arange(np.ceil(max_shift / disp)) * disp
            ar_before += self.multispectrum.offset-ar_before[-1]-disp

        if min_shift == 0:
            ar_after = []
        else:
            ar_after = np.arange(np.ceil(min_shift / disp))\
                       + self.multispectrum.energy_axis[-1] + disp

        # the new energy axis where everything should be okay.
        E_ax = np.concatenate((ar_before, self.multispectrum.energy_axis,
                               ar_after))

        # cropping modifies the total size EEL spectrum
        if self.cropping:
            ind = [int(np.ceil(max_shift / disp)),
                   -1 * int(np.ceil(min_shift / disp))]

            if (ind[1]-ind[0]) % 2 == 1:
                ind[1] -= 1

            if ind[1] == 0:
                ind[1] = None

            subE = self.multispectrum.energy_axis[ind[0]: ind[1]]
            new_offset = subE[0]
            new_size = subE.size

            delta_offset = self.multispectrum.offset - new_offset

            ms = MultiSpectrumshape(self.multispectrum.dispersion, new_offset,
                                    new_size, shape[0], shape[1])
            self.aligned = MultiSpectrum(ms, data=np.zeros((shape[0], shape[1],
                                                            new_size)))

            # make new multispectra with the new size of the cropped spectra
            for index, spec in enumerate(self.other_spectra):
                prev_off = spec.offset
                ms = MultiSpectrumshape(self.multispectrum.dispersion,
                                        prev_off-delta_offset, new_size,
                                        shape[0], shape[1])
                nshape = (shape[0], shape[1], new_size)
                self.aligned_others.append(MultiSpectrum(ms, np.zeros(nshape)))

        else:
            self.aligned = self.multispectrum.copy()
            for spectra in self.other_spectra:
                self.aligned_others.append(spectra.copy())
            ind = [0, None]

        for index in tqdm(np.ndindex(shape)):
            islice = np.s_[index]
            dat = np.pad(self.multispectrum.multidata[islice],
                         pad_width=(len(ar_before), len(ar_after)))
            si = interpolate.interp1d(E_ax, dat)
            i_dat = si(self.multispectrum.energy_axis+self.shift[islice])
            self.aligned.multidata[islice] = i_dat[ind[0]:ind[1]]

            # apply the shift to every other spectrum
            for ii, spec in enumerate(self.other_spectra):
                dat = np.pad(spec.multidata[islice],
                             pad_width=(len(ar_before), len(ar_after)))
                si = interpolate.interp1d(E_ax, dat)
                i_dat = si(self.multispectrum.energy_axis+self.shift[islice])
                self.aligned_others[ii].multidata[islice] \
                    = i_dat[ind[0]:ind[1]]

    def show_signal_range(self, index=(0, 0)):
        """
        Show the signal range used in the determination of the shift.

        Parameters
        ----------
        index : tuple
            The spectrum of the multispectrum which will be shown.
            (default: (0,0))

        Returns
        -------
        fig: Figure
            The figure which can then be used to modify or save
        """
        ind0 = self.multispectrum.get_energy_index(self.signal_range[0])
        ind1 = self.multispectrum.get_energy_index(self.signal_range[1])
        boolean = np.zeros(self.multispectrum.size)
        boolean[ind0:ind1] = True
        self.multispectrum.setcurrentspectrum(index)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(self.multispectrum.energy_axis, self.multispectrum.data,
                color='black')
        ax.fill_between(self.multispectrum.energy_axis, 0,
                        self.multispectrum.data.max(), where=boolean,
                        color='green', alpha=0.5)
        ax.set_xlabel(r'Energy loss [eV]')
        return fig

    def show_shift(self, show_index=False, nbins=20):
        """
        Shows the shift which will be applied to the spectra.

        Returns
        -------
        fig: Figure
            The figure which can then be used to modify or save
        """

        if self.multispectrum.xsize == 1 or self.multispectrum.ysize == 1:
            fig, ax = plt.subplots()
            if show_index:
                ax.plot(np.squeeze(self.index_shift))
                ax.set_ylabel(r'Shift [pixel]')
            else:
                ax.plot(np.squeeze(self.shift))
                ax.set_ylabel(r'Shift [eV]')

        else:
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            if show_index:
                data = self.index_shift - self.index_shift[0, 0]
                label = 'index'
                bins = np.arange(data.min(), data.max()+1)

            else:
                data = self.shift
                label = 'eV'
                bins = np.linspace(data.min(), data.max(), nbins)

            im = ax[0].imshow(data)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(label, rotation=270)

            h = np.histogram(data.flatten(), bins=bins)
            ax[1].bar(h[1][:-1], h[0], width=(bins[1]-bins[0]))
            ax[1].set_xlabel('Applied shift [eV]')

        return fig

    def show_alignment_result(self):
        """
        Figure outputs the average spectrum before and after alignment

        Returns
        -------
        fig: Figure
            The figure which can then be used to modify or save
        """
        fig, ax1 = plt.subplots(figsize=(6, 3))
        avg_r = self.multispectrum.mean().data
        avg_a = self.aligned.mean().data
        en_r = self.multispectrum.energy_axis[np.argmax(avg_r)]
        en_a = self.multispectrum.energy_axis[np.argmax(avg_a)]
        intv = 10
        ax1.plot(self.multispectrum.energy_axis, avg_r, label='Raw')
        ax1.plot(self.aligned.energy_axis, avg_a, label='Aligned')
        ax1.legend()
        xlim = [min(en_r, en_a)-intv, max(en_r, en_a)+intv]
        ax1.set_xlim(xlim)

        return fig

    def make_pdf_result(self):
        """
        Creates a pdf which summarizes the results obtained from the
        alignment procedure. This is helpfull to identify a problem
        in this procedure since it is not possible for each multispectrum
        which is acquired.

        Returns
        -------
        fig: Figure
            The figure which can then be used to modify or save
        """
        pdf = PDF(orientation='P', unit='in', format='A4')
        pdf.add_page()

        pdf.intitialize(path=self.pdf_path)
        pdf.write(pdf.dh, txt='Method used is: ' + self.method + '\n')
        pdf.write(0.5, txt=' \n')

        # figure in the resulting shift
        pdf.write(pdf.dh, txt='Cropping was set to: '
                              + str(self.cropping) + '\n')

        dE = np.diff(self.multispectrum.energy_axis[:2])[0]
        pdf.write(pdf.dh, txt='Energy range of raw data is ' + str(dE)
                              + ' eV \n')
        dE = self.aligned.energy_axis[-1] - self.aligned.energy_axis[0]
        pdf.write(pdf.dh, txt='Energy range of the aligned data is ' + str(dE)
                              + ' eV \n')
        pdf.write(0.5, txt=' \n')

        pdf.write(pdf.dh, txt='The shift which is applied to each '
                              'individual spectrum')

        fig = self.show_shift(show_index=False, nbins=20)
        pdf.add_figure(fig)

        # figure on the signal range used
        pdf.add_page()
        pdf.write(pdf.dh, txt='The signal range used in determining the '
                              'shift  \n')
        pdf.write(pdf.dh, txt='Signal range is: '+str(self.signal_range))
        fig1 = self.show_signal_range()
        pdf.add_figure(fig1)

        # figure showing the result
        pdf.add_page()
        pdf.write(pdf.dh, txt='The average raw and aligned spectrum \n')
        fig2 = self.show_alignment_result()
        pdf.add_figure(fig2)

        plt.close('all')
        name = pdf.get_savename()
        pdf.output(name)
