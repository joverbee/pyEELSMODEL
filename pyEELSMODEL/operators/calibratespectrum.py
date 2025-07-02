# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import logging

from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge

logger = logging.getLogger(__name__)


class CalibrateSpectrum(Operator):
    """
    The CalibrateSpectrum class provides a workflow to change the energy axis
    which is sometimes wrong. It does this by performing a linear fit between
    the measured onset energies (defined by user) and values obtained from
    literature.
    """

    def __init__(self, spectrum, element_list, edge_list, other_spectra=[]):
        """
        Creates CalibrateSpectrum object.

        Parameters
        ----------
        spectrum: Spectrum
            The spectrum used for the calibration
        element_list: List of str
            List which contains the different elemtens seen on the EEL spectrum
        edge_list: List of str
            List which contains the different edges seen on the EEL spectrum
        other_spectra: List of Spectra of MultiSpectra
            List on which the same type of correction should be applied

        """
        self.spectrum = spectrum.copy()
        self.element_list = element_list
        self.edge_list = edge_list

        spectra = []
        for other in other_spectra:
            spectra.append(other.copy())

        self.other_spectra = spectra

        self.set_onset_edge_energies()

    def set_onset_edge_energies(self):
        """
        Calculates the list of tabulated onset energies of the edges.
        """
        self.onset_edge_energies = []

        sh = self.spectrum.get_spectrumshape()
        for elem, edge in zip(self.element_list, self.edge_list):
            core = CoreLossEdge(sh, 1, 1, 1, 1, elem, edge)
            self.onset_edge_energies.append(core.onset_energy)

    def show_real_energies(self):
        """
        Show the real energies from the tabulated data on the spectrum
        """

        cte = 1.1
        fig, ax = plt.subplots()
        if isinstance(self.spectrum, MultiSpectrum):
            ndata = self.spectrum.sum().data
        else:
            ndata = self.spectrum.data

        ax.plot(self.spectrum.energy_axis, ndata)
        for index, en in enumerate(self.onset_edge_energies):
            lab = self.element_list[index]+' '+self.edge_list[index]+' edge'
            ax.plot([en, en], [ndata.min(), cte*ndata.max()], label=lab)

        ax.set_ylim([ndata.min(), cte*ndata.max()])
        ax.set_xlabel(r'Energy axis')
        ax.legend()

    def determine_edge_positions(self):
        """
        Use user input by using right mouse click to select the edge energy.
        After the click is performed, it is shown which edge you had selected
        and you can verify if this is correct.

        :return:
        """

        def onclick(event):
            if event.button is MouseButton.LEFT:
                pass
            else:
                global ix, iy
                ix, iy = event.xdata, event.ydata
                ylim = ax.get_ylim()
                ax.plot([event.xdata, event.xdata], [ylim[0], ylim[1]],
                        color='black')
                ax.set_ylim(ylim)
                fig.canvas.draw()
                print('Energy of ' + self.element_list[len(coords)] + ' ' +
                      self.edge_list[len(coords)] + ' is:' + str(ix))

                coords.append(ix)
                if len(coords) == len(self.onset_edge_energies):
                    fig.canvas.mpl_disconnect(cid)

                return coords

        print('right click the edge')
        cte = 1.1
        fig, ax = plt.subplots()
        if isinstance(self.spectrum, MultiSpectrum):
            ndata = self.spectrum.sum().data
        else:
            ndata = self.spectrum.data
        ax.plot(self.spectrum.energy_axis, ndata)
        ax.set_ylim([ndata.min(), cte*ndata.max()])
        ax.set_xlabel(r'Energy axis')
        global coords
        coords = []
        self.coords = coords
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    def calibrate_energy_axis(self):
        """
        Calibrates the energy axis using the input energies.

        """
        if len(self.coords) != len(self.onset_edge_energies):
            print('Run the determine_edge_positions function '
                  'again to define the edges')

        self.coeff = np.polyfit(np.array(self.coords),
                                np.array(self.onset_edge_energies), 1)
        n_en_axis = self.coeff[0] * self.spectrum.energy_axis + self.coeff[1]
        disp = n_en_axis[1]-n_en_axis[0]
        offset = n_en_axis[0]
        delta_offset = self.spectrum.offset - offset

        self.spectrum.dispersion = disp
        self.spectrum.offset = offset

        for spectrum in self.other_spectra:
            spectrum.dispersion = disp
            spectrum.offset -= delta_offset

    def show_calibration_fit(self):
        plt.figure()
        plt.plot(self.coords, self.onset_edge_energies,
                 marker='o', label='Data points')
        plt.plot(self.spectrum.energy_axis,
                 self.coeff[0] * self.spectrum.energy_axis + self.coeff[1],
                 label='Fit')
        plt.legend()
