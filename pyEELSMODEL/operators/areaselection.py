import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib.backend_bases import MouseButton
import logging
from pyEELSMODEL.core.operator import Operator

logger = logging.getLogger(__name__)


class AreaSelection(Operator):

    """
    Class which uses user input to return an average area of a multispectrum.
    This only will not work for a line scan since chosing an area out of this
    is a bit easier.
    """
    def __init__(self, multispectrum, input_map=None, max_points=4,
                 other_spectra=[]):
        """
        Parameters
        ----------
        spectrum: Multispectrum
            The multispectrum form which an area need to be selected
        input_map: 2d numpy array
            The image used to select the proper region. This could be the HAADF
            signal or other calculated images. It is necessary that the scan
            size is the same as for the multispectrum. If input_map is None,
            the summed signal from the MultiSpectrum is used. (default: None)
        max_points: uint > 1
            The number of points used to determine the area. (default: 4)
        """
        self.multispectrum = multispectrum
        self.input_map = input_map
        self.max_points = max_points
        self.other_spectra = other_spectra
        self.avg_spectrum = None
        self.xcoords = None
        self.ycoords = None
        self.other_avg_spec = []

    @property
    def input_map(self):
        return self._input_map

    @input_map.setter
    def input_map(self, input_map):
        if input_map is None:
            im = self.multispectrum.integrate((0, -1), index_type=True)
            self._input_map = im
        else:
            self._input_map = input_map

    @property
    def max_points(self):
        return self._max_points

    @max_points.setter
    def max_points(self, max_points):
        if max_points < 2:
            raise ValueError(r'The maximum point should be larger than 1.')
        self._max_points = max_points

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def calculate_mask_from_coords(self):
        """
        Calculates the mask used from the points selected.
        This mask can be accessed via the attribute self.mask.

        """
        marray = np.array([self.xcoords, self.ycoords]).transpose()
        poly_path = mplPath.Path(marray)

        shape = (self.multispectrum.xsize, self.multispectrum.ysize)
        mask = np.zeros(shape, dtype=bool)
        X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        for index in (np.ndindex(shape)):
            islice = np.s_[index]
            point = (X[islice], Y[islice])
            mask[islice] = poly_path.contains_point(point)
        self.mask = mask

    def get_mean_from_area(self):
        """
        Uses the user input area to calculate the average EEL spectrum.
        This function uses the self.xcoords and self.ycoords to calculate
        the area. Therefore, it is necessary to have some values for these
        attributes. User input on these coordinates can be obtained via the
        self.determine_input_area() function. The average spectra of the
        self.other_spectra attribute can be accessed via the
        self.other_avg_spec attribute.

        Returns
        -------
        s: Spectrum
            The average spectrum from the area.


        """

        self.calculate_mask_from_coords()

        ndata = self.multispectrum.multidata[self.mask, :].mean((0))
        # also the exclude and all these things are carried over.
        s = self.multispectrum.sum().copy()
        s.data = ndata
        self.avg_spectrum = s

        for spec in self.other_spectra:
            ndata = spec.multidata[self.mask, :].mean((0))
            sn = spec.sum().copy()
            sn.data = ndata
            self.other_avg_spec.append(sn)

        return s

    def show_area(self, **kwargs):
        """
        Shows the area which is selected.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.input_map, **kwargs)
        ax.fill(self.xcoords, self.ycoords, alpha=0.5, color='green')

    def determine_input_area(self):
        """
        Shows an image (or plot)  which is given by input map. By right
        clicking on the image, an area can be selected where the number of
        corners is defined by the max_points parameter.
        This function sets the proper values for self.xcoords and self.ycoords.

        """

        def onclick(event):
            if event.button is MouseButton.LEFT:
                pass
            else:
                global ix, iy
                ix, iy = event.xdata, event.ydata
                xcoords.append(ix)
                ycoords.append(iy)

                ax.scatter(event.xdata, event.ydata, color='black')
                if len(xcoords) > 1:
                    ax.plot([event.xdata, xcoords[-2]],
                            [event.ydata, ycoords[-2]], color='black')
                if len(ycoords) == self.max_points:
                    ax.plot([event.xdata, xcoords[0]],
                            [event.ydata, ycoords[0]], color='black')
                    ax.fill(xcoords, ycoords, alpha=0.5, color='black')
                    print('the shape is drawn, this will be used to determine '
                          'the average spectrum')
                    self.xcoords = xcoords
                    self.ycoords = ycoords

                fig.canvas.draw()

                if len(xcoords) == self.max_points:
                    fig.canvas.mpl_disconnect(cid)

                return xcoords

        fig, ax = plt.subplots()
        ax.imshow(self.input_map)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

        global xcoords, ycoords
        xcoords = []
        ycoords = []
        # self.xcoords = np.array(xcoords)
        # self.ycoords = np.array(ycoords)
        self.xcoords = xcoords
        self.ycoords = ycoords

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
