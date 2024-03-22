import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.patches import Rectangle
from pyEELSMODEL.core.operator import Operator

logger = logging.getLogger(__name__)


class MultiSpectrumVisualizer(Operator):

    """
    Class which visualizes the multspectrum.
    """
    def __init__(self, multispectra, input_map=None, labels=None,
                 logscale=False):
        """
        Class which visualizes the multspectrum.
        Parameters
        ----------
        multispectra: list of MultiSpectrums
            A list of multispectrum
        input_map: 2d numpy array
            An 2d array which can be used to visualize the regions. For
            instance the ADF acquired simultaneously can be used. If None,
            then the sum of the spectrum at each probe positions will be
            used.
        labels: List of strings
            A label to indicate which spectrum is visualized with which color
        logscale: bool
            Indicates if the y axis should be visualized as a log scale
            (default: False)

        """
        self.multispectra = multispectra
        self.logscale = logscale
        self.is_line = False
        if self.multispectra[0].xsize == 1 or self.multispectra[0].ysize == 1:
            self.is_line = True
            input_map = np.squeeze(self.multispectra[0].multidata)
            self.w = self.multispectra[0].size

        if input_map is None:
            self.input_map = self.multispectra[0].multidata.sum(2)
        else:
            self.input_map = input_map

        self.sx = 0
        self.sy = 0
        self.h = 1
        self.w = 1
        self.linewidth = 5
        self.title = None

        # if we have a line scan, we auto scale the imshow
        if 1 in self.input_map.shape:
            self.aspect = 'auto'
        else:
            self.aspect = None
        self.press = None

        self.xlim_eels = None
        if labels is None:
            self.labels = [None]*len(self.multispectra)
        else:
            self.labels = labels

        self.fig, self.ax = plt.subplots(1, 2)
        self.plot()
        self.connect()

    def get_indextitle(self, x0, y0, w, h):
        title = 'x0, y0: ' + str(int(x0)) + ', '+str(int(y0)) + '; w, h: ' \
                + str(int(h)) + ', '+str(int(w))
        return title

    def plotrect(self):
        # plot selection rectangle on top of the image plot
        ax = self.ax
        if self.is_line:
            ax[0].axis("tight")  # fill window space
            xmin, xmax = ax[0].get_xlim()
            self.w = self.multispectra[0].size
            self.rect = Rectangle((self.sx - 0.5, self.sy - 0.5),
                                  self.w, self.h, fill=False, edgecolor='red',
                                  linewidth=self.linewidth)
        else:
            self.rect = Rectangle((self.sx - 0.5,  self.sy - 0.5),
                                  self.w, self.h, fill=False, edgecolor='red',
                                  linewidth=self.linewidth)

            ax[0].axis("image")  # square pixels

        ax[0].axis('off')
        ax[0].add_patch(self.rect)

    def updaterect(self):
        # print(self.get_indextitle(self.sx, self.sy, self.w, self.h))
        self.rect.set_xy((self.sx-0.5, self.sy - 0.5))
        self.rect.set_width(self.w)
        self.rect.set_height(self.h)
        self.rect.figure.canvas.draw()

    def plot(self, **kwargs):
        ax = self.ax
        ax[0].imshow(self.input_map, aspect=self.aspect)
        self.plotrect()
        # print(self.w)
        # print(self.h)
        self.plotline = []

        for spectra, label in zip(self.multispectra, self.labels):
            y = [self.sx, self.sx+self.h]
            x = [self.sy, self.sy+self.w]
            mydata = spectra.multidata[x[0]:x[1], y[0]:y[1], :].mean((0, 1))
            plotline = ax[1].plot(spectra.energy_axis, mydata, label=label)
            # print('initialize')
            self.plotline.append(plotline)
        ax[1].legend()
        ax[1].set_title(self.get_indextitle(self.sy, self.sx, self.h, self.w))
        self.xlim = ax[0].get_xlim()
        self.ylim = ax[0].get_ylim()

        if self.logscale:
            ax[1].set_ylim([1, None])
            ax[1].set_yscale('log')

        for key, value in kwargs.items():
            if key == 'xlim':
                ax[1].set_xlim(value)
                self.xlim_eels = value

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

        self.keypress = self.rect.figure.canvas.mpl_connect('key_press_event',
                                                            self.on_key_press)

    def on_key_press(self, event):
        dx = 1
        dy = 1
        dw = 1
        dh = 1
        # print('key pressed')

        x0 = self.sx
        y0 = self.sy
        w = self.w
        h = self.h
        if event.key == 'up':
            self.sy = int(max(0, y0 - dy))

        if event.key == 'down':
            self.sy = int(min(self.ylim[0] + 0.5 - h, y0 + dy))

        if event.key == 'left' and not self.is_line:
            self.sx = int(max(0, x0 - dx))

        if event.key == 'right' and not self.is_line:
            self.sx = int(min(self.xlim[1] + 0.5 - w, x0 + dx))

        if event.key == '+':
            bolx = (x0 + w + dw >= self.xlim[1])
            boly = (y0 + h + dh >= self.ylim[0])

            if bolx and not boly and self.is_line is False:
                self.h = h + dh

            # if not(bolx) and self.is_line==False:
            elif boly and not bolx:
                self.w = w + dw

            elif bolx and boly:
                print('Cannot make square larger')

            else:
                self.w = w + dw
                self.h = h + dh

        if event.key == '-':
            self.w = max(1, w - dw)
            self.h = max(1, h - dh)

        # self.rect.set_x(self.sx)
        # self.rect.set_y(self.sy)

        self.update_eels()

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.rect.axes:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        # print('event contains', self.rect.xy)
        self.press = self.rect.xy, (event.xdata, event.ydata)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None or event.inaxes != self.rect.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        # print(f'x0={x0}, xpress={xpress}, event.xdata={event.xdata}, '
        #       f'dx={dx}, x0+dx={x0+dx}')
        shiftx = int(max(0,
                         min(self.xlim[1] - self.rect.get_width(), x0 + dx)))
        shifty = int(max(0,
                         min(self.ylim[0] - self.rect.get_height(), y0 + dy)))

        self.sx = shiftx
        self.sy = shifty
        self.update_eels()
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None
        self.rect.figure.canvas.draw()

    def update_eels(self):
        self.updaterect()

        ax = self.ax
        # self.rect.figure.axes[1].cla()
        # y0 = max(0, y0 + 0.5)
        # x0 = max(0, x0 + 0.5)
        #
        # print(self.sx)
        # print(self.sy)
        # print(self.h)
        # print(self.w)
        miny = 0
        maxy = 0
        for plotline, spectra, label in zip(self.plotline,
                                            self.multispectra, self.labels):
            y = [self.sx, self.sx+self.w]
            x = [self.sy, self.sy+self.h]
            mydata = spectra.multidata[x[0]:x[1], y[0]:y[1], :].mean((0, 1))
            plotline[0].set_ydata(mydata)
            miny = min(mydata.min(), miny)
            maxy = max(mydata.max(), maxy)
            # print('hello')
        ax[1].set_title(self.get_indextitle(self.sy, self.sx, self.w, self.h))

        ax[1].set_ylim([miny, maxy])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
