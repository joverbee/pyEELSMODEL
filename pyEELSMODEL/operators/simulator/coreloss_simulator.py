import numpy as np
import logging
from tqdm import tqdm
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model

from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import \
    ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.plasmon import Plasmon
from pyEELSMODEL.components.fixedpattern import FixedPattern
from pyEELSMODEL.core.multispectrum import MultiSpectrumshape, MultiSpectrum

logger = logging.getLogger(__name__)


class CoreLossSimulator(Operator):
    """
    Coreloss EELS spectrum simulator for maps. This uses the atomic cross
    sections for the edge shapes. The simulated low loss is consists out of the
    zer-loss peak and the bulk plasmons. Note that the low loss is only used
    to modify the shape of the edges and the background. Hence the real shape
    is not so important. The background in the core-loss is found
    by convolving a powerlaw with the background to include multiple
    scattering.

    """
    def __init__(self, multispecshape, elements, edges, maps, tlambda_map,
                 settings):
        """

        Parameters
        ----------
        multispecshape: MultiSpectrumShape
            The multispectrum form which an area need to be selected
        elements: list of strings
            The elements list should contain the elements as indicated on the
            periodic table
        edges: list of strings
            The edges list should contain the letters of the edges
            (K, L, M, ...)
        settings: tuple
            Tuple containing E0 (acceleration voltage), alpha (convergence
             angle) and beta (collection angle).
        ll: Spectrum or MultiSpectrum
            The low loss which can be added. This will be used to align the
            spectra with each other and will be used to model the multiple
            scattering.
        """
        self.spectrumshape = multispecshape.spectrumshape
        self.scan_size = (multispecshape.xsize, multispecshape.ysize)
        self.size = multispecshape.Esize
        self.ll = None

        self.elements = elements
        self.edges = edges
        self.onsets = np.zeros((len(elements)))
        self.maps = maps  # contains number of elements 2d maps

        self.E0 = settings[0]
        self.alpha = settings[1]
        self.beta = settings[2]
        self.qsteps = 20  # the number of points used to integrate q vector

        # low loss settings
        self.tlambdas = tlambda_map
        # energy of bulk plasmon at each probe position
        self.Ep_map = 22*np.ones_like(self.tlambdas)
        # width of the bulk plasmon at each probe position
        self.Ew_map = 5*np.ones_like(self.tlambdas)
        self.n_plasmon = 5  # number of plasmon included
        self.fwhm = 1  # eV
        self.llspectrumshape = Spectrumshape(multispecshape.dispersion,
                                             -self.fwhm*10,
                                             multispecshape.Esize)

        # background settings
        self.A_bg = 1e3  # amplitude background
        self.r_bg = 3  # powerlow
        self.bg_method = 'convolved'  # indicates if background needs to be
        # convolved with the low loss. This gives a more accurate description
        # of reality.
        # to validate the CRLB, it is usefull to know the exact shape of the
        # background so this convolution can be turned off.
        self.noise = 1  # noise level

        # shift settings
        self.use_shift = False

        # detector + noise
        self.add_poisson = True  # indicates if poisson noise needs to be added

    def make_lowloss(self):
        """
        Creates the low loss multispectrum using the information on the zero-
        loss peak (fwhm) and bulk plasmon (Ep, tlambda, Ew, beta).

        """
        zlp = Gaussian(self.llspectrumshape, A=1, centre=0, fwhm=self.fwhm)
        zlp.calculate()

        shape = self.scan_size + (self.spectrumshape.size,)
        lldata = np.zeros(shape)
        for index in tqdm(np.ndindex(self.scan_size)):
            islice = np.s_[index]
            A = zlp.data.sum()
            plasmon = Plasmon(self.llspectrumshape, A=A,
                              Ep=self.Ep_map[islice], Ew=self.Ew_map[islice],
                              n=self.n_plasmon,
                              tlambda=self.tlambdas[islice], beta=self.beta,
                              E0=self.E0)
            plasmon.calculate()
            lldata[islice] = zlp.data + plasmon.data
            # lldata[islice] = zlp.data
        llsh = MultiSpectrumshape(self.llspectrumshape.dispersion,
                                  self.llspectrumshape.offset,
                                  self.llspectrumshape.size,
                                  self.scan_size[0],
                                  self.scan_size[1])
        ll = MultiSpectrum(llsh, data=lldata)

        self.ll = ll

    def _calculate_convoluted_background(self, llcomp):
        """
        Calculates a background which is convoluted with the low loss.
        This represents the difference in background when thickness
        changes for the same core loss spectrum.

        Returns
        -------
        bg : FixedPattern
            Background from which only the amplitude can be modified.
            This background is also set to not convolute when applied
            into a model.

        """
        ndispersion = self.spectrumshape.dispersion
        noffset = max(self.spectrumshape.offset -
                      0.5 * self.spectrumshape.size
                      * self.spectrumshape.dispersion,
                      self.spectrumshape.dispersion)

        # nspecshape = Spectrumshape(ndispersion,noffset,2*specshape.size)
        nspecshape = Spectrumshape(ndispersion, 100,
                                   2 * self.spectrumshape.size)

        bkg = PowerLaw(nspecshape, self.A_bg, self.r_bg)
        bkg.canconvolute = True
        bkg.calculate()

        energy_axis = np.arange(nspecshape.size) * ndispersion + noffset
        bkgs = Spectrum.from_numpy(bkg.data, energy_axis)
        mspecshape = bkgs.get_spectrumshape()

        llspec = Spectrum.from_numpy(llcomp.data, llcomp.energy_axis)
        llspec.dispersion = mspecshape.dispersion
        llcompg = MscatterFFT(mspecshape, llspec)

        mod = Model(mspecshape, [bkg, llcompg])

        mod.calculate()

        bg = FixedPattern(self.spectrumshape, mod)
        # print(bg.parameters[0].getvalue())
        bg.calculate()
        # bg.plot()
        bg.parameters[0].setvalue(self.A_bg / bg.data[0])
        bg.calculate()
        bg.canconvolute = False
        return bg

    def get_background(self, ll):
        """
        Return the background dependent on the bg_method attribute.

        Parameters
        ----------
        ll: MultiSpectrum
            The low loss multispectrum.

        """
        if self.bg_method == 'convolved':
            bg = self._calculate_convoluted_background(ll)
        else:
            bg = PowerLaw(self.spectrumshape, self.A_bg, self.r_bg)
        return bg

    def make_elements(self):
        """
        Makes a list containing the atomic core-loss edges.

        """
        comp_elements = []
        for elem, edge, onset in zip(self.elements, self.edges, self.onsets):
            comp = ZezhongCoreLossEdgeCombined(self.spectrumshape, 1, self.E0,
                                               self.alpha,
                                               self.beta, elem, edge,
                                               eshift=onset,
                                               q_steps=self.qsteps)

            comp_elements.append(comp)
        self.element_components = comp_elements

    def make_coreloss(self):
        """
        Calculated the core-loss multispectrum.

        """
        shape = self.scan_size + (self.spectrumshape.size,)
        cldata = np.zeros(shape)

        for index in tqdm(np.ndindex(self.scan_size)):
            islice = np.s_[index]
            llcomp = MscatterFFT(self.spectrumshape, self.ll[islice])
            bg = self.get_background(self.ll[islice])
            for jj, comps in enumerate(self.element_components):
                comps.parameters[0].setvalue(self.maps[jj][islice])

            self.mod = Model(self.spectrumshape,
                             components=[bg]+self.element_components+[llcomp])
            self.mod.calculate()
            if self.add_poisson:
                cldata[islice] = np.random.poisson(self.noise * self.mod.data)
            else:
                cldata[islice] = self.noise * self.mod.data

        s = MultiSpectrum.from_numpy(cldata, self.mod.energy_axis)
        self.multispectrum = s

    def add_shift(self):
        """
        Shifts both low loss and core loss with a sine function.
        The amplitude is 0.5% of the energy range and the period is chosen such
        that 20 periods are in the STEM-EELS map.
        """
        Ashift = 0.005*(self.ll.energy_axis[-1] - self.ll.energy_axis[0])
        period = (self.ll.xsize * self.ll.ysize)/20
        x = np.arange(self.ll.xsize*self.ll.ysize)
        shift1d = Ashift*np.sin(2*np.pi*x/period)
        self.shift2d = np.reshape(shift1d,
                                  (self.ll.xsize, self.ll.ysize)).astype('int')

        cldata = np.zeros_like(self.multispectrum.multidata)
        lldata = np.zeros_like(self.ll.multidata)

        for index in tqdm(np.ndindex(self.scan_size)):
            islice = np.s_[index]
            cldata[islice] = np.roll(self.multispectrum.multidata[islice],
                                     self.shift2d[islice])
            lldata[islice] = np.roll(self.ll.multidata[islice],
                                     self.shift2d[islice])

        cldata = cldata[:, :, self.shift2d.max():-self.shift2d.max()]
        lldata = lldata[:, :, self.shift2d.max():-self.shift2d.max()]

        cl_offset = self.multispectrum.offset +\
            self.shift2d.min()*self.multispectrum.dispersion
        cl_energy = np.arange(cldata.shape[2]) * \
            self.multispectrum.dispersion + cl_offset

        ll_offset = self.ll.offset + self.shift2d.min() * self.ll.dispersion
        ll_energy = np.arange(cldata.shape[2]) * self.ll.dispersion + ll_offset

        self.multispectrum = MultiSpectrum.from_numpy(cldata, cl_energy)
        self.ll = MultiSpectrum.from_numpy(lldata, ll_energy)

    def simulate_multispectrum(self):
        self.make_lowloss()
        self.make_elements()
        self.make_coreloss()

        if self.use_shift:
            self.add_shift()

        print('Multispectrum is simulated')
