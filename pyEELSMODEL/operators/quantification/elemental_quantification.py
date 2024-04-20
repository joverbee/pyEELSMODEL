import numpy as np
import matplotlib.pyplot as plt
import logging
from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.core.model import Model

from pyEELSMODEL.operators.aligns.fastalignzeroloss import FastAlignZeroLoss

from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.linear_background import LinearBG
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import\
    ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.CLedge.kohl_coreloss_edgecombined import \
    KohlLossEdgeCombined
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.components.gdoslin import GDOSLin

from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.fitters.lsqfitter import LSQFitter

logger = logging.getLogger(__name__)


class ElementalQuantification(Operator):
    """
    Object which has the workflow to perform elemental quantification.
    Many different attributes can be easily modified to have some flexibility
    in the workflow.


    """
    def __init__(self, spectrum, elements, edges, settings, ll=None):
        """
        Makes an ElementalQuantification object.

        Parameters
        ----------
        spectrum: Spectrum or MultiSpectrum
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
        self.spectrum = spectrum
        self.is_multispectrum = isinstance(spectrum, MultiSpectrum)
        self.ll = ll

        self.do_align = True

        # show feedback
        self.feedback = True  # inidicates if inbetween results are shown

        # attributes connected to core loss components
        self.elements = elements
        self.edges = edges
        # onset is defined with respect to the energy of the edge
        self.onsets = np.zeros((len(elements)))

        self.E0 = settings[0]
        self.alpha = settings[1]
        self.beta = settings[2]
        self.qsteps = 20  # the number of points used to integrate q vector

        # attributes connected to fine structure
        self.use_fine = False  # indicates if fine structure will be used.
        self.fine_components = []
        self.fine_intervals = []
        # how much the fine structure should be fitter before onset energy
        self.pre_fine = None
        self.dE = None

        # attributes connected to background
        self.background_model = 'linear'
        self.n_bgterms = 4  # number of terms in the linear background

        # attributes connected to fitter
        self.linear_fitter_method = 'nnls'
        self.use_weights = False

        # which GOS array to use: zhang or segger
        self.gos_array = 'zhang'

        if ll is not None:
            self.estimate_sampling()

    def align_zlp(self):
        """
        Align the multispectra using the low loss. If no multispectrum is
        used, then this method will not do anything.
        """
        if self.ll is None:
            print('No low loss is added to the object so zlp alignement is '
                  'impossible')
            return None

        if not self.is_multispectrum:
            print('zlp alignment is only performed on multispectra')
            return None

        align = FastAlignZeroLoss(self.ll, other_spectra=[self.spectrum])
        align.perform_alignment()
        # align.show_shift()

        self.align = align
        self.spectrum = align.aligned_others[0]
        self.ll = align.aligned

    def _get_onset_energies(self):
        """
        Calculates the onset energies of the different edges. Is needed in
        the estimation of the autofit when using the power-law as background
        """
        onset_energies = np.zeros(len(self.element_components))
        for ii, comp in enumerate(self.element_components):
            onset_energies[ii] = comp.onset_energy
        return onset_energies

    def estimate_autofit_powerlaw(self):
        """
        Estimates the indices used to get information on the starting position
        of the power law. This is a region before the first edge in the
        spectrum.
        """
        onset_energies = self._get_onset_energies()
        first_onset = np.min(onset_energies)
        ind_f = self.spectrum.get_energy_index(first_onset)
        ind_i = 5  # the 10 pixel is used as estimate

        if ind_f <= ind_i:
            print('powerlaw estimation region too small')
            return None
        indices = [ind_i, ind_f]
        return indices

    def visualize_autofit(self, bg):
        """
        Visualizes the region used for the estimation. Helpful for debugging
        """

        indices = self.estimate_autofit_powerlaw()

        if isinstance(self.spectrum, MultiSpectrum):
            s = self.spectrum.mean()
        else:
            s = self.spectrum

        xx = np.arange(bg.size)
        where = (xx >= indices[0]) & (xx <= indices[1])
        fig, ax = plt.subplots()
        ax.plot(s.energy_axis, s.data)
        ax.plot(bg.energy_axis, bg.data)
        ax.fill_between(bg.energy_axis, 0, s.data.max(),
                        where=where, color='green', alpha=0.5)

    def make_background(self):
        """
        Makes the background component.
        """
        spsh = self.spectrum.get_spectrumshape()
        if self.background_model == 'powerlaw':
            bg = PowerLaw(spsh, A=10, r=3)
            indices = self.estimate_autofit_powerlaw()

            if isinstance(self.spectrum, MultiSpectrum):
                s = self.spectrum.mean()
            else:
                s = self.spectrum

            bg.autofit(s, indices[0], indices[1])

            if self.feedback:
                self.visualize_autofit(bg)

        elif self.background_model == 'linear':
            rlist = np.linspace(1, 5, self.n_bgterms)
            bg = LinearBG(specshape=spsh, rlist=rlist)
        self.bg = bg

    def make_coreloss(self):
        """
        Makes the atomic cross sections components. It uses Zezhongs calculated
        GOS array to compute the cross sections.

        """
        spsh = self.spectrum.get_spectrumshape()

        # The components of the edges
        comp_elements = []
        for elem, edge, onset in zip(self.elements, self.edges, self.onsets):
            if self.gos_array == 'zhang':
                comp = ZezhongCoreLossEdgeCombined(spsh, 1, self.E0,
                                                   self.alpha, self.beta,
                                                   elem, edge, eshift=onset,
                                                   q_steps=self.qsteps)
            elif self.gos_array == 'kohl':
                comp = KohlLossEdgeCombined(spsh, 1, self.E0, self.alpha,
                                            self.beta, elem, edge,
                                            eshift=onset, q_steps=self.qsteps)

            comp_elements.append(comp)
        self.element_components = comp_elements

    def estimate_sampling(self):
        if self.is_multispectrum:
            self.dE = np.sqrt(2)*self.ll.mean().get_numerical_fwhm()
        else:
            self.dE = np.sqrt(2)*self.ll.get_numerical_fwhm()

    def make_finestructure(self):
        """
        Creates fine structure components for edge core loss edge. It uses
        the fine_intervals attribute to know the interval region of the
        fine structure. The sampling is determined by the fwhm of the zlp.
        The self.pre_fine is used to indicate how much before the edge
        onset the fine structure should start.
        """
        spsh = self.spectrum.get_spectrumshape()

        # if no dE is given then estimate it from zero loss peak
        if self.dE is None:
            self.estimate_sampling()

        print(self.dE)
        for ii, comp in enumerate(self.element_components):
            n = int(self.fine_intervals[ii]/self.dE)
            print(n)
            if self.pre_fine is None:
                pre_fine = 0
            else:
                pre_fine = self.pre_fine[ii]
            fine = GDOSLin.gdoslin_from_edge(spsh, comp,
                                             ewidth=self.fine_intervals[ii],
                                             degree=n,
                                             interpolationtype='cubic',
                                             pre_e=pre_fine)

            self.fine_components.append(fine)

    def make_mscatter(self):
        """
        Creates the multiple scattering component. It uses the lowloss
        for this.
        """

        spsh = self.spectrum.get_spectrumshape()

        if self.ll is None:
            self.llcomp = None
        else:
            self.llcomp = MscatterFFT(spsh, self.ll)

    def make_model(self):
        """
        Creates the model from the available components.
        """
        spsh = self.spectrum.get_spectrumshape()

        self.make_coreloss()
        self.make_background()
        self.make_mscatter()

        components = [self.bg] + self.element_components
        if self.use_fine:
            self.make_finestructure()
            components.extend(self.fine_components)
        if self.llcomp is not None:
            components.append(self.llcomp)

        self.model = Model(spsh, components)

    def make_fitter(self):
        """
        Makes the fitter object. If the self.background_model is linear then
        a linear fitter is used. Else a non-linear least squares fitter
        will be used.

        """
        if self.background_model == 'linear':
            self.fitter = LinearFitter(self.spectrum, self.model,
                                       method=self.linear_fitter_method,
                                       use_weights=self.use_weights)

        else:
            for comps in self.element_components:
                comps.parameters[0].setboundaries(0, 1e10)
            self.fitter = LSQFitter(self.spectrum, self.model, method='trf',
                                    use_bounds=True)

    def do_procedure(self):
        """
        Workflow to process core loss results. First it aligns the
        multispectrum using the low loss. Then the model is calculated and
        the fitter object is created and the fitting procedure begins.

        """

        if self.do_align:
            self.align_zlp()
        self.make_model()
        self.make_fitter()

        if isinstance(self.spectrum, MultiSpectrum):
            self.fitter.multi_fit()
            self.multimodel = self.fitter.model_to_multispectrum()
            fig, maps, names = \
                self.fitter.show_map_result(self.element_components)
            self.elemental_maps = maps
            self.elemental_names = names
        else:
            self.fitter.perform_fit()

    def show_elements_maps(self):
        """
        Shows the elemental maps.
        """
        fig, maps, names = self.fitter.show_map_result(self.element_components)

    def get_elemental_maps(self):
        maps, names = self.fitter.get_map_results(self.element_components)
        self.elemental_maps = maps
        return maps, names

    def get_multimodels(self):
        """
        Calculates the different multimodels of the different components.
        The first multimodel is contains the fitted background. Then it adds
        the next edge until all edges are added. Hence the last multimodel in
        the list is the resulting fit. The inbetween results help interpreting
        the fitted cross sections.

        Returns
        -------
        multimodels: list of MultiSpectrum
            A list constaining multimodels to visualize how the resulting fit
            A list constaining multimodels to visualize how the resulting fit
            looks like.

        """
        multimodels = []
        multimodels.append(
            self.fitter.model_to_multispectrum_with_comps([self.bg]))

        onset_energies = self._get_onset_energies()
        co = np.argsort(onset_energies)

        use_comps = [self.bg]
        for ii in range(co.size):
            comp = self.element_components[co[ii]]
            comps = use_comps + [comp]
            s = self.fitter.model_to_multispectrum_with_comps(comps)
            multimodels.append(s)
            use_comps.append(comp)
            if self.use_fine:
                comp = self.fine_components[co[ii]]
                comps = use_comps + [comp]
                s = self.fitter.model_to_multispectrum_with_comps(comps)
                multimodels.append(s)
                use_comps.append(comp)

        return multimodels

    def get_CRLB(self):
        """
        Returns the CRLB for the different components of the coreloss
        edges.
        """
        if self.is_multispectrum:
            crlbs = np.zeros((len(self.element_components),
                              self.spectrum.xsize, self.spectrum.ysize))

        else:
            crlbs = np.zeros(len(self.element_components))

        for ii, comp in enumerate(self.element_components):
            if self.is_multispectrum:
                crlbs[ii] = self.fitter.CRLB_map(comp.parameters[0])
            else:
                crlbs[ii] = self.fitter.CRLB(comp.parameters[0])

        self.crlbs = crlbs
        return crlbs
