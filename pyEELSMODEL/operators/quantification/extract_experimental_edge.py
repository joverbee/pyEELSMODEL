from pyEELSMODEL.core.operator import Operator
from pyEELSMODEL.operators.areaselection import AreaSelection
from pyEELSMODEL.operators.quantification.elemental_quantification import \
    ElementalQuantification
from pyEELSMODEL.components.gdoslin import GDOSLin
from pyEELSMODEL.components.fixedpattern import FixedPattern
import matplotlib.pyplot as plt


class ExperimentalEdgeExtractor(Operator):
    """
    Extract experimental edge from an EELS map which can be used to
    do oxidation state mapping or other things.

    """
    def __init__(self, multispectrum, settings, ll):
        """
        Initialize a ExperimentalEdgeExractor object.
        Parameters
        ----------
        multispectrum: MultiSpectrum
            The multispectrum from which the reference edges will be extracted
        settings: tuple
            Tuple containing the acceleration voltage [V], convergence angle
             [rad] and collection angle [rad]
        ll: MultiSpectrum
            The low loss multispectrum which is acquired from the same region.
            This is needed to incorporate the effect of multiple scattering.
        """

        self.multispectrum = multispectrum
        self.ll = ll
        self.areas = []
        self.settings = settings

    def define_new_region(self, input_map=None, max_points=4, coords=None):
        """
        Define an AreaSelection object. This object is able to extract the
        average spectrum from a given area which can be selected via the
        openend image. This is done by right clicking on the image.
        One could also provide coords which then selects the area from
        the given coordinates.

        Parameters
        ----------
        input_map: numpy 2d array
            The image which is shown. Could be the ADF image acquired simul-
            taneously. If None, the image shows the total intensity of the
            detector at each probe position. (default: None)
        max_points: uint
            The number of points used to draw the area.
        coords: list
            A list with x and y-coordinates of the points. For instance for
            a rectangle with starting position (1,2), width=5 and height=7
            coords = [[1,1,6,6],[2,9,9,2]].

        """
        if coords is None:
            area = AreaSelection(self.multispectrum, input_map=input_map,
                                 other_spectra=[self.ll],
                                 max_points=max_points)
            area.determine_input_area()
        else:
            area = AreaSelection(self.multispectrum, input_map=input_map,
                                 other_spectra=[self.ll],
                                 max_points=max_points)
            area.xcoords = coords[0]
            area.ycoords = coords[1]

        self.areas.append(area)

    def show_regions(self):
        """
        Shows the selected areas together with the index. This index is
        usefull for the self.extract_edge() function.

        Returns
        -------
        fig: Figure
            The create figure
        """
        fig, ax = plt.subplots()
        ax.imshow(self.areas[0].input_map, cmap='gray')
        for ii, area in enumerate(self.areas):
            ax.fill(area.xcoords, area.ycoords, alpha=0.8)
            ax.scatter(area.xcoords, area.ycoords, color='black')
            ax.text(area.xcoords[0], area.ycoords[0], 'Area: '+str(ii),
                    fontsize=16, color='red')
        return fig

    def calculate_spectra(self):
        """
        Calculates the average spectra from the areas which have been
        selected. These are stored in the self.spectra and self.llspectra
        attributes.
        """
        self.spectra = []
        self.llspectra = []
        for area in self.areas:
            s = area.get_mean_from_area()
            self.spectra.append(s)
            self.llspectra.append(area.other_avg_spec[0])

    def show_average_spectra(self):
        """
        Shows the average spectra from the different areas.

        Returns
        -------
        fig: Figure
            The create figure
        """

        fig, ax = plt.subplots()
        for ii, spec in enumerate(self.spectra):
            ax.plot(spec.energy_axis, spec.data, label='Area: '+str(ii))

    def add_quantification_method(self,  hl, ll, elements, edges, intervals,
                                  pre_fine):
        """
        Creates a quantification object. This could be used to modify some
        settings to optimize the fit itself.

        Parameters
        ----------
        hl: Spectrum
            The spectrum from which to get the reference edge
        ll: Spectrum
            The low loss spectrum which is from the same part as hl.
        elements: list
            List of elements which are present in the spectrum hl
        edges: list
            List of edge of the elements. The length of this list should be
            equal to the elements list.
        intervals: list
            List of energy intervals [eV] over which to fit the fine structure.
        pre_fine: list
            List of energies onset energy with respect to the onset energy
            of the atomic cross section. Hence a positive value indicates
            that the fine structure starts before expected onset from the
            atomic cross section. Negative values would be unexpected.
        """
        quant = ElementalQuantification(hl, elements, edges, self.settings,
                                        ll=ll)
        quant.linear_fitter_method = 'ols'
        quant.n_bgterms = 4
        quant.pre_fine = pre_fine
        quant.use_fine = True
        quant.fine_intervals = intervals
        quant.do_align = False
        self.quant = quant

    def extract_edge(self, index, elements, edges, intervals, pre_fine):
        """
        Fits the average spectrum indicated by index using the elements and
        edges. The interval indicates the energy region over which to fit
        the fine structure. Pre_fine is used to modify the start position
        of the fine structure with respect to the atomic cross section. The
        sampling of the fine structure is determined by the fwhm of the zlp.

        Parameters
        ----------
        index: int
            The index of the chosen area. The first area selected has index 0
            and so forth. The show_regions() function also shows which area
            corresponds to which index.
        elements: list
            List of elements which are present in the spectrum of index
        edges: list
            List of edge of the elements. The length of this list should be
            equal to the elements list.
        intervals: list
            List of energy intervals [eV] over which to fit the fine structure.
        pre_fine: list
            List of energies onset energy with respect to the onset energy
            of the atomic cross section. Hence a positive value indicates
            that the fine structure starts before expected onset from the
            atomic cross section. Negative values would be unexpected.
        """
        hl = self.spectra[index]
        ll = self.llspectra[index]

        self.add_quantification_method(hl, ll, elements, edges, intervals,
                                       pre_fine)
        self.quant.do_procedure()

        self.quant.fitter.plot()

        fixs = []
        for ii in range(len(elements)):
            fix = self._get_norm_cross_section(ii)
            fixs.append(fix)

        return fixs

    def _get_norm_cross_section(self, index):
        """
        Calulates the normalized cross section from the fit. It puts the
        atomic cross sections amplitude A=1 and scales the fine structure
        with this. The new reference edge is the sum of atomic and fine
        structure and is a FixedPattern object.

        Parameters
        ----------
        index: uint
            The index from the elements in the fitter. This is needed to
            extract the atomic cross section and fine structure for an edge.

        Returns
        -------
        fix: FixedPattern
            The new edge which is the sum of atomic cross section and fine
            structure.

        """

        component = self.quant.element_components[index]

        fines = []
        for comp in self.quant.fitter.model.components:
            check_gdos = isinstance(comp, GDOSLin)

            if check_gdos:
                print(comp.connected_edge)
                check_connected_edge = comp.connected_edge is component
                print(check_connected_edge)
                if check_connected_edge:
                    print('here')
                    fines.append(comp)

        new_comp = component.copy()
        for fine in fines:
            new_comp.data += fine.data

        new_comp.data = new_comp.data / component.parameters[0].getvalue()
        fix = FixedPattern(new_comp.get_spectrumshape(), new_comp,
                           name=component.name)
        fix.calculate()
        return fix
