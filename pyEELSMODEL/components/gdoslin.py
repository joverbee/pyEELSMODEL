from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
import numpy as np
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


class GDOSLin(Component):
    """
    Linear fine structure DOS Component
    This component will create in the onset region [estart-estart+ewidth] of a
    core loss a interpolated curve going through degree points which are free
    parameters of the model
    This curve is then added to the core loss cross section and will increase
    or decrease certain areas in the cross section
    As this is added, this is also a linear component

    """

    def __init__(self, specshape, estart, ewidth=50, degree=20,
                 interpolationtype='linear'):
        """
        Initializes the gdoslin object.

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model

        estart: float [eV]
            The onset value of the fine structure. This is mostly
            the onset energy of the edge on wich you want to apply the
            fine structure.

        ewidth : float [eV]
            The energy width over which the fine structure is modelled.
            (default: 50)

        degree: int
            The number of parameters used to model the fine structure.
            (default: 20)

        interpolationtype: string
            The type of interpolation used.

        Returns
        -------

        """
        super().__init__(specshape)
        self.setdescription(
            "Linear Fine Structure used to alter atomic cross sections"
            " in the near edge region")

        self.degree = degree
        self.interpolationtype = interpolationtype

        p0 = Parameter("Estart", estart, False)
        self._addparameter(p0)

        p1 = Parameter("Ewidth", ewidth, False)
        self._addparameter(p1)

        # and a list of variables controlling the shape of the fine struct
        for i in range(self.degree):
            pname = 'a' + str(i)
            p = Parameter(pname, 1.0, True)
            p.setboundaries(-np.inf, np.inf)
            p.setlinear(True)
            self._addparameter(p)

        self.ctes = np.ones(self.degree + 2)
        self.initDOS()

        self.connected_edge = None
        self.set_gdos_name()

        # normalization such that each fine structure has the same integral
        self.calculate_integral_per_parameter()

    @classmethod
    def gdoslin_from_edge(cls, specshape, component, pre_e=5, ewidth=50,
                          degree=20, interpolationtype='linear'):
        """
        Class method is made to create an gdoslin which is connected to a
        coreloss edge.
        No need to input the onset energy

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model

        component: CoreLossEdge
            The coreloss edge on which to add the fine structure.

        pre_e: float [eV]
            The value of the energy onset of the fine structure with respect
            to the onset energy of the coreloss edge. Hence for oxygen at 532eV
            with pre_e=5, the starting energy for the fine structure is 527 eV.
            (default: 5)

        ewidth : float [eV]
            The energy width over which the fine structure is modelled.
            (default: 50)

        degree: int
            The number of parameters used to model the fine structure.
            (default: 20)

        interpolationtype: string
            The type of interpolation used.

        Returns
        -------

        """

        if not isinstance(component, CoreLossEdge):
            raise TypeError(r'Component should be a CoreLossEdge')

        estart = component.onset_energy - pre_e
        comp = GDOSLin(specshape, estart, ewidth=ewidth, degree=degree,
                       interpolationtype=interpolationtype)
        comp.connected_edge = component
        # print(comp.connected_edge)
        comp.set_gdos_name()
        return comp

    def set_gdos_name(self):
        """
        Set the proper name of the gdos
        """
        s1 = "Linear Fine Structure (DOS):"
        s2 = 'Onset' + str(self.parameters[0].getvalue()) + 'eV'
        s3 = 'Width' + str(self.parameters[1].getvalue()) + 'eV: '
        if self.connected_edge is None:
            self._setname(' '.join([s1, s2, s3]))
        else:
            self._setname(' '.join([s1, s2, s3, self.connected_edge.name[:2]]))

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p0 = self.parameters[0]
        p1 = self.parameters[1]
        Estart = p0.getvalue()
        Ewidth = p1.getvalue()
        Estop = Estart + Ewidth

        # calculate the DOS
        changes = False

        for i in range(self.degree):
            p = self.parameters[i + 2]
            changes = changes or p.ischanged()

        if p0.ischanged() or changes or p1.ischanged():
            logger.debug("parameters changed calculating DOS. degree: %d",
                         self.degree)
            logger.debug("Estart: %e", Estart)
            logger.debug("Estop: %e", Estop)

            en = self.energy_axis

            self.initDOS()

            yp = self.yp / self.ctes
            # yp = self.yp

            f = interp1d(self.Ep, yp, kind=self.interpolationtype)
            self.data[self.indexstart:self.indexstop] = f(
                en[self.indexstart:self.indexstop])

            # set parameters as unchanged since last time we calculated
            self.setunchanged()

        else:
            logger.debug(
                "parameters have not changed, i don't need to calculate again")

    def initDOS(self):
        # prepare all for the DOS to work with the given options
        p0 = self.parameters[0]
        p1 = self.parameters[1]
        Estart = p0.getvalue()
        Ewidth = p1.getvalue()
        Estop = Estart + Ewidth
        en = self.energy_axis
        self.indexstart = self.get_energy_index(Estart)
        if self.get_energy_index(Estop) == self.size:
            self.indexstop = self.get_energy_index(Estop) - 1
        else:
            self.indexstop = self.get_energy_index(Estop)
        if self.indexstop <= self.indexstart:
            logger.warning('Estart and Estop are too close together')
            return

        self.Ep = []
        self.yp = []
        Estep = (en[self.indexstop] - en[self.indexstart]) / (self.degree + 1)
        # start with yp=0 at estart
        self.Ep.append(en[self.indexstart])
        self.yp.append(0)
        for i in range(self.degree):
            self.Ep.append(en[self.indexstart] + (i + 1) * Estep)
            self.yp.append(self.parameters[i + 2].getvalue())
        self.Ep.append(en[self.indexstop])
        self.yp.append(0)
        self.yp = np.array(self.yp)

    def couple_parameter_to_edge(self):
        """
        Couples the parameters of the gdoslin to the edge such that this
        template edge can be used to fit the spectrum where the gdoslin is no
        variable anymore.
        """
        if self.connected_edge is None:
            print('no edge is connected to this gdoslin')
            return

        for i in range(self.degree):
            p = self.parameters[i + 2]
            fraction = p.getvalue() / self.connected_edge.parameters[
                0].getvalue()
            p.couple(self.connected_edge.parameters[0], fraction=fraction)

    def calculate_integral_per_parameter(self):
        """
        Since the integral for each individual parameter is not the same we
        calculate the difference which is then applied to the component
        as a multiplication factor to resolve this interpolation problem
        """
        ctes = np.ones(len(self.parameters))

        for ii, param in enumerate(self.parameters[2:]):
            for par in self.parameters[2:]:
                if par == param:
                    par.setvalue(1)
                else:
                    par.setvalue(0)
            self.calculate()

            # ctes[ii+1] = self
            ctes[ii + 1] = self.data.sum()
            ctes[ii + 1] /= (self.Ep[ii + 2] - self.Ep[ii]) / (
                        self.Ep[2] - self.Ep[0])

        self.ctes = ctes
        # print('done')

        for param in self.parameters[2:]:
            param.setvalue(1)

        self.calculate()
