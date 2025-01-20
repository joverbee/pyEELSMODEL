from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np


class AfterGlow(Component):
    """
    Component which can add a afterglow if it is observed in the experimental
    data.
    Simple approximation which uses the same zlp for each spectrum whereas
    drift could make this different. Additionally, the position of the after
    glow needs to be indicated which would not be needed

    #todo find a generic way to handle the afterglow with any need of Epos

    """

    def __init__(self, specshape, zlp, Epos, A=1):
        """
        Initiates the Afterglow component

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        zlp : numpy array
            The zlp array which is best taken to be the average.
        Epos: energy position of the peak of the afterglow
        """
        super().__init__(specshape)
        p1 = Parameter('A', A)
        p1.setlinear(True)
        p1.setboundaries(0, 1e20)
        p1.sethasgradient(True)
        self._addparameter(p1)

        self.setdisplayname('Afterglow')
        self.Epos = Epos
        self.zlp = zlp
        self.set_fixeddata(self.zlp, self.Epos)
        # don't convolute the background it only gives problems
        # and adds no extra physics
        self._setcanconvolute(False)

        self.calculate()

    def set_fixeddata(self, zlp, Epos):
        """
        Sets the component using the zlp data and the energy where it is seen.

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        zlp : numpy array
            The zlp array which is best taken to be the average.
        Epos: energy position of the peak of the afterglow

        """
        index = self.get_energy_index(Epos)

        shift = np.argmax(zlp) - index

        zlpdata = np.roll(zlp, -shift)[:self.size]
        pad_size = self.size - zlpdata.size
        print(pad_size)

        if pad_size > 0:
            zlpdata = np.pad(zlpdata, (0, pad_size))
            print(zlpdata.size)

        self.fixeddata = zlpdata

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        p1 = self.parameters[0]

        if p1.ischanged():
            A = p1.getvalue()
            self.data = A * self.fixeddata
        self.setunchanged()  # put parameters to unchanged
