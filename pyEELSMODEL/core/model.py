"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import numpy as np
import matplotlib.pyplot as plt

from pyEELSMODEL.core.spectrum import Spectrum
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.components.MScatter.mscatter import Mscatter
import logging

logger = logging.getLogger(__name__)


class Model(Spectrum):
    """
    Model class
    A Model is a Spectrum that can calculate itself based on a number of
    Components which each depend on a number of Parameters.
    Note that all actions like convolving with low loss, adding or multiplying
    stuff is encoded in the Components, some of which are special types which
    tells the model how to treat them to come up with a single model result.
    Philosophy of Model: a model with a number of parameters mimicking an EELS
    experiment. Everything that happens in experiments needs to be modelled
    somehow by the components in the model. The model doesnt care whether the
    spectrum it simulates is a single spectrum or an SI. It is the task of the
    fitter to deal with the parameters for each position in an SI
    (different from cpp EELSMODEL!)
    """

    def __init__(self, spectrumshape, components=None):

        """
        Instantiate a Model object.

        Parameters
        ----------
        spectrumshape : Spectrumshape
            Shape of the spectrum the model should mimick (see Spectrum
            documentation).
        components : list of Components, optional
            List of Components the model is made of, allows to load the model
            from a file. The default is None in which case the model will be
            empty to start.

        Returns
        -------
        None.

        """
        super().__init__(spectrumshape)
        if components is None:
            self.components = []
        else:
            self.components = components

        self.allparameters = []
        self.freeparameters = []
        self.freelinparameter = []
        self.freenonlinparameters = []
        self.locked = False
        self.changed = True
        self.prepare()

    def release(self):
        for comp in self.components:
            comp.release()

    def lock(self, b):
        """
        Set the locked state of the model. If the model is locked, no
        components can be added or removed. This is useful during fitting
        where the model structure should not change.

        Parameters
        ----------
        b : bool
            True=locked, False is unlocked.

        Returns
        -------
        None.

        """
        self.locked = b

    def islocked(self):
        return self.locked

    def setchanged(self, b):
        self.changed = b

    def ischanged(self):
        return self.changed

    def addcomponent(self, component):
        if self.islocked():
            # don't do it when the model is locked, the fitter typically locks
            # the model during fitting
            return
        if isinstance(component, Component):
            # give every new component the same eshift as we have
            # component.seteshift(self.geteshift())
            component.eshift = self.eshift
            self.components.append(component)
            self.setchanged(True)
            self.prepare()
            self.calculate()

        else:
            raise TypeError(r'Input should be Component object')

    def removecomponent(self, component):
        """
        Removes a component from the model.

        Parameters
        ----------
        component : Component
            The component which needs to be removed.

        Returns
        -------
        None.

        """
        if self.islocked():
            return
        if isinstance(component, Component):
            self.components.remove(component)
            self.setchanged(True)
            self.prepare()
            self.calculate()
            return True
        else:
            raise TypeError(r'Input should be Component object')

    def getcomponents(self):
        return self.components

    def getallparameters(self):
        return self.allparameters

    def getfreeparameters(self):
        return self.freeparameters

    def getfreelinparameters(self):
        return self.freelinparameter

    def getfreenonlinparameters(self):
        return self.freenonlinparameters

    def prepare(self):
        """
        Reallocate space to hold parameters. Needs to be called every time the
        number of components or state of parameters (changeable or linear)
        changes. The parameterlists are what is handled by the fitter. So good
        time to call this function is right before a fitting loop to make sure
        we are working with an up to date status of all parameters. Changing
        parameter status (changeble or linear) during a fit should not be
        allowed in the UI. Calling this everytime before calculate would be
        too time consuming.

        Returns
        -------
        None.

        """
        # empty storage space
        self.allparameters = []
        self.freeparameters = []
        self.freelinparameter = []
        self.freenonlinparameters = []
        # and fill it up again
        for comp in self.components:
            for p in comp.parameters:
                self.allparameters.append(p)
                if p.ischangeable():
                    self.freeparameters.append(p)
                    if p.islinear():
                        self.freelinparameter.append(p)
                    else:
                        self.freenonlinparameters.append(p)

    def getcomponentbyparameter(self, parameter):
        """
        Get component that belongs to a given parameter reference.

        Parameters
        ----------
        parameter : Parameter
            Reference to a parameter.

        Returns
        -------
        comp : Component
            reference to a component that has this parameter reference.
            None if parameter not found in any component in the model.

        """
        for comp in self.components:
            if parameter in comp.parameters:
                return comp
        return None

    def islinear(self):
        """
        Returns whether all free parameters in the model are linear. If this
        is the case we have a full linear model which allows the use of a
        linear fitter.

        Returns
        -------
        boolean : bool
            True if linear model, False otherwise.

        """
        boolean = True
        for comp in self.components:
            for param in comp.parameters:
                if param.ischangeable():
                    boolean = boolean and param.islinear()
                    if not param.islinear():
                        logger.warning('non linear parameter found %s in '
                                       'component:%s', param.getname(),
                                       comp.getname())
        return boolean

    def getnumcomponents(self):
        """
        Return the number of components in the model.

        Returns
        -------
        int
            Number of components in the model.

        """
        self.prepare()  # make sure to return up to date information
        return len(self.components)

    def getnumparameters(self):
        """
        Get total number of parameters in the model. This includes the free
        and the non-free parameters.

        Returns
        -------
        n_param : int
            Number of parameters in the model.

        """
        self.prepare()  # make sure to return up to date information
        return len(self.allparameters)

    def getnumfreeparameters(self):
        """
        Get total number of free parameters in the model.

        Returns
        -------
        n_free : int
            Number of free parameters in the model.

        """
        self.prepare()  # make sure to return up to date information
        return len(self.freeparameters)

    def saveparams(self, filehandle, fmt='txt', header=True):
        """
        Store all current parameters in a file that is already open
        this allows to append the parameters of all estimated params in a
        spectrum image in one big file.
        TODO work eg with elementtree for XML writing

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.
        fmt : TYPE, optional
            DESCRIPTION. The default is 'txt'.

        Returns
        -------
        None.

        """

        # in first line of text file should appear the names
        # if header:
        #     for p in allparameters:
        #         print(p.getdisplayname(),'\tCRLB\t',end='')
        #     print('')
        # #decide on text format style, tab separated is easy
        # for p in allparameters:
        #     print(p.getvalue(),'\t',p.getsigma,'\t',end='')
        print('')
        # replace with file writing
        # and more compact w binary

    def savemodel(self):
        print('saves the model')

    def order_coupled_components(self):
        """
        The components which have parameters which are coupled to other
        parameters should be first calculated in the model. Therefore the
        component list should be ordered such that first the coupled parameters
        are calculated and then the rest.
        """
        worker = []
        main = []

        for comp in self.components:
            b = False
            for param in comp.parameters:
                if param.iscoupled():
                    cm = self.getcomponentbyparameter(param.coupled_parameter)
                    if cm is not comp:
                        b = True

            if b:
                worker.append(comp)
            else:
                main.append(comp)

        components = worker + main
        self.components = components

    def calculate(self, use_ll=True):
        """
        Calculate the model by calculating all components.
        For the linear least squares fitting it would be interesting to not use
        the low loss for computing the model since we can do the low loss
        convolution at a later step to improve speed for the linear fitting.

        Parameters
        ----------
        use_ll: boolean
            Indicates if it needs to use the low loss for calculating the
            model.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.erase()  # start with a clean slate

        # apply shift components if present they come first since they change
        # the energy scale for all the rest
        for comp in self.components:
            if comp.getshifter():
                comp.calculate()

        # prepare multiplier components if present
        for comp in self.components:
            if comp.get_ismultiplier():
                comp.calculate()

        # calculate each component and add up
        # first all components which need to be convoluted

        # rearrange the order of the components such that the coupled ones
        # are calculated first
        self.order_coupled_components()

        for comp in self.components:
            if not comp._isconvolutor:
                if comp.getcanconvolute() and (not (comp.get_ismultiplier())
                                               and not (comp.getshifter())):

                    # only those components that can get convoluted and are
                    # not shifters or multipliers
                    comp.calculate()
                    if comp.gethasmultiplier():  # if it has a multiplier
                        multi = comp.getmultiplierptr()

                        if multi is not None:
                            dummyspec = comp.copy()
                            # multiply the result with the multiplier component
                            dummyspec *= multi
                            # add the multiply result to what we have already
                            self += dummyspec

                    else:
                        self += comp  # just add the component

        # then apply the convolution if needed
        if use_ll:
            for comp in self.components:
                if isinstance(comp, Mscatter):
                    comp.data = np.ndarray.copy(
                        self.data)  # copy in component the model data so far
                    comp.calculate()  # convolve

                    self.data = np.ndarray.copy(
                        comp.data)  # and copy it back in the model

        # then add components which can not be convoluted and are not
        # convolutors and are not shifters and are not multipliers
        for comp in self.components:
            if not (comp.getcanconvolute()) and not (comp.get_ismultiplier()) \
                    and not isinstance(comp, Mscatter) \
                    and not (comp.getshifter()):
                comp.calculate()
                self += comp

    def getconvolutor(self):
        if not self.hasconvolutor():
            return None

        for comp in self.components:
            if isinstance(comp, Mscatter):
                return comp

    def resetsigmas(self):
        """
        Reset the error bars on all parameters.

        Returns
        -------
        None.

        """
        for comp in self.components:
            for p in comp.parameters:
                p.setsigma(0.0)

    def plot(self, spectrum=None, externalplt=None, **kwargs):
        """
        Plot the Model

        Parameters
        ----------
        spectrum: Spectrum
            A spectrum can be plotted simultaneously
        externalplt : matplotlib reference
              A reference to an external matplotlib reference, if None we use
              our own matplotlib and create a new figure.
        """
        self.calculate()
        tempplt = plt
        if isinstance(externalplt, plt.Figure):
            tempplt = externalplt
        else:
            # create our own figure
            plt.figure()
            plt.title('Model')
        # show components if visible
        for comp in self.components:
            tempplt.plot(self.energy_axis, comp.data, **kwargs)
        tempplt.plot(self.energy_axis, self.data,
                     **kwargs)  # and the total spectrum
        tempplt.xlabel(r'Energy Loss [eV]')
        if spectrum is not None:
            tempplt.plot(spectrum.energy_axis, spectrum.data, color='black',
                         label='Experimental data')
        tempplt.ylabel('Counts')
        tempplt.legend()

    def componentshow(self):
        """
        show contents of all component in a separate graph if they are visible
        """
        for comp in self.components:
            comp.show()

    def printcomponents(self):
        """
        Print a list summarising all components.

        Returns
        -------
        None.

        """
        for comp in self.components:
            print('name: ', comp.getname())

    def __iadd__(self, spectrum):
        """
        += operator, only add the spectral data.

        Parameters
        ----------
        spectrum : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.data += spectrum.data
        return self

    def __imul__(self, spectrum):
        """
        *= operator, only multiply the spectral data

        Parameters
        ----------
        spectrum : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.data *= spectrum.data
        return self

    def seteshift(self, energy):
        # override of Spectrum::seteshift
        # set eshift for the model
        # super().seteshift(energy)
        super().eshift = energy
        # do the same for all components, they also get a setchanged command so
        # they will recalculate completely when calculate is called
        for comp in self.components:
            # comp.seteshift(energy)
            comp.eshift = energy
        # every new component that gets added needs also to get the same eshift
        # this is done in addcomponent}

    def hasconvolutor(self):
        for comp in self.components:
            if comp._isconvolutor:
                return True
        return False

    def hasshifter(self):
        for comp in self.components:
            if comp._isshifter:
                return True

        return False

    def hasmultiplier(self):
        for comp in self.components:
            if comp._ismultiplier:
                return True

        return False

    def getgradient(self, parameter):
        # ask for an analytical gradient if available
        if parameter.gethasgradient():
            # analytical gradients don't work if the model contains convolutor
            # shifters or multipliers
            # we could apply these to the individual analytical gradients but
            # this takes almost the same time
            # as a numerical derivative

            # find the component
            component = self.getcomponentbyparameter(parameter)
            if component is None:
                return None

            if self.hasmultiplier() or self.hasconvolutor() \
                    or self.hasshifter():
                return None

            # TODO get gradient from a parameter,
            #  create it in component for convenience
            return component.getgradient(parameter)
        else:
            return None
