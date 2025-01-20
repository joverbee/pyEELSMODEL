"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.spectrum import Spectrum
from pyEELSMODEL.core.parameter import Parameter
import logging

logger = logging.getLogger(__name__)


class Component(Spectrum):
    """
    A Component is a Spectrum that can calculate itself from a number of
    parameters. Base class from which all components can be derived with
    specific functionality

    """

    def __init__(self, spectrumshape, params=None):
        """
        Instantiate a Component object

        Parameters
        ----------
        spectrumshape : Spectrumshape
            Describing the size of the spectrum this component will generate,
            see info on Spectrum.
        params : list of Parameter, optional
            A list of parameters controlling this component
            This is typically passed when loading a previously made component
            from a file. If a new component is constructed, typically the
            component will create and take care of its own parameters. The
            default is None.

        Returns
        -------
        None.

        """
        super().__init__(spectrumshape)
        self.parameters = []
        self.gradient = []
        self._setparameters(
            params)  # use preloaded parameters if given in params
        self.spectrumshape = spectrumshape

        
        self._isshifter = False # indicates a special type of component that can shift the energies
        self._multiplier = None
        self._ismultiplier = False
        self._hasmultiplier = False
        self._isconvolutor = False
        self.displayname = ""
        self.description = ""
        self._name = ""
        self.visible = True
        self.suppress = False

        self._setshifter(False)
        self._setcanconvolute(True)
        self._set_ismultiplier(False)
        self.setvisible(False)
        self.setdisplayname("")
        self.setdescription("unitialized component")
        self._setname("unitialized component")

    def _addparameter(self, parameter):
        """
        Add parameter to a component and indicate if the component
        provides an analytical gradient for this parameter.
        This is an internal function meant to be used only in the init of a
        specific component

        Parameters
        ----------
        parameter : Parameter
            A parameter to add to the component.

        Returns
        -------
        None.

        """
        self.parameters.append(parameter)
        self.gradient.append(Spectrum(self.spectrumshape))

    def release(self):
        # do all you need to do to tell that we are no longer there
        # release all parameters
        # (eg in case they might be coupled to other parameters)

        for p in self.parameters:
            p.release()
    
    def setsuppress(self,b):
        """
        Sets the suppress state of the component. Allows to switch of a component in the sense that calling
        calculate will return a zero spectrum. Use with caution as forgetting to disable this function
        may lead to hard to interpret behaviour

        Parameters
        ----------
        b : bool
            suppress state.

        Returns
        -------
        None.

        """
        self.suppress=b
        self.setchanged()
        #if you have parameters that are coupled, than also that component needs to know that something changed
        for p in self.parameters:
            if p.iscoupled():
                cp=p.getcontroller()
                cp.setchanged()


    def setvisible(self, b):
        """
        Sets the visible state of the component. Can indicate whether in a
        model spectrum showing individual components, this component needs to
        be visible or not. default True.

        Parameters
        ----------
        b : bool
            visible state.

        Returns
        -------
        None.

        """
        self.visible = b
        self.show()  # show it if it was set to true

    def _setname(self, name):
        """
        Sets the name of the component. This name needs to be unique for each
        component as it will be used to load components from file when loading
        a model.

        Parameters
        ----------
        name : string
            The name of the component.

        Returns
        -------
        None.

        """
        self.name = name
        # also give that name to the spectrum that is at the base of the
        # component
        super().setname(name)

    def setdisplayname(self, name):
        """
        Sets the displayed name of the component. This name can differ from the
        actual name of the component and allows the user to change the name to
        a more meaningful name without affecting the official name that should
        not change.

        Parameters
        ----------
        name : string
            Display name for the component.

        Returns
        -------
        None.

        """
        self.displayname = name
        super().setname(name)

    def setdescription(self, description):
        """
        Set the description string of the component. This is typically the
        business of the component itself, but can be changed by the user. The
        description can be used e.g. inthe UI to provide help explaining the
        function of the component

        Parameters
        ----------
        description : string
            A description of what the component does.

        Returns
        -------
        None.

        """
        self.description = description

    def getname(self):
        """
        Get the official and fixed name of a component. This name is a unique
        identifier of the type of component. This name should not be changed
        by the user.

        Returns
        -------
        string
            Unque identifier name of the component.

        """
        return self.name

    def getdisplayname(self):
        """
        Get the friendly name of a component. This name can be changed by the
        user and will typically be used for designating the component in eg. a
        user interface.

        Returns
        -------
        string
            A name describing this component.

        """
        return self.displayname

    def getdescription(self):
        """
        Get a string describing the functionality of the component.
        Can be used in e.g. help or tooltips in the UI.

        Returns
        -------
        string
            String describing the functionality.

        """
        return self.description

    def _setparameters(self, params):
        """
        Set the list of Parameter relevant to this component. Note that
        typically a component has a well defined set of parameters and the
        user should not need to tamper with these. This function is used when
        loading a component from file and sending it a previously defined
        list of parameters.

        Parameters
        ----------
        params : list of Parameter
            list of parameters which needs to be added to the component.

        Returns
        -------
        None.

        """
        if params is None:
            return
        else:
            for p in params:
                self._addparameter(p)  # this also reserves space for gradients

    def print_parameter_values(self):
        """
        Print all parameters of a component as name : value

        Returns
        -------
        None.

        """
        for p in self.parameters:
            print(p.getname(), ' : ', p.getvalue())

    def setunchanged(self):
        """
        Set all parameters of a Component as unchanged. This is typically done
        internally after a component.calculate() but can also be called
        externally to avoid that a component will be recalculated if fiddling
        with some parameters externally. Note however that if you call this
        function, the next time you call recalculate the component will falsely
        assume that it doesnt need to do anything as the parameters are marked
        as unchanged!

        Returns
        -------
        None.

        """
        for p in self.parameters:
            p.setunchanged()

    def setchanged(self):
        """
        Mark all parameters as changed. This notifies the component that on the
        next component.calculate() call, the full component needs to be
        recalculated. Note that it even marks unchangeable Parameters as
        changed even though their value doesnt.

        Returns
        -------
        None.

        """
        for p in self.parameters:
            p.setchanged()

    def ischanged(self):
        changed = False
        for p in self.parameters:
            changed = changed or p.ischanged()
        return changed

    def has_parameter(self, p):
        """
        Search if Parameter p is one of the parameters in the parameter list of
         this component.

        Parameters
        ----------
        p : Parameter
            Reference to input parameter to find in the parameter list of this
            component.

        Raises
        ------
        TypeError
            In case p is not a Parameter object.

        Returns
        -------
        bool
            True if parameter p indeed is one of the parameters of this
            component, False otherwise.

        """
        if isinstance(p, Parameter):
            return p in self.parameters
        else:
            raise TypeError(r'Input should be Parameter object')

    def _setcanconvolute(self, b):
        """
        Set the canconvolute state of the component. Default=True This
        indicates whether this Component should be modified by Convoluter type
        components when calculating the result of a Model. A typical use case
        is for an EELS background component that would not improve when
        convolved with a low loss spectrum while it leads to nasty edge
        artefacts. The solution is to not convolve this component at all which
        can be obtained by setting this setcanconvolute state to False

        Parameters
        ----------
        b : bool
            canconvolute state.

        Returns
        -------
        None.

        """
        self.canconvolute = b

    def getcanconvolute(self):
        """
        Gets wether this component can be convoluted

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        return self.canconvolute

    def _setshifter(self, b):
        """
        Sets whether this component is a Shifter type component (b=True).
        Default is False. A shifter Component can alter the energy values of
        the model to mimick the effect of experimental energy drift or eg.
        nonlinearity of the dispersion. All 'normal' components should not
        alter this. The model will treat shifter components in a special way
        when calculating

        Parameters
        ----------
        b : bool
            isshifter state.

        Returns
        -------
        None.

        """
        self.isshifter = b

    def getshifter(self):
        """
        Get whether this is a shifter component

        Returns
        -------
        bool
            True if shifter type component.

        """
        return self.isshifter

    def calculate(self):
        """
        This calculates the spectral data of the component. Each derived class
        from Component will need to implement its own functionality here.

        Returns
        -------
        None.

        """
        logger.warning(
            'if you see this you have called calculate on a '
            'component that didnt implement calculate')
        # raise NotImplementedError() #this forces a derived class to
        # implement it

    def seteshift(self, eshift):
        """
        Sets an energy shift for this component which shifts its energy values
        with respect to the energy axis given when instantiating the component.
        This can be used to mimick experimental energy drift. In order to make
        sure the component gets a full recalculation at the next calculate()
        call, all parameters will be indicated as changed.

        Parameters
        ----------
        eshift : float
            energy shift.

        Returns
        -------
        None.

        """
        super().seteshift(
            eshift)  # shift the energy of the underlying spectrum
        # make sure to recalculate as the energy values have changed
        self.setchanged()

    def getparameter(self, index):
        """
        Return the parameter with index.

        Parameters
        ----------
        index : int
            Index to indicate which parameter to return
        Raises
        ------
        ValueError
            When index out of bounds.
        TypeError
            When index is not int

        Returns
        -------
        Parameter
            Reference to the requested Parameter in the params list.

        """
        if self.indexOK(index):
            return self.parameters[index]
        else:
            raise IndexError('bad index')
            return None

    def indexOK(self, index):
        """
        Check if index is within range of params list

        Parameters
        ----------
        index : int
            index.

        Returns
        -------
        bool
            True: index is good,, False otherwise.

        """
        return 0 <= index < len(self.parameters)

    def getmultiplier(self):
        """
        Return a reference to a Multiplier Component if one is attached to this
        component Otherwise returns None. Multiplier components can be used
        e.g. to mimick effect of gain variation of a camera.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        multiplier : Component
            Reference to another component which is indicated to be a
            multiplier of this component.

        """
        if not self._hasmultiplier:
            return None
        try:
            self.multiplier.get_ismultiplier()
        except Exception:
            raise AttributeError('Invalid reference to a multiplier component')
            self._hasmultiplier = False
            self.multiplier = None
        return self.multiplier

    def gethasmultiplier(self):
        return self.hasmultiplier

    def sethasmultiplier(self, b):
        self.hasmultiplier = b

    def setmultiplier(self, component):
        """
        Set a multiplier component to this component. The component needs to
        be of the multiplier type. The multiplier component will take the
        output of this component and multiply it with its content

        Parameters
        ----------
        component : Component
            The component which needs to be set as multiplier.
        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        dummy = False
        try:
            dummy = component.get_ismultiplier()
        except Exception:
            self.hasmultiplier = False
            self.multiplier = None
            raise AttributeError('invalid component')
        if dummy:
            self.multiplier = component
            self.hasmultiplier = True
        else:
            self.multiplier = None
            self.hasmultiplier = False

    def _set_ismultiplier(self, b):
        self.ismultiplier = b
        # a multiplier can not have a multiplier itself
        self.hasmultiplier = False
        self.multiplier = None

    def get_ismultiplier(self):
        return self._ismultiplier

    def releasemultiplier(self):
        # tell the component that we no longer have a multiplier
        self.hasmultiplier = False
        self.multiplier = None

    def getfreelinparameters(self):
        freelinparameters = []
        for p in self.parameters:
            if p.ischangeable():
                if p.islinear():
                    freelinparameters.append(p)
        return freelinparameters

    def _pullparameter(self):
        """
        Pull last parameter from parameter list and releases that parameter.
        This effectively reduces the number of parameters of a component
        NOT sure why we would need this???and definitely not a user function

        Returns
        -------
        None.

        """
        # remove last parameter from the parameter list
        try:
            p = self.parameters.pop()
            p.release()
        except Exception:
            return

    def show(self):
        """
        Plot component contents if the component has isvisible==True

        Returns
        -------
        None.

        """
        if self.visible:
            self.plot()

    def getgradient(self, parameter):
        """
        Get a reference to the gradient of this component with respect to
        parameter, if an analytical gradient is available. This function needs
        to be overoaded for every specific component

        Parameters
        ----------
        parameter : Parameter
            Reference to an existing parameter in the component.

        Returns
        -------
        A reference to a Spectrum holding the gradient if the parameter was
        indeed a parameter of this component and has an analytical gradient
        defined gradient. None in all other cases.

        """
        return None

    def save(self, fh):
        # save the details of this component to file in such a way
        # that you can bring it to the same state later

        # save the file location of the ll spectrum to open it later
        # when reloading the model

        # TODO: probably XML with elementtree would be most logica
        # component can override this if they need to store more details

        return

    def load(self, fh):
        # load this component from saved detaile

        # TODO: probably XML with elementtree would be most logica
        # component can override this if they need to store more details
        return
