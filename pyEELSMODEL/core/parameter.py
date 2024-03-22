"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
import sys
import copy
import logging

logger = logging.getLogger(__name__)


class Parameter():
    """
    Parameter class to store a parameter to be used in e.g. a component that
    typically has a list of parameters. Parameters hold a single numerical
    value, can have lower and upper boundaries, can be locked can be coupled to
    other parameters, can have the 'linear' property, indicating that in a
    component this parameter is linear as opposed to a nonlinear parameter that
    affects the component in a nonlinear way. Parameters can also be monitored
    by a Monitor class and they can be watched by a Watcher class
    """

    def __init__(self, name, val, changeallowed=True):
        """
        Instantiates a Parameter
        Parameters
        ----------
        name : string
            A meaningful name for the parameter.
        val : float or int
            The initial value of the parameter.
        changeallowed : bool, optional
            Wether the parameter can be changed after instantiating.
            The default is True.

        Returns
        -------
        None.

        """
        self.setdefaults()
        self.setname(name)
        self.setvalue(val)
        self.setchangeable(changeallowed)
        logger.debug('Parameter init')

    def setdefaults(self):
        """
        Sets default values, typically only called in __init__

        Returns
        -------
        None.

        """
        self.value = 0
        self.changed = True
        self.displayed = False
        self.changeable = True
        self.couple_fraction = 1.0
        self.bound = False
        self.monitored = False
        self.watched = False
        self.linear = False
        self.watcher = None
        self.coupled = False
        self.coupled_parameter = None
        self.monitor = 0
        self.setname("")
        self.sigma = 0.0
        self.setboundaries(-sys.float_info.max, sys.float_info.max)
        self.hasgradient = False
        self.valid = True

    def release(self):
        """
        Call this function whenever you no longer need this parameter
        it warns other coupled Parameter, Monitor and Watcher instances
        that might be linked to it that they should no longer count on this
        parameter.

        Returns
        -------
        None.

        """
        self.valid = False

        self.releasecoupling()
        # self.releasemonitor()
        # self.releasewatcher()

    def sethasgradient(self, b):
        self.hasgradient = b

    def gethasgradient(self):
        return self.hasgradient

    def copy(self):
        """
        Returns a copy of the object
        :return:
        """
        return copy.deepcopy(self)

    def setvalue(self, val):
        """
        Sets the value of the parameter. But only if it is not a coupled slave,
        if this is changeable and if the new value is within the bounds if the
        parameter is bound.

        Parameters
        ----------
        val : float
            New value of the parameter.

        Returns
        -------
        True if value was set.

        """
        # only change when changeable and within boundaries,
        # otherwise do nothing...
        if self.iscoupled():
            logger.info('This parameters cannot be changed since it is '
                        'coupled')
            return False  # this parameter cant'be changed

        # if self.coupled and self.ismaster():
        #     #mark the slave as changed
        #     self.coupled_parameter.setchanged() #important!!!

        okbd = self.boundaryOK(val, self.getlowerbound(), self.getupperbound())
        if self.changeable and okbd:
            self.value = val
            self.changed = True
        if not okbd:
            # try:
            #     v = float(val)
            # except:
            #     logger.warning('invalid float (prob infinity')
            #     return False
            logger.info(
                'Hitting the boundary on parameter %s, ub=%e, lb=%e,'
                ' requested val=%e. either increase the boundary or'
                ' try to find out why the fit is diverging',
                self.getname(), self.getupperbound(),
                self.getlowerbound(), val)
            return False
        return True

    def getvalue(self):
        """
        Gets the current value of the Parameter. If the parameter is a coupled
        slave, the value of the master times a coupling constant will be
        returned

        Returns
        -------
        float
            Value of parameter or of the master parameter times a coupling
             constant if this is a coupled slave.

        """
        if self.iscoupled():
            # get the value of the coupled parameter if the master parameter
            # still exists
            coupledval = self.coupled_parameter.getvalue()
            return coupledval * self.couple_fraction
        # in normal case, just return the value
        else:
            return self.value

    def interactivevalue(self, description):  # interactive setup of the value
        """
        Not implemented.
        """

        # not clear what to do here
        return

    def setdisplayed(self, b):
        """
        Set the displayed state of the parameter. This can be used from a UI
        to indicate if the UI has to be informed if e.g. something changes.

        Parameters
        ----------
        b : bool
            True if displayed on a UI, False otherwise.

        Returns
        -------
        None.

        """
        if b:
            self.displayed = True
        else:
            self.displayed = False

    def isdisplayed(self):
        """
        Return whether the parameter is marked as 'displayed'. This can be set
         with setdisplayed(True/False)

        Returns
        -------
        bool
            True is displayed state is set, False otherwise.

        """
        return self.displayed

    @property
    def sigma(self):
        """
        Get the error bar on this parameter.

        Returns
        -------
        sig : float
            value of the error bar for this parameter.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, sig):
        """
        Sets the error bar on this parameter. This is merely a variable that
        stores any number you put in it, however it is meant as a storage for
        a calculated precision estimate on the parameter.

        Parameters
        ----------
        sig : float
            value of the error bar for this parameter.

        Returns
        -------
        None.
        """
        self._sigma = sig

    def setunchanged(self):
        """
        Sets the parameter to the unchanged state. This can be used by the
        model to signal that the model has already been calculated with these
        parameters and we don't need to repeat the calculation unless someone
        changes the parameter value.
        """
        self.changed = False

    def setchanged(self):
        """
        Force the parameter to a changed state.
        This is useful to e.g. set a coupled slave parameter to the changed
        state even though that parameter is typically unchangeable as it should
        follow the master. This is important as a component needs to know that
        this parameter has changed in order to force a full recalculation when
        recalculate is called. This can also be used to force a recalculate of
        a Component as changed parameters will avoid that the Component takes
        a shortcut because it thinks it already calculated with this parameter
        value.

        Returns
        -------
        None.

        """
        # this is a warning received by a slave of coupling
        # that parameter has changed
        # if self.iscoupled() and self.isslave():

        self.changed = True  # even if a slave is unchangeable...

    def setboundaries(self, lb, ub, force=False):
        """
        Sets numerical boundaries to the parameter but only if the current
        value fits within those boundaries. The boundaries are inclusive >=
        and <=. Swapping lower for upper boundary is allowed and will
        automatically be corrected. If force is True, the value of the
        parameter will be changed to make the boundaries work.

        Parameters
        ----------
        lb : float
            lower boundary.
        ub : float
            upper boundary.
        force: bool
            If force is true, then the boundaries are set even if the value
            of the parameter falls outside this boundary. The new value of the
            parameter will be set in the center of the boundaries.
            (default: True)


        Returns
        -------
        bool
            True if bound, False otherwise.

        """
        # check if value fits in range, if not, don't set the boundaries
        # return false if bounds are not set
        self.bound = True
        if (self.boundaryOK(self.value, lb, ub)):
            self.lowerbound = lb
            self.upperbound = ub
            return True
        if (self.boundaryOK(self.value, ub, lb)):
            self.lowerbound = lb
            self.upperbound = ub
            return True
        if force:
            self.lowerbound = lb
            self.upperbound = ub
            val = (ub + lb) / 2
            self.setvalue(val)

        logger.info(r'Boundaries are not set')
        self.bound = False
        return self.bound

    def setupperbound(self, ub):
        """
        Sets the upper numerical bound on the parameter. No checking is
        performed on whether the current value would violate this bound

        Parameters
        ----------
        ub : float
            upper boundary.

        Returns
        -------
        None.

        """
        self.upperbound = ub

    def setlowerbound(self, lb):
        """
        Sets the lower numerical bound on the parameter. No checking is
        performed on whether the current value would violate this bound

        Parameters
        ----------
        ub : float
            lower boundary.

        Returns
        -------
        None.

        """
        self.lowerbound = lb

    def boundaryOK(self, x, lb, ub):
        """
        Check if a proposed value x is within the bounds given by lower bound
        (lb) and upper bound (ub).

        Parameters
        ----------
        x : float
            value to be tested.
        lb : float
            lower bound.
        ub : float
            upper bound.

        Returns
        -------
        bool
            returns True if x is within bounds OR if the parameter is not
            bound.

        """
        return not (self.bound) or ((x >= lb) and (x <= ub))

    def getupperbound(self):
        """
        Returns the current upper bound.

        Returns
        -------
        float
            upper bound value.

        """
        return self.upperbound

    def getlowerbound(self):
        """
        Returns the current lower bound.

        Returns
        -------
        float
            lower bound value.

        """
        return self.lowerbound

    def setchangeable(self, b):
        """
        Set the changeable state of the parameter. If not changeable, the
        parameter value can not be updated. This can be used to indicate to a
        Model which are free parameters and which parameters are fixed.

        Parameters
        ----------
        b : bool
            True if parameter can be changed, False if it can't be changed.

        Returns
        -------
        None.

        """
        if b:
            # only allow to set to changeable if the parameter is not coupled
            if self.iscoupled():
                logger.warning('Attempt to set a coupled parameter to '
                               'changeable, ignoring this request')
            else:
                self.changeable = True
        else:
            self.changeable = False

    def setlinear(self, b):
        """
        Sets a boolean variable to indicate whether a parameter is linear or
        not. This can be used e.g. by the Fitter to know if a pure linear Model
        (all free parameters are linear) is present which allows to use a much
        faster fitting algorithm.

        Parameters
        ----------
        b : bool
            True for linear parameter, False otherwise.

        Returns
        -------
        None.

        """
        self.linear = b

    def islinear(self):
        """
        Returns whether this parameter is indicated as linear. This is used
        e.g. by a model to check if a linear fit can be used (all free
        parameters are linear) instead of a nonlinear.

        Returns
        -------
        bool
            True if parameter is indicated as linear, False otherwise.

        """
        return self.linear

    def ischanged(self):
        """
        Returns wether a parameter is indicated as changed (typically since
        last time eg. a Component used it for calculation).

        Returns
        -------
        bool
            True if parameter has changed, False otherwise.

        """
        if self.coupled and self.coupled_parameter is not None:
            return self.coupled_parameter.changed
        else:
            return self.changed

    def ischangeable(self):
        """
        Returns wether a parameter is changeable with setvalue().

        Returns
        -------
        bool
            True if parameter can be changed, False otherwise.

        """
        if self.iscoupled():
            return False  # this parameter cant'be changed
        else:
            return self.changeable

    def isbound(self):
        """
        Returns whether a parameter has active bounds.

        Returns
        -------
        bool
            True if parameter is bound, False otherwise.

        """
        return self.bound

    def iscoupled(self):
        """
        Returns whether a parameter is coupled to another parameter. Checks are
        performed to see if the coupled parameter is still valid.

        Returns
        -------
        bool
            True if coupled, False otherwise.

        """
        if self.coupled and self.coupled_parameter is not None:
            try:
                if not self.coupled_parameter.valid:
                    raise ValueError('invalid coupling reference in Parameter')
            except Exception:
                logger.warning('coupled parameter doesnt respond, deleting the'
                               'coupling')
                self.releasecoupling()
                return False
            return True
        else:
            return False

    def getcontroller(self):
        """
        Get a reference to the controller parameter if this is a coupled
        worker.

        Returns
        -------
        Parameter
            reference to a master Parameter instance if we are a coupled
            worker.
        None
            if we are not a coupled worker.

        """
        if self.iscoupled():
            return self.coupled_parameter
        else:
            return None

    def setname(self, name):
        """
        Sets the name of the parameter. name can be any string, preferably a
        meaningful description
        of the parameter

        Parameters
        ----------
        name : string
            name of the parameter.

        Returns
        -------
        None.

        """
        self.name = name

    def getname(self):
        """
        Get name string including "coupled to:" in case this parameter is
        coupled.

        Returns
        -------
        string
            the name of the parameter including 'coupled to...' in case the
            parameter is coupled.

        """
        if self.iscoupled():
            totalname = ' '.join((self.name, ':coupled to ',
                                  self.coupled_parameter.getname()))
            return totalname
        else:
            return self.name

    def printall(self):
        """
        Print some details on the parameter, mostly for debugging and testing.

        Returns
        -------
        None.

        """
        print(self.getname(), ' val=', self.getvalue(), ' changeable=',
              self.ischangeable(), ' lb=', self.lowerbound,
              ' ub=', self.upperbound)

    def getpurename(self):
        """
        Get the 'pure' name of the parameter. This is the exact name as given
        with setname() or during init() and doesnt include any info on eventual
        coupling to another parameter.

        Returns
        -------
        string
            The name of the parameter.

        """
        return self.name

    def couple(self, parameter, fraction=1.0):
        """
        Couples this parameter to another parameter with a certain coupling
        constant. This will make us into slaves of the master parameter that is
        passed. We no longer will be changeable and a call to getvalue() will
        result in the value of the controller parameter.

        Parameters
        ----------
        parameter : Parameter
            reference a another parameter to couple to.
        fraction : float, optional
            coupling constant. The default is 1.0.

        Returns
        -------
        None.

        """

        logger.info('coupling parameter %s with: %s', self.getname(),
                    parameter.getname())
        self.coupled = True
        self.coupled_parameter = parameter  # pointer to the master
        self.couple_fraction = fraction
        # test the master parameter to see if it is valid
        try:
            if not parameter.valid:
                raise ValueError('parameter no longer valid')
        except Exception:
            # something went wrong
            self.releasecoupling()
            logger.warning('failed to access the parameter')
            return

        # check if the proposed parameter is not yourself
        # otherwise recursion! and crash
        if (self == parameter):
            # something went wrong
            logger.warning('recursive coupling detected\n')
            self.releasecoupling()
            return
        self.setchangeable(
            False)
        # we no longer need to be changeable by the user since from now on
        # the master determines our value
        logger.info('done coupling')

    def getcouplingfactor(self):
        """
        Return the current value of the coupling factor, irrespective of
        whether the parameter is actually coupled or not.

        Returns
        -------
        float
            value of the coupling factor.

        """
        return self.couple_fraction

    def __sub__(self, other):
        """
        Subtract parameters.
        If other is an int or float, the number is subtracted from the value of
        the self parameter. The result has no predefined name and default
        (widest) boundaries.

        Parameters
        ----------
        other : Parameter or int or float
            The parameter or int or float to subtract.

        Raises
        ------
        TypeError
            In case other is not a Parameter or float or int.

        Returns
        -------
        s : Parameter
            Returns a reference to a -new- Parameter that holds the subtraction
            result.

        """
        if type(other) is Parameter:
            s = Parameter('', self.getvalue() - other.getvalue(),
                          changeallowed=True)
            return s
        elif type(other) is int or type(other) is float:
            s = Parameter('', self.getvalue() - other, changeallowed=True)
            return s

        else:
            raise TypeError('other should either be a Parameter, int or '
                            'float.')

    def __add__(self, other):
        """
        Add parameters. If other is an int or float, the number is added to the
        value of the self parameter. The result has no predefined name and
        default (widest) boundaries.

        Parameters
        ----------
        other : Parameter or int or float
            The parameter or int or float to add.

        Raises
        ------
        TypeError
            In case other is not a Parameter or float or int.

        Returns
        -------
        s : Parameter
            Returns a reference to a -new- Parameter that holds the addition
            result.

        """
        if type(other) is Parameter:
            s = Parameter('', self.getvalue() + other.getvalue(),
                          changeallowed=True)
            return s
        elif type(other) is int or type(other) is float:
            s = Parameter('', self.getvalue() + other, changeallowed=True)
            return s

        else:
            raise TypeError('other should either be a Parameter, int or '
                            'float.')

    def __mul__(self, other):
        """
        Multiply parameters. If other is an int or float, the number multiplied
        with the value of the self parameter. The result has no predefined
        name and default (widest) boundaries.

        Parameters
        ----------
        other : Parameter or int or float
            The parameter or int or float to multiply.

        Raises
        ------
        TypeError
            In case other is not a Parameter or float or int.

        Returns
        -------
        s : Parameter
            Returns a reference to a -new- Parameter that holds the
            multiplication result.

        """

        if type(other) is Parameter:
            s = Parameter('', self.getvalue() * other.getvalue(),
                          changeallowed=True)
            return s
        elif type(other) is int or type(other) is float:
            s = Parameter('', self.getvalue() * other, changeallowed=True)
            return s

        else:
            raise TypeError('other should either be a Parameter, int or '
                            'float.')

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        """
        Divide parameter by another parameter or float or int. The resulting
        Parameter has no predefined name and default (widest) boundaries.

        Parameters
        ----------
        other : Parameter or int or float
            The parameter or int or float to divide.

        Raises
        ------
        TypeError
            In case other is not a Parameter or float or int.

        Returns
        -------
        s : Parameter
            Returns a reference to a -new- Parameter that holds the division
            result.

        """
        if type(other) is Parameter:
            s = Parameter('', self.getvalue() / other.getvalue(),
                          changeallowed=True)
            return s
        elif type(other) is int or type(other) is float:
            s = Parameter('', self.getvalue() / other, changeallowed=True)
            return s
        else:
            raise TypeError('other should either be a Parameter, int or '
                            'float.')

    def releasecoupling(self):
        """
        Stops any coupling of this parameter with another

        Returns
        -------
        None.

        """
        self.coupled = False
        self.coupled_parameter = None
        self.setchangeable(True)  # not sure if this is needed, in principle
