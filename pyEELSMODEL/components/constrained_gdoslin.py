from pyEELSMODEL.components.gdoslin import GDOSLin
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
import pyEELSMODEL.misc.physical_constants as pc
import numpy as np


class ConstrainedGDOSLin(GDOSLin):
    def __init__(self,  specshape, estart, ewidth=50, degree=21,
                 interpolationtype='quadratic', beta=50e-3, alpha=1e-9,
                 E0=300e3, connected_edge=None):

        super().__init__(specshape, estart, ewidth, degree, interpolationtype)

        self.connected_edge = connected_edge
        self.beta = beta
        self.E0 = E0
        self.alpha = alpha

        self.calculate_integral()

    @classmethod
    def gdoslin_from_edge(cls, specshape, component, pre_e=5, ewidth=50,
                          degree=21, interpolationtype='quadratic'):
        """
        Class method is made to create an gdoslin which is connected to a
        coreloss edge. No need to input the onset energy
        """

        if not isinstance(component, CoreLossEdge):
            raise TypeError(r'Component should be a CoreLossEdge')

        estart = component.onset_energy - pre_e

        beta = component.parameters[2].getvalue()
        alpha = component.parameters[3].getvalue()
        E0 = component.parameters[1].getvalue()

        comp = ConstrainedGDOSLin(specshape, estart, ewidth=ewidth,
                                  degree=degree,
                                  interpolationtype=interpolationtype,
                                  beta=beta, alpha=alpha, E0=E0,
                                  connected_edge=component)
        comp.set_gdos_name()
        return comp

    def calculate_prefactor(self):
        """
        This is the prefactor which determines the constrains used for
        our work
        """

        theta_E = pc.characteristic_angle(self.energy_axis, self.E0,
                                          use_rel=True)
        print('collection angle is: ' + str(self.beta))
        prefactor = self.energy_axis \
                    / np.log(1 + self.beta ** 2 / theta_E ** 2)
        prefactor = prefactor / prefactor[0]
        self.prefactor = prefactor #de behoudswet voor onze GOS

    def calculate_convergent_prefactor(self):
        """
        Calculates the prefactor when a convergent probe is used. The integral
        needs to be solved numerically due to the factor coming from the
        convergent probe.
        """

        F, theta = self.connected_edge.get_convergence_correction_factor(nsamples=1000)
        theta_E = pc.characteristic_angle(self.energy_axis, self.E0,
                                          use_rel=True)

        int_angle = np.zeros_like(self.energy_axis)
        for ii in range(self.energy_axis.size):
            dtheta = np.diff(theta)
            int_angle[ii] = np.sum(F[:-1]*dtheta*2*np.pi*theta[:-1]
                                   /(theta[:-1]**2 + theta_E[ii]**2))

        print('convergence angle is: ' + str(self.alpha))
        print('collection angle is: ' + str(self.beta))


        prefactor = self.energy_axis/int_angle
        prefactor = prefactor / prefactor[0]
        self.convergent_prefactor = prefactor


    def calculate_integral(self):
        """
        Test case when bethe sum rule says that  NOT the total cross section
        is conserved but (Ei-En)*sigma(E) integrated is conserved. This
        modifies the constrains in a straightforward why but needs to know the
        integral of each basis function x dE. This is what gets calculated

        :return:
        """
        if self.alpha > 1e-6:
            print('Convergent incoming beam is used')
            self.calculate_convergent_prefactor()
        else:
            print('Parallel incoming beam is used')
            self.calculate_prefactor()

        self.integral = []
        for ii, param in enumerate(self.parameters[2:]):
            for par in self.parameters[2:]:
                if par == param:
                    par.setvalue(1)
                else:
                    par.setvalue(0)
            self.calculate()

            if self.alpha > 1e-6:
                res = self.data*self.convergent_prefactor
            else:
                res = self.data*self.prefactor
            # res = self.data
            self.integral.append(res.sum())

        for param in self.parameters[2:]:
            param.setvalue(1)

        self.integral = np.array(self.integral)[np.newaxis, :]

    def total_atomic_cross_section(self):
        """
        Calculate the total atomic cross section times the prefactor to have
        a fair comparison to evaluate the constrains or too apply some
        inequality constrains.
        """
        index0 = self.get_energy_index(self.parameters[0].getvalue())
        index1 = self.get_energy_index(self.parameters[0].getvalue()
                                       + self.parameters[1].getvalue())

        if self.alpha > 1e-6:
            atomic_sum = np.sum(self.convergent_prefactor[index0:index1] *
                                self.connected_edge.cross_section[
                                index0:index1])
        else:
            atomic_sum = np.sum(self.prefactor[index0:index1] *
                                self.connected_edge.cross_section[
                                index0:index1])


        self.atomic_sum = atomic_sum

    def get_equality_constraints(self):
        '''
        The sum to zero could be interpreted as a equality constrain.
        :return:
        '''
        return self.integral


    def shadow(self, specshape, onset_shift=6, ratio=.5):
        '''
        Returns a dependent copy of itself applied to specshape
        '''

        onset = self.parameters[0].getvalue()
        interval = self.parameters[1].getvalue()

        shadowed = ConstrainedGDOSLin(specshape, estart=onset + onset_shift,
                                      ewidth=interval, degree=self.degree,
                                      interpolationtype=self.interpolationtype)

        for ii in range(self.degree):
            shadowed.parameters[ii + 2].couple(self.parameters[ii + 2],
                                               fraction=ratio)

        return shadowed

    def check_equality_constrain(self):
        """
        Function which checks if the values
        :return:
        """
        values = np.zeros(self.integral.shape[1])
        for ii, param in enumerate(self.parameters[2:]):
            values[ii] = param.getvalue()

        sum_rule = np.sum(values * self.integral[0])

        print('The equality constrain should put the value to '
              'zero: {}'.format(sum_rule))