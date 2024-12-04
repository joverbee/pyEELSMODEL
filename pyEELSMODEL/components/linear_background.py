"""
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
"""
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
import numpy as np

def dec2bin(n, idx):
    if n > 1:
        dec2bin(n // 2, idx)

    idx.append(n % 2)


def harmonic_space(x0, x1, n, eps=1e-3):

    lnx0 = np.log(x0)
    lnx1 = np.log(x1)
    x = np.exp(np.linspace(lnx0, lnx1, n))
    ii_test = n // 2

    for jj in range(100):
        x_test = x[ii_test] + 0.
        for ii in range(1, len(x) - 1):
            x[ii] = 2. * x[ii-1] * x[ii+1] / (x[ii-1] + x[ii+1])

        if np.abs(x[ii_test] - x_test) < eps:
            break

    return x


class LinearBG(Component):
    """
    Fast background model, where the number of components can be varied.
    Convexity constraints are added to the model 

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model
    rlist : list
        List of the r values used in the linear background model.
        (default: [1,2,3,4,5])
    constrains : str
        The inequality constraints used in the background model. 
        One can chose between 'sufficient', 'necessary', 'non-neg'.

    Returns
    -------
    """

    def __init__(self, specshape, rlist=[1, 2, 3, 4, 5],
                 constrains='sufficient'):
        super().__init__(specshape)

        n = len(rlist)
        for i in range(n):
            pname = 'a' + str(i)
            p = Parameter(pname, 1.0, True)
            p.setboundaries(-np.inf, np.inf)
            # is this true as we will multiply this with another cross section
            p.setlinear(True)
            self._addparameter(p)

            qname = 'r' + str(i)
            q = Parameter(qname, rlist[i], changeallowed=False)
            q.sethasgradient(False)
            self._addparameter(q)

        self.n = n  # number of terms in the sum

        # don't convolute the background it only gives problems and
        # adds no extra physics
        self._setcanconvolute(False)
        self._setshifter(False)  # it is not a shifter type of component

        self.npoints = 3
        self.use_approx = constrains #which approximation for convexity to use
        self._setname('Linear constrained background')

    def calculate(self):
        if self.suppress:
            self.data[:]=0
            self.setunchanged()
            return
        changes = False
        for param in self.parameters:
            changes = changes or param.ischanged()
        if changes:
            Alist = []
            rlist = []
            for i in range(self.n):
                p = self.parameters[2 * i]
                Alist.append(p.getvalue())
                q = self.parameters[2 * i + 1]
                rlist.append(q.getvalue())

            self.data = self.linear_background(Alist, rlist)
        self.setunchanged()  # put parameters to unchanged

    def linear_background(self, Alist, rlist):
        E = self.energy_axis
        Estart = E[0]
        if Estart < 1:
            Estart = 1

        mask = (E > 0)  # only meaningfull when E>0 and r>0
        signal = np.zeros(E.size)
        for i in range(len(Alist)):
            signal += mask * (Alist[i] * (E / Estart) ** (-rlist[i]))
        return signal

    def getgradient(self, parameter):
        """calculate the analytical partial derivative wrt parameter j
        returns true if succesful, gradient is stored in component.gradient
        """
        # todo implement the analytical gradient
        for ii, param in enumerate(self.parameters):
            if param is parameter:
                index = ii

        if parameter in self.parameters[::2]:
            r = self.parameters[index + 1].getvalue()
            partial = self.linear_background([1], [r])
            return partial

        else:
            return None

    def get_rlist(self):
        rlist = []
        for param in self.parameters:
            if param.linear:
                pass
            else:
                rlist.append(param.getvalue())
        return rlist
    
    def get_convex_boundaries(self):
        if (self.use_approx == None):
            return None

        # 'relaxed' is legacy
        if (self.use_approx.lower() == 'sufficient') \
                or (self.use_approx.lower() == 'relaxed'):
            return self.convexity_constraints_sufficient()
        # 'approx' is legacy
        elif (self.use_approx.lower() == 'necessary') \
                or (self.use_approx.lower() == 'approx'):
            return self.convexity_constraints_necessary()

        elif (self.use_approx.lower() == 'non-neg'):
            return self.convexity_constraints_nonneg()
        
        else:
            print('Error: \'' + self.use_approx + '\' is not a valid mode.')
            print('Use \'sufficient\', \'necessary\', or \'non-neg\'.')
            return None

    def convexity_constraints_sufficient(self):
        '''
        :param x_01: array with start and endpoint of energy range
        :param r: array with the exponents, the model is E^-r. So do not forget
        about the minus sign
        :return: G: matrix with the constraints
        '''

        r = np.asarray(self.get_rlist())
        delta_r = r[0] - r

        x0 = self.energy_axis[0]
        x1 = self.energy_axis[-1]

        r_m = np.abs(delta_r[1:])
        xh = ((r_m + 1.) / r_m) * (x0 ** (-r_m) - x1 ** (-r_m)) \
             / (x0 ** (-r_m - 1.) - x1 ** (-r_m - 1.))
        xh = np.mean(xh)

        n = len(r)
        pf = r * (r + 1.) * (x0**r)  # added the E0 in the formula DJ
        pf = np.reshape(pf, (1, n))

        G = np.zeros((2 ** (n - 1) + 2, n)) + pf

        for ii in range(2 ** (n - 1)):
            idx0 = []
            dec2bin(ii, idx0)
            idx = np.zeros((n - 1,))

            # Fill out the zeros. Start from the back.
            for jj in range(len(idx0)):
                kk = -jj - 1
                idx[kk] = idx0[kk]

            # Fill out G. Start from the back.
            for jj in range(n - 1):
                kk = -jj - 1
                if idx[kk] == 0:
                    if ii == 0:
                        G[ii, kk] *= x0 ** delta_r[kk]
                    else:
                        G[ii, kk] *= xh ** delta_r[kk]
                else:
                    if ii == (2 ** (n - 1) - 1):
                        G[ii, kk] *= x1 ** delta_r[kk]
                    else:
                        G[ii, kk] *= x1 ** delta_r[kk] * \
                                     (1. + (1. - xh / x1) * (r[kk] + 2.))

        # Value at the end of the axis is positive
        G[-1, :] = (x0 ** r) * x1 ** delta_r
        # Derivative at the end of the axis is negative
        G[-2, :] = r * (G[-1, :] + 0.)

        return G

    def convexity_constraints_necessary(self):
        '''
        :param x_01: array with start and endpoint of energy range
        :param r: array with the exponents, the model is E^-r. So do not forget
         about the minus sign
        :param m: number of constraints.
        :param flag_decr: set to 1 if the 1st derivative
        :return: G: matrix with the constraints
        '''

        r = np.asarray(self.get_rlist())
        x0 = self.energy_axis[0]  # can be changed when only fitting a subregion
        x1 = self.energy_axis[-1]

        m = max((self.npoints, 2))  # at least two points for forced convexity
        # in approx method
        x = harmonic_space(x0, x1, m)  # m is guaranteed to be 2 or higher

        n = len(r)
        pf = r * (r + 1.) * (x0**r)  # pre-factor
        delta_r = r[0] - r
        G = np.zeros((m + 2, n)) + np.reshape(pf, (1, n))

        for ii in range(0, m):
            G[ii, 1:] *= x[ii] ** delta_r[1:]
        # Value at the end of the axis is positive
        G[-1, :] = (x0 ** r) * x1 ** delta_r
        # Derivative at the end of the axis is negative
        G[-2, :] = r * (G[-1, :] + 0.)

        return G

    def convexity_constraints_nonneg(self):
        '''
        :param x_01: array with start and endpoint of energy range
        :param r: array with the exponents, the model is E^-r. So do not forget
        about the minus sign
        :param m: number of constraints.
        :param flag_decr: set to 1 if the 1st derivative
        :return: G: matrix with the constraints
        '''

        r = np.asarray(self.get_rlist())
        n = len(r)

        G = np.eye(n)

        return G