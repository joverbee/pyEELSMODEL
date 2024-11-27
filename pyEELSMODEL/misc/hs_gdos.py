import numpy as np
from scipy import interpolate

import pyEELSMODEL.misc.physical_constants as pc
import pyEELSMODEL.misc.hydrogen_gdos as hdos
import warnings


def getgosq(info1_1, info1_2, n):
    return info1_1 * (np.exp(np.arange(n) * info1_2) - 1) * 1e10


def getgosenergy(info2_1, info2_2, n):
    return info2_1 * (np.exp(np.arange(n) * info2_2 / info2_1) - 1)


def linear_interpolation(x, y, x1):
    m = (y[1] - y[0]) / (x[1] - x[0])
    q = y[0] - m * x[0]
    return m * x1 + q


def powerlaw(x, A, r):
    return A * x ** (-r)


def getinterpolatedgos(E, q, E_axis, q_axis, GOSmatrix, swap_axes=False):
    """
    Gets the interpolated value of the GOS from the E and q value.


    Parameters
    ----------
    E: float
        The energy from which the GOS should be interpolated
    q: float
        The q from the GOS should be interpolated
    E_axis: numpy array
        The energy axis on which the GOS is calculated
    q_axis: numpy array
        The q axis on which the GOS is calculated
    swap_axes: boolean
        Indicates if the axes for q and E should be swapped. There is a
        difference between the GOS from Segger and Zhang.

    Returns
    -------
    interpolated GOS matrix

    """
    index_q = np.searchsorted(q_axis, q, side='left')
    index_E = np.searchsorted(E_axis, E, side='left')

    if index_E == 0:
        return 0
    if index_E == E_axis.size:
        return 0.
    if index_q == 0:
        return GOSmatrix[index_E, 0]
    if index_q == q_axis.size:
        return 0.0

    dE = E_axis[index_E] - E_axis[index_E - 1]
    dq = q_axis[index_q] - q_axis[index_q - 1]

    distE = E - E_axis[index_E - 1]
    distq = q - q_axis[index_q - 1]

    if swap_axes:
        r0 = GOSmatrix[index_q - 1, index_E - 1] * (1 / (dE * dq)) * (
                    dE - distE) * (dq - distq)
        r1 = GOSmatrix[index_q, index_E - 1] * (1 / (dE * dq)) * (
                    dE - distE) * (distq)
        r2 = GOSmatrix[index_q - 1, index_E] * (1 / (dE * dq)) * (distE) * (
                    dq - distq)
        r3 = GOSmatrix[index_q, index_E] * (1 / (dE * dq)) * (distE) * (distq)
    else:
        r0 = GOSmatrix[index_E - 1, index_q - 1] * (1 / (dE * dq)) * (
                    dE - distE) * (dq - distq)
        r1 = GOSmatrix[index_E - 1, index_q] * (1 / (dE * dq)) * (
                    dE - distE) * (distq)
        r2 = GOSmatrix[index_E, index_q - 1] * (1 / (dE * dq)) * (distE) * (
                    dq - distq)
        r3 = GOSmatrix[index_E, index_q] * (1 / (dE * dq)) * (distE) * (distq)

    return r0 + r1 + r2 + r3


def get_powerLaw_extrapolation(rel_energy_axis, q_axis, GOSmatrix, E0, beta,
                               alpha,
                               q_steps=100, swap_axes=False):
    """
    If the energy axis for the cross section extends the range of calculated
    GOS then the last two point are used to extrapolate a power-law from.


    Parameters
    ----------
    rel_energy_axis: float
        The energy from which the GOS should be interpolated
    q: float
        The q from the GOS should be interpolated
    E_axis: numpy array
        The energy axis on which the GOS is calculated
    q_axis: numpy array
        The q axis on which the GOS is calculated
    swap_axes: boolean
        Indicates if the axes for q and E should be swapped. There is a
        difference between the GOS from Segger and Zhang.

    Returns
    -------
    A: float
        The amplitude of the power-law
    r: float
        The power of the power-law
    """
    E_axis = rel_energy_axis[-2:]
    dsigma_dE = np.zeros(E_axis.size)
    R = pc.R()
    T = pc.T(E0)
    gamma = pc.gamma(E0)
    for i in range(E_axis.size):
        E = E_axis[i]
        integral = 0

        qa0sq_min, qa0sq_max = hdos.get_qmin_max(E, E0, beta, alpha=alpha)
        logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                    q_steps)
        lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
        for j in range(logqa0sq_axis.size):
            q = np.sqrt(np.exp(logqa0sq_axis[j])) / pc.a0()
            theta = 2. * np.sqrt(np.abs(
                R * (np.exp(logqa0sq_axis[j]) - qa0sq_min) / (
                            4. * gamma ** 2 * T)))
            df_dE = getinterpolatedgos(E, q, rel_energy_axis, q_axis,
                                       GOSmatrix, swap_axes=swap_axes)
            # integral += df_dE * lnqa0sqstep
            integral += df_dE * lnqa0sqstep * hdos.correction_factor_kohl(
                alpha, beta, theta)
        dsigma_dE[i] = 4 * np.pi * pc.a0() ** 2 * (R / E) * (R / T) * integral

    r = np.log(dsigma_dE[1] / dsigma_dE[0]) / np.log(E_axis[0] / E_axis[1])
    A = dsigma_dE[0] / (pow(E_axis[0], -r))

    return A, r


# def test_getinterpolatedgos():
#     z_matrix = np.ones((50,50))
#     x = np.arange(z_matrix.shape[0])
#     y = np.arange(z_matrix.shape[1])
#
#     z_matrix[11,9] = 2
#     z_matrix[12,9] = 4
#     z_matrix[11,10] = 2
#     z_matrix[12,10] = 4
#
#     print(getinterpolatedgos(11.5, 9.5, x, y, z_matrix))


def dsigma_dE_HS(energy_axis, Z, ek, E0, beta, alpha, filename, q_steps=100):
    """
    Calculates the cross section from the Hartree-Slater cross sections
    from Rez. They are not used since they are not open-access.

    """

    with open(filename) as f:
        GOS_list = f.read().replace('\r', '').split()
    R = pc.R()
    # Map the parameters
    q_1 = float(GOS_list[2])
    q_2 = float(GOS_list[3])
    ncol = int(GOS_list[5])
    E_1 = float(GOS_list[6])
    E_2 = float(GOS_list[7])
    nrow = int(GOS_list[8])
    gos_array = np.array(GOS_list[9:], dtype=np.float64)
    # careful data is stored per rydberg and we want
    # per eV to be compatible with sigmak
    GOSmatrix = gos_array.reshape(nrow, ncol) / R

    rel_energy_axis = getgosenergy(E_1, E_2, nrow) + ek
    q_axis = getgosq(q_1, q_2, ncol)
    dsigma_dE = dsigma_dE_from_GOSarray(energy_axis, rel_energy_axis, ek, E0,
                                        beta, alpha, q_axis,
                                        GOSmatrix, q_steps=q_steps)
    return dsigma_dE


def dsigma_dE_from_GOSarray(energy_axis, rel_energy_axis, ek, E0, beta, alpha,
                            q_axis, GOSmatrix, q_steps=100, swap_axes=False):
    """
    Calculates the cross section from the GOS array. The integral over q-axis
    is done on a logarithmic scale.
    Note the speed of the calculation is not super-fast and could be improved
    by a better implementation of the interpolation and integration.

    Parameters
    ----------
    energy_axis: 1d numpy array
        The energy axis on which the cross section is calculated. [eV]
    rel_energy_axis: 1d numpy array
        The energy axis on which the GOS table is calculated. [eV]
    ek: float
        The onset energy of the calculated edge [eV]
    E0: float
        The acceleration voltage of the incoming electrons [V]
    alpha: float
        The convergence angle of the incoming probe [rad]
    beta:
        The collection angle of the outgoing electrons [rad]
    q_axis: 1d numpy array
        The momentum on which the GOS table are calculated. [kg m /s]?
    GOSmatrix: 2d numpy array
        The GOS
    q_steps: uint
        The number of q points used to numerically calculate the integral
        over the q direction. (default: 100)
    swap_axes: boolean
        The two GOS tables from Segger and Zhang have different axes. So
        for one, the energy axis is the first one and for the other it is
        the q-axis. Hence by swapping the axes we can use the same function
        for both. (default: False)

    Returns
    -------
    dsigma_dE: 1d numpy array
        The calculated cross section in m^2

    """
    R = pc.R()
    T = pc.T(E0)
    gamma = pc.gamma(E0)

    dsigma_dE = np.zeros(energy_axis.size)

    # check if there are energies larger then the max energy
    if energy_axis[-1] > rel_energy_axis[-1]:
        # then calculate a power law dependence of the cross sections
        powA, powr = get_powerLaw_extrapolation(rel_energy_axis, q_axis,
                                                GOSmatrix,
                                                E0, beta, alpha,
                                                q_steps=q_steps,
                                                swap_axes=swap_axes)

    for i in range(energy_axis.size):
        E = energy_axis[i]
        integral = 0
        if (E > ek) & (E <= rel_energy_axis[-1]):
            qa0sq_min, qa0sq_max = hdos.get_qmin_max(E, E0, beta, alpha=alpha)
            logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                        q_steps)
            lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
            for j in range(logqa0sq_axis.size):
                q = np.sqrt(np.exp(logqa0sq_axis[j])) / pc.a0()
                theta = 2. * np.sqrt(np.abs(
                    R * (np.exp(logqa0sq_axis[j]) - qa0sq_min) / (
                                4. * gamma ** 2 * T)))
                df_dE = getinterpolatedgos(E, q, rel_energy_axis, q_axis,
                                           GOSmatrix, swap_axes=swap_axes)
                # integral+= df_dE*lnqa0sqstep
                integral += df_dE * lnqa0sqstep * hdos.correction_factor_kohl(
                    alpha, beta, theta)
            # dsigma_dE[i] = 4*np.pi*pc.a0()**2*(R/E)*(R/T)*integral*dispersion
            dsigma_dE[i] = 4 * np.pi * pc.a0() ** 2 * (R / E) * (
                        R / T) * integral

        elif E > rel_energy_axis[-1]:
            dsigma_dE[i] = powerlaw(E, powA, powr)
        else:
            dsigma_dE[i] = 0

    return dsigma_dE


def dsigma_dE_from_GOSarray_approx(energy_axis, rel_energy_axis, E0, beta,
                                   GOSmatrix, swap_axes=False):
    """
    Calculates the cross section by using the approximation shown in
    https://doi.org/10.1016/0304-3991(89)90197-6, see Eq. 3
    """
    R = pc.R()
    T = pc.T(E0)
    # gamma = pc.gamma(E0)

    a0 = pc.a0()
    theta_E = pc.characteristic_angle(energy_axis, E0)

    # fac1 = 8 * a0**2 * R**2 / (energy_axis*pc.m0()*pc.speed_electron(E0)**2)
    fac1_ = 4 * np.pi * a0 ** 2 * (R / energy_axis) * (R / T)

    fac2 = np.log(1 + beta ** 2 / (theta_E ** 2))

    f = interpolate.interp1d(rel_energy_axis, GOSmatrix[:, 0], kind='linear',
                             fill_value='extrapolate')
    GOS = f(energy_axis)

    dsigma_dE = fac1_ * fac2 * GOS
    boolean = energy_axis < rel_energy_axis[0]
    dsigma_dE[boolean] = 0

    return dsigma_dE


def dsigma_dE_from_GOSarray_bound(energy_axis, free_energies, ek, E0, beta,
                                  alpha, q_axis, GOSmatrix, q_steps=100):
    """
    Calculates the cross section from the GOS array. The integral over q-axis
    is done on a logarithmic scale.
    This is a new method which is able to take the bounded states into account.
    Provided by Zezhong Zhang.

    Parameters
    ----------
    energy_axis: 1d numpy array
        The energy axis on which the cross section is calculated. [eV]
    free_energies: 1d numpy array
        The energy axis on which the GOS table is calculated without the onset
        energy [eV]
    ek: float
        The onset energy of the calculated edge [eV]
    E0: float
        The acceleration voltage of the incoming electrons [V]
    alpha: float
        The convergence angle of the incoming probe [rad]
    beta:
        The collection angle of the outgoing electrons [rad]
    q_axis: 1d numpy array
        The momentum on which the GOS table are calculated. [kg m /s]?
    GOSmatrix: 2d numpy array
        The GOS
    q_steps: uint
        The number of q points used to numerically calculate the integral
        over the q direction. (default: 100)
    swap_axes: boolean
        The two GOS tables from Segger and Zhang have different axes. So
        for one, the energy axis is the first one and for the other it is
        the q-axis. Hence by swapping the axes we can use the same function
        for both. (default: False)

    Returns
    -------
    dsigma_dE: 1d numpy array
        The calculated cross section in m^2

    """
    R = pc.R()
    T = pc.T(E0)
    gamma = pc.gamma(E0)

    # Define new energy axis which is uses the energies of the discreet
    # bounded states and in the continuum the usual energy axis.

    bool0 = free_energies < 0
    Ebound = free_energies[bool0] + ek

    dsigma_dE = np.zeros(energy_axis.size)
    dsigma_dE_bound = np.zeros(energy_axis.size)
    sigma = 2*(energy_axis[1] - energy_axis[0])

    rel_energy_axis = free_energies + ek
    # check if there are energies larger then the max energy
    if energy_axis[-1] > rel_energy_axis[-1]:
        # then calculate a power law dependence of the cross sections
        powA, powr = get_powerLaw_extrapolation(rel_energy_axis, q_axis,
                                                GOSmatrix,
                                                E0, beta, alpha,
                                                q_steps=q_steps,
                                                swap_axes=True)
    # the for loop over the bound states
    for i in range(Ebound.size):
        E = Ebound[i]

        integral = 0
        # the bounded states are differently interpolated
        qa0sq_min, qa0sq_max = hdos.get_qmin_max(E, E0, beta, alpha=alpha)
        logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                    q_steps)
        lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
        for j in range(logqa0sq_axis.size):
            q = np.sqrt(np.exp(logqa0sq_axis[j])) / pc.a0()
            theta = 2. * np.sqrt(np.abs(
                R * (np.exp(logqa0sq_axis[j]) - qa0sq_min) / (
                            4. * gamma ** 2 * T)))
            GOSarray = GOSmatrix[i, :]
            df_dE = getinterpolatedq(q, GOSarray, q_axis)

            # integral+= df_dE*lnqa0sqstep
            integral += df_dE * lnqa0sqstep * hdos.correction_factor_kohl(
                alpha, beta, theta)

        sig = 4 * np.pi * pc.a0() ** 2 * (R / E) * (R / T) * integral
        dsigma_dE_bound += gaussian(energy_axis, sig, E, sigma)

    # the for loop over the bound states
    for i in range(energy_axis.size):
        E = energy_axis[i]
        integral = 0
        if (E > ek) & (E <= rel_energy_axis[-1]):
            qa0sq_min, qa0sq_max = hdos.get_qmin_max(E, E0, beta, alpha=alpha)
            logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                        q_steps)
            lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
            for j in range(logqa0sq_axis.size):
                q = np.sqrt(np.exp(logqa0sq_axis[j])) / pc.a0()
                theta = 2. * np.sqrt(np.abs(
                    R * (np.exp(logqa0sq_axis[j]) - qa0sq_min) / (
                            4. * gamma ** 2 * T)))
                df_dE = getinterpolatedgos(E, q, rel_energy_axis, q_axis,
                                           GOSmatrix)
                # integral+= df_dE*lnqa0sqstep
                integral += df_dE * lnqa0sqstep * hdos.correction_factor_kohl(
                    alpha, beta, theta)
            # dsigma_dE[i] = 4*np.pi*pc.a0()**2*(R/E)*(R/T)*integral*dispersion
            dsigma_dE[i] = 4 * np.pi * pc.a0() ** 2 * (R / E) * (
                    R / T) * integral

        elif E > rel_energy_axis[-1]:
            dsigma_dE[i] = powerlaw(E, powA, powr)
        else:
            dsigma_dE[i] = 0

    return dsigma_dE + dsigma_dE_bound


def gaussian(E, integral, x0, sigma):
    # A = integral / (np.sqrt(2 * np.pi) * sigma)
    g = np.exp(-0.5 * (E - x0) ** 2 / sigma ** 2)
    g = integral * g / g.sum()
    return g


def getinterpolatedq(q, GOSarray, q_axis):
    """
    Gets the interpolated value of the GOS array as a function of q.
    Usefull for the bounded states


    Parameters
    ----------
    q: float
        The q from the GOS should be interpolated
    GOSarray:
        dddd
    q_axis: numpy array
        The q axis on which the GOS is calculated


    Returns
    -------
    interpolated GOS matrix

    """
    index_q = np.searchsorted(q_axis, q, side='left')

    if index_q == 0:
        return GOSarray[0]
    if index_q == q_axis.size:
        return 0.0

    dq = q_axis[index_q] - q_axis[index_q - 1]

    distq = q - q_axis[index_q - 1]

    r0 = GOSarray[index_q - 1] * (1/dq) * (dq-distq)
    r1 = GOSarray[index_q] * (1/dq) * (distq)

    return r0 + r1


def dsigma_dE_from_GOSarray_FastKohl(energy_axis, rel_energy_axis_, ek, E0, beta, alpha,
    q_axis, GOSmatrix_, q_steps=100):
    """
    Calculates the cross section from the GOS array. The integral over q-axis
    is done on a logarithmic scale.
    Note the speed of the calculation is imporved by a proper
    implementation of the interpolation and integration.

    Parameters
    ----------
    energy_axis: 1d numpy array
        The energy axis on which the cross section is calculated. [eV]
    rel_energy_axis: 1d numpy array
        The energy axis on which the GOS table is calculated. [eV]
    ek: float
        The onset energy of the calculated edge [eV]
    E0: float
        The acceleration voltage of the incoming electrons [V]
    alpha: float
        The convergence angle of the incoming probe [rad]
    beta:
        The collection angle of the outgoing electrons [rad]
    q_axis: 1d numpy array
        The momentum on which the GOS table are calculated. [kg m /s]?
    GOSmatrix: 2d numpy array
        The GOS
    q_steps: uint
        The number of q points used to numerically calculate the integral
        over the q direction. (default: 100)

    Returns
    -------
    dsigma_dE: 1d numpy array
        The calculated cross section in m^2

    """

    #add 0 for proper interpolation using the scipy method
    rel_energy_axis = np.pad(rel_energy_axis_-ek,(1,0))+ek #add 0 for proper interpolation using scipy method
    GOSmatrix = np.pad(GOSmatrix_,((0,0),(1,0),(0,0)))
    R = pc.R()
    T = pc.T(E0)
    gamma = pc.gamma(E0)

    dsigma_dE = np.zeros(energy_axis.size)

    # check if there are energies larger then the max energy
    if energy_axis[-1] > rel_energy_axis[-1]:
        # then calculate a power law dependence of the cross sections
        powA, powr = get_powerLaw_extrapolation(rel_energy_axis, q_axis,
                                                GOSmatrix,
                                                E0, beta, alpha,
                                                q_steps=q_steps,
                                                swap_axes=True)

    regime1 = np.logical_and(energy_axis > ek, energy_axis <= rel_energy_axis[-1])
    regime2 = energy_axis > rel_energy_axis[-1]
    regime3 = energy_axis <= ek


    dsigma_dE [regime1] = _FastKohl(energy_axis[regime1], E0, beta, alpha,q_steps,pc,T,gamma,rel_energy_axis,q_axis,GOSmatrix,R)

    if regime2.any():
            dsigma_dE[regime2]=powerlaw(energy_axis[regime2], powA, powr)

    dsigma_dE[regime3]=0

    return dsigma_dE

def _FastKohl(E, E0, beta, alpha,q_steps,pc,T,gamma,rel_energy_axis,q_axis,GOSmatrix,R):

    qa0sq_min, qa0sq_max = hdos.get_qmin_max(E, E0, beta, alpha=alpha)

    logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                q_steps)
    lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
    qs = np.sqrt(np.exp(logqa0sq_axis)) / pc.a0()
    
    thetas = 2. * np.sqrt(np.abs(R * (np.exp(logqa0sq_axis) - qa0sq_min) / (4. * gamma ** 2 * T)))

    rgi = interpolate.RegularGridInterpolator((q_axis,rel_energy_axis),GOSmatrix[:,:,0],)
    
    es = np.repeat(E[np.newaxis,:],q_steps,0)
    points = np.dstack((qs,es)).reshape((-1,2))
    out = rgi(points)


    integral=out.reshape(qs.shape)*_Fast_correction_factor_kohl(alpha, beta, thetas)

    integral=integral.sum(0)
    integral*=lnqa0sqstep


    integral[E<rel_energy_axis[1]]=0# gos and e_axis are padded for a proper behaviour of RegularGridInterpolator. This sets to 0 values affected by this.

    return 4 * np.pi * pc.a0() ** 2 * (R / E) * (R / T) * integral

def _Fast_correction_factor_kohl(alpha, beta, theta, min_alpha=1e-6):
    """
    STILL NEEDS TO BE VALIDATED
    Calculates the correction factor when using a convergent
    probe. For probes having is convergence angle smaller than
    min_alpha no correction is applied.
    Ultramicroscopy 16 (1985) 265-268:
    https://doi.org/10.1016/0304-3991(85)90081-6
     Parameters
     ----------
     alpha : float
         Convergence angle in radians
     beta : float
         Collection angle in radians
     theta : float
         The angle for which the correction factor should be calculated
     min_alpha : float
        Minimum convergence angle for which the correction is applied

     Returns
     -------
     corr_factor : float
        correction factor used in the integration
    """
    thetasq = theta ** 2
    alphasq = alpha ** 2
    betasq = beta ** 2

    
    if alpha < min_alpha:
        corr_factor = 1.
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")# invalid values for the correction are calculated, but tossed by np.where.
            corr_factor = np.where(theta <= np.abs(alpha - beta),min(alphasq, betasq)/ alphasq, _Fast_Fcorr_factor(alpha,beta,theta,thetasq,alphasq,betasq))
    return corr_factor
    
def _Fast_Fcorr_factor(alpha,beta,theta,thetasq,alphasq,betasq):
    x = (alphasq + thetasq - betasq) / (2. * alpha * theta)
    y = (betasq + thetasq - alphasq) / (2. * beta * theta)
    wortel = np.sqrt(4 * alphasq * betasq - (alphasq + betasq - thetasq) ** 2)
    corr_factor = (1 / np.pi) * (np.arccos(x) + (betasq / alphasq * np.arccos(y)) - (1 / (2 * alphasq) * wortel))
    return corr_factor


