import numpy as np
from pyEELSMODEL.misc import physical_constants as pc

global XU, IE3, IE1, EK
XU = (0.52, 0.42, 0.30, 0.29, 0.22, 0.30, 0.22, 0.16, 0.12, 0.13, 0.13, 0.14,
      0.16, 0.18, 0.19, 0.22, 0.14, 0.11, 0.12, 0.12, 0.12, 0.1, 0.1, 0.1)
IE3 = (73.0, 99.0, 135.0, 164.0, 200.0, 245.0, 294.0, 347.0, 402.0, 455.0,
       513.0, 575.0, 641.0, 710.0, 779.0, 855.0, 931.0, 1021.0, 1115.0,
       1217.0, 1323.0, 1436.0, 1550.0, 1675.0)


IE1 = (118.0, 149.0, 189.0, 229.0, 270.0, 320.0, 377.0, 438.0, 500.0, 564.0,
       628.0, 695.0, 769.0, 846.0, 926.0, 1008.0, 1096.0, 1194.0, 1142.0,
       1248.0, 1359.0, 1476.0, 1596.0, 1727.0)
EK = (13, 25, 55, 111, 188, 284, 402, 532, 685, 867, 1072, 1305, 1560, 1839,
      2146, 2472, 2822, 3203, 3607, 4038, 4493, 4966)


def get_XU():
    return XU


def get_IE3():
    return IE3


def get_IE1():
    return IE1


def get_EK():
    return EK


def gdos_k(E, qa0sq, Z):
    """
    Calculates the hydrogenic GDOS for the k edge of an atom with atomic number
    Z.

    Parameters
    ----------
    E : float
        The energy in eV.
    qa0sq : float
        The q*a0**2 value (derived from momentum)
    Z : int
        Atomic number
    Returns
    -------
        GDOS
    """
    Zs = pc.Zs_k(Z)
    Q = (qa0sq / Zs ** 2)
    kHsq = E / (Zs ** 2 * pc.R()) - 1
    akH = np.sqrt(np.abs(kHsq))
    eps = 1e-99  # to not devide by zero
    # implemented in hyperspy (maybe value to small gives erros)
    if akH < 0.01:
        akH = 0.01

    if kHsq >= 0.0:

        beta = np.arctan(2 * akH / (Q - kHsq + 1))
        if beta < 0:
            beta = beta + np.pi
        teller = 256 * E * (Q + kHsq / 3 + 1 / 3) * np.exp(-2 * beta / akH)
        noemer = Zs ** 4 * pc.R() ** 2 * (
                    (Q - kHsq + 1) ** 2 + (4 * kHsq)) ** 3 * (
                             1 - np.exp(-2 * np.pi / akH)) + eps
        df_dE = teller / noemer
    else:
        y = -(1 / akH) * np.log(
            (Q + 1 - kHsq + 2 * akH) / (Q + 1 - kHsq - 2 * akH + eps))
        teller = 256 * E * (Q + kHsq / 3 + 1 / 3) * np.exp(y)
        noemer = Zs ** 4 * pc.R() ** 2 * (
                    (Q - kHsq + 1) ** 2 + (4 * kHsq)) ** 3 + eps

        df_dE = teller / noemer
    return df_dE


def gdos_l(E, qa0sq, Z):
    """
     Calculates the hydrogenic GDOS for the l edge of an atom with atomic
     number Z.

     Parameters
     ----------
     E : float
         The energy in eV.
     qa0sq : float
         The q*a0**2 value (derived from momentum)
     Z : int
         Atomic number
     Returns
     -------
         GDOS
     """
    if Z == 6:
        # small test for the L edge of oxygen
        EL3 = 5
        EL1 = 7
        U = 0.5
    else:
        index = int(Z) - 13
        EL3 = IE3[index]
        EL1 = IE1[index]
        U = XU[index]

    Zs = Z - 0.35 * 7 - 1.7
    Q = (qa0sq / Zs ** 2)
    kHsq = E / (Zs ** 2 * pc.R()) - 0.25
    akH = np.sqrt(np.abs(kHsq))

    RF = ((E + 0.1 - EL3) / (1.8 * Z ** 2)) ** U

    if akH < 0.01:
        akH = 0.01

    if kHsq >= 0.0:
        beta = np.arctan(akH / (Q - kHsq + 0.25))
        if beta < 0:
            beta = beta + np.pi
        C = np.exp(-2 * beta / akH)
        D = 1 - np.exp(-2 * np.pi / akH)
    else:
        y = (-1 / akH) * np.log(
            (Q + 0.25 - kHsq + akH) / (Q + 0.25 - kHsq - akH))
        C = np.exp(y)
        D = 1.0
    if E <= EL1:
        G = 2.25 * Q ** 4 - (0.75 + 3.0 * kHsq) * Q ** 3 \
            + (0.59375 - 0.75 * kHsq - 0.5 * kHsq ** 2) * Q ** 2 \
            + (0.11146 + 0.85417 * kHsq + 1.8833 * kHsq ** 2 + kHsq ** 3) * Q \
            + 0.0035807 + kHsq / 21.333 + kHsq ** 2 / 4.5714 \
            + kHsq ** 3 / 2.4 + kHsq ** 4 / 4

        A = ((Q - kHsq + 0.25) ** 2 + kHsq) ** 5
    else:
        G = Q ** 3 - ((5 / 3) * kHsq + 11. / 12.) * Q ** 2 + \
            (kHsq ** 2 / 3 + 1.5 * kHsq + 65. / 48.) * Q + kHsq ** 3 / 3 \
            + 0.75 * kHsq ** 2 + kHsq * 23. / 48. + 5. / 64.
        A = ((Q - kHsq + 0.25) ** 2 + kHsq) ** 4

    df_dE = RF * 32 * G * C * E / (A * D * pc.R() ** 2 * Zs ** 4)

    return df_dE


def correction_factor_kohl(alpha, beta, theta, min_alpha=1e-6):
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
    elif theta <= np.abs(alpha - beta):
        min_thetasq = min(alphasq, betasq)
        corr_factor = min_thetasq / alphasq
    # elif (theta>np.abs(alpha-beta))&(theta<alpha+beta):
    else:
        x = (alphasq + thetasq - betasq) / (2. * alpha * theta)
        y = (betasq + thetasq - alphasq) / (2. * beta * theta)
        wortel = np.sqrt(
            4 * alphasq * betasq - (alphasq + betasq - thetasq) ** 2)
        corr_factor = (1 / np.pi) * (
                    np.arccos(x) + (betasq / alphasq * np.arccos(y)) - (
                        1 / (2 * alphasq) * wortel))
    # else:
    #     corr_factor = 1

    return corr_factor


def convergence_correction_factor(alpha, beta, theta_sampling):
    """
    geometric correction factor for convergent beam in STEM
    #BY ZEZHONG
    Reference:

    Kohl, H. "A simple procedure for evaluating effective scattering
    cross-sections in STEM." Ultramicroscopy 16.2 (1985): 265-268.

    Args:
        alpha (float): incident beam convergence angle in rad
        beta (float): collection angle in rad
        theta_sampling (np.array): scattering angle in rad



    Returns:
        np.array: correction factor for each scattering angle
    """

    F = np.zeros(theta_sampling.shape)

    mask_1 = theta_sampling <= np.abs(alpha - beta)
    # print(mask_1)
    mask_2 = np.logical_and(theta_sampling > np.abs(alpha - beta),
                            theta_sampling < alpha + beta)
    # print(mask_2)
    mask_3 = theta_sampling >= (alpha + beta)
    # print(mask_3)

    x = (alpha ** 2 + beta ** 2 - theta_sampling[mask_2] ** 2) / (
                2 * alpha * beta)

    y = (beta ** 2 + theta_sampling[mask_2] ** 2 - alpha ** 2) / (
                2 * beta * theta_sampling[mask_2])

    F[mask_1] = min(alpha, beta) ** 2 / alpha ** 2

    term1 = np.arccos(x)
    term2 = (beta ** 2 / alpha ** 2) * np.arccos(y)
    wortel = np.sqrt(4 * alpha ** 2 * beta ** 2 - (
                (alpha ** 2 + beta ** 2 - theta_sampling[mask_2] ** 2) ** 2))
    term3 = (1 / 2 / alpha ** 2) * wortel

    F[mask_2] = (1 / np.pi) * (term1 + term2 - term3)

    F[mask_3] = 0

    return F


def dsigma_dE_hydrogenic(energy_axis, Z, ek, E0, beta, alpha, shell='K',
                         q_steps=100):
    """
    Calculates the differential cross section for a given energy axis. It uses
    the approximation of a convergent
    beam (alpha).


     Parameters
     ----------
     energy_axis : numpy array
         The energy in eV
     Z : int
         Atomic number
    ek: float
        The energy of the edge onset
    E0: float
        Energy of the incoming electron in eV
    beta: float
        Collection angle in radians
    alpha: float
        Convergence angle in radians (NOT YET IMPLEMENTED)
    shell: string
        Should be "K" or "L" to indicate which edge to calculate
    q_steps: int
        The number of calculations for every energy which is used to integrate
        over in q-space



     Returns
     -------
         dsigma_dE: numpy_array
            The differential energy cross section
    """

    if shell == 'K':
        gdos = gdos_k
    elif shell == 'L':
        gdos = gdos_l

    T = pc.T(E0)
    R = pc.R()
    gamma = pc.gamma(E0)
    dsigma_dE = np.zeros(energy_axis.size)
    for i in range(energy_axis.size):
        E = energy_axis[i]
        if E > ek:
            qa0sq_min, qa0sq_max = get_qmin_max(E, E0, beta, alpha=alpha)

            logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                        q_steps)
            lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
            integral = 0
            for j in range(logqa0sq_axis.size):
                qa0sq = np.exp(logqa0sq_axis[j])
                theta = 2. * np.sqrt(
                    np.abs(R * (qa0sq - qa0sq_min) / (4. * gamma ** 2 * T)))
                # integral += gdos(E, qa0sq, Z)*lnqa0sqstep
                integral += correction_factor_kohl(alpha, beta, theta) * gdos(
                    E, qa0sq, Z) * lnqa0sqstep  # correction does not work

            # dsigma_dE[i] = 4*np.pi*pc.a0()**2*(R/E)*(R/T)*integral*dispersion
            dsigma_dE[i] = 4 * np.pi * pc.a0() ** 2 * (R / E) * (
                        R / T) * integral

        else:
            dsigma_dE[i] = 0

    return dsigma_dE


def get_qmin_max(E, E0, beta, alpha=1e-9, return_q=False):
    """
    Returns the minimum and maximum q vector over which to integrate to
    get the cross section.

    """
    qa0sq_min = E ** 2 / (4 * pc.R() * pc.T(E0)) + (E ** 3) / (
                8 * pc.gamma(E0) ** 3 * pc.R() * pc.T(E0) ** 2)
    qa0sq_max = qa0sq_min + 4 * pc.gamma(E0) ** 2 * (pc.T(E0) / pc.R()) * (
        np.sin((beta + alpha) / 2)) ** 2
    if return_q:
        q_min = np.sqrt(qa0sq_min / pc.a0() ** 2)
        q_max = np.sqrt(qa0sq_max / pc.a0() ** 2)
        return q_min, q_max
    else:
        return qa0sq_min, qa0sq_max
