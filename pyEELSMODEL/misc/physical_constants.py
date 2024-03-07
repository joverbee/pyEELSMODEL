"""
Definition of the physical constants used with its units.
"""

import numpy as np


def R():
    """
    Rydberg constant in eV
    """

    return 13.606


def e():
    """
    Electron charge in C
    """

    return 1.602e-19


def m0():
    """
    Rest mass in kg
    """

    return 9.110e-31


def a0():
    """
    Bohr radius in m
    """

    return 5.292e-11


def speed_electron(E0):
    """
    Relativistic speed of electron

    """
    return c() * np.sqrt(1 - (1 / (1 + E0 * e() / (m0() * c() ** 2)) ** 2))


def joule_to_eV(J):
    return J * 6.24150913e18


def c():
    """
    Speed of light in m/s
    """

    return 2.998e8


def gamma(E0, m=m0()):
    """
    The formula works for energies of the electron
    which are not to close to the speed of light?

    :param E0: Energy incoming electron (eV)
        m: mass of the particle (kg)
        electron mass as default
    :return: gamma: float
    """
    gamma = 1 + e() * E0 / (m * c() ** 2)
    return gamma


def T(E0, m=m0()):
    """
    The effective incident energy in eV

    :param E0:energy in eV?
    :param m:mass in kg
    :return:
    """
    gam = gamma(E0, m=m)

    return E0 * (1 + gam) / (2 * gam ** 2)


def characteristic_angle(E, E0, use_rel=True):
    if use_rel:
        # theta_E = E / ((E0 + m0()*c()**2) *(speed_electron(E0)/c())**2)
        # theta_E = E / gamma(E0)*m0()*(speed_electron(E0)**2)
        theta_E = E / (2 * gamma(E0) * T(E0))
    else:
        theta_E = E / (2 * E0)
    return theta_E


def Zs_k(Z):
    """
    Corrects for the screening of the electrons
    for the k shell.
    :param Z: The number of protons
    :return: Corrected Z value
    """
    # return Z-0.3125 #value of EELSmodel
    return Z - 0.5  # value of hyperspy (Sigma3K)


def h(unit='J'):
    if unit == 'J':
        return 6.625607015e-34  # J/s
    elif unit == 'eV':
        return 4.135667696e-16  # eV/s


def hbar(unit='J'):
    return h(unit=unit) / (2 * np.pi)
