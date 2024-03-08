# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys

from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.core.spectrum import Spectrumshape
import numpy as np
sys.path.append("..")  # Adds higher directory to python modules path.


def test_powerlaw():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = PowerLaw(specshape, 1000, 3)  # create a component
    bkg.calculate()

    pw = 1000 * (bkg.energy_axis / bkg.energy_axis[0]) ** (-3)
    assert np.all(np.abs(pw - bkg.data)) < 1e-6


def test_powerlaw_gradient():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = PowerLaw(specshape, 1000, 3)  # create a component
    bkg.calculate()

    gradA = bkg.getgradient(bkg.parameters[0])
    gradA_exp = (bkg.energy_axis / bkg.energy_axis[0]) ** (-3)
    assert np.all(np.abs(gradA - gradA_exp)) < 1e-6

    gradb = bkg.getgradient(bkg.parameters[1])
    gradb_exp = 1000 * (bkg.energy_axis / bkg.energy_axis[0]) ** (-3) * (
        -1) * np.log(bkg.energy_axis / bkg.energy_axis[0])
    assert np.all(np.abs(gradb - gradb_exp)) < 1e-6


def test_powerlaw_autofit():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = PowerLaw(specshape, 1000, 3)  # create a component
    bkg.calculate()

    bkg1 = PowerLaw(specshape, 1, 1)  # create a component
    bkg1.autofit(bkg, 0, 500)

    assert np.abs(bkg1.parameters[0].getvalue() - 1000) < 1e-3
    assert np.abs(bkg1.parameters[1].getvalue() - 3) < 1e-3


def main():
    test_powerlaw()
    test_powerlaw_gradient()
    test_powerlaw_autofit()


if __name__ == "__main__":
    main()
