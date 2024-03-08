# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys
from pyEELSMODEL.components.exponential import Exponential
from pyEELSMODEL.core.spectrum import Spectrumshape
import numpy as np
sys.path.append("..")  # Adds higher directory to python modules path.


def test_exponential():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Exponential(specshape, 1000, -0.001)  # create a component
    bkg.calculate()

    exp = 1000*np.exp(bkg.energy_axis*-0.001)
    assert np.all(np.abs(exp-bkg.data)) < 1e-6


def test_exponential_gradient():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Exponential(specshape, 1000, -0.001)  # create a component
    bkg.calculate()

    gradA = bkg.getgradient(bkg.parameters[0])
    gradA_exp = np.exp(bkg.energy_axis*-0.001)
    assert np.all(np.abs(gradA-gradA_exp)) < 1e-6

    gradb = bkg.getgradient(bkg.parameters[1])
    gradb_exp = bkg.energy_axis*1000*np.exp(bkg.energy_axis*-0.001)
    assert np.all(np.abs(gradb-gradb_exp)) < 1e-6


def test_exponential_autofit():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Exponential(specshape, 1000, -0.001)  # create a component
    bkg.calculate()

    bkg1 = Exponential(specshape, 1, -1)  # create a component
    bkg1.autofit(bkg, 0, 500)

    assert np.abs(bkg1.parameters[0].getvalue() - 1000) < 1e-3
    assert np.abs(bkg1.parameters[1].getvalue() + 0.001) < 1e-3


def main():
    test_exponential()
    test_exponential_gradient()
    test_exponential_autofit()


if __name__ == "__main__":
    main()
