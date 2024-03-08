# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys
from pyEELSMODEL.components.polynomial import Polynomial
from pyEELSMODEL.core.spectrum import Spectrumshape
import numpy as np
sys.path.append("..")  # Adds higher directory to python modules path.


def test_polynomial():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Polynomial(specshape, order=3)  # create a component
    values = [-0.1, 20, 5, 3]
    for param, value in zip(bkg.parameters, values):
        param.setvalue(value)
    bkg.calculate()

    E = bkg.energy_axis

    norm3 = (E ** 3).sum()
    norm2 = (E ** 2).sum()
    norm1 = (E ** 1).sum()
    norm0 = (E ** 0).sum()

    poly = -(0.1 / norm3) * E ** 3 + (20 / norm2) * E ** 2 + (
                5 / norm1) * E + 3 / norm0

    assert np.all(np.abs(poly - bkg.data) < 1e-6)


def test_polynomial_gradient():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Polynomial(specshape, order=3)  # create a component
    values = [-0.1, 20, 5, 3]
    for param, value in zip(bkg.parameters, values):
        param.setvalue(value)
    bkg.calculate()

    E = bkg.energy_axis
    orders = [3, 2, 1, 0]
    for param, order in zip(bkg.parameters, orders):
        grad = bkg.getgradient(param)
        assert np.all(np.abs(E ** order - grad) < 1e-6)


def main():
    test_polynomial()
    test_polynomial_gradient()


if __name__ == "__main__":
    main()
