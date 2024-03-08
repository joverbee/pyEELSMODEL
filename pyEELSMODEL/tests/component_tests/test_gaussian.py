# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.core.spectrum import Spectrumshape, Spectrum
from pyEELSMODEL.core.fitter import Fitter
from pyEELSMODEL.core.model import Model
import numpy as np
sys.path.append("..")  # Adds higher directory to python modules path.


def test_gaussian():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Gaussian(specshape, 100, 500, 10)  # create a component
    bkg.calculate()

    sigma = np.abs(10) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = 100*np.exp(-0.5*((bkg.energy_axis-500)/sigma)**2)
    assert np.all(np.abs(gauss-bkg.data) < 1e-6)


def test_gaussian_gradient():
    specshape = Spectrumshape(1, 100, 1024)  # dispersion, offset, size
    bkg = Gaussian(specshape, 100, 500, 10)  # create a component
    bkg.calculate()

    gradA = bkg.getgradient(bkg.parameters[0])
    sigma = np.abs(10) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = 1*np.exp(-0.5*((bkg.energy_axis-500)/sigma)**2)
    assert np.all(np.abs(gradA-gauss)) < 1e-6

    # test the gradients of the gaussian by comparing to the numerical
    # derivatives
    mod = Model(specshape)
    mod.addcomponent(bkg)
    s = Spectrum(specshape, data=np.ones(specshape.size))
    fit = Fitter(s, mod)

    dat = fit.numerical_partial_derivative(bkg.parameters[1])
    grad = bkg.getgradient(bkg.parameters[1])
    assert np.all(np.abs(dat-grad) < 1.5)

    dat = fit.numerical_partial_derivative(bkg.parameters[2])
    grad = bkg.getgradient(bkg.parameters[2])

    # plt.figure()
    # plt.plot(dat-grad)
    # plt.plot(dat)
    assert np.all(np.abs(dat - grad) < 0.01)


def main():
    test_gaussian()
    test_gaussian_gradient()


if __name__ == "__main__":
    main()
