"""
Tester of the fitter class.
"""

import sys

from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model
import numpy as np

from pyEELSMODEL.components.linear import Linear
from pyEELSMODEL.core.fitter import Fitter

from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT

sys.path.append("..")  # Adds higher directory to python modules path.


def test_numerical_derivative():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    lin.calculate()

    mod = Model(specshape)
    mod.addcomponent(lin)

    s = Spectrum(specshape, data=np.ones(specshape.size))
    fit = Fitter(s, mod)

    dat = fit.numerical_partial_derivative(mod.getfreeparameters()[0])
    assert np.all(np.abs(dat - s.energy_axis) < 1e-3)

    dat = fit.numerical_partial_derivative(mod.getfreeparameters()[1])
    assert np.all(np.abs(dat - np.ones(dat.size)) < 1e-3)


def test_use_gradients():
    specshape = Spectrumshape(1, 100, 2048)
    gaus = Gaussian(specshape, A=100, centre=500, fwhm=50)
    mod = Model(specshape)
    mod.addcomponent(gaus)

    s = Spectrum(specshape, data=np.ones(specshape.size))
    fit = Fitter(s, mod)
    assert fit.usegradients

    gaus1 = Gaussian(specshape, A=10, centre=500, fwhm=100)
    gaus1.calculate()
    ll = MscatterFFT(specshape, gaus1)

    mod1 = Model(specshape, components=[gaus, ll])
    mod1.calculate()
    fit1 = Fitter(s, mod1)

    assert not fit1.usegradients


def test_deriv_matrix():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    lin.calculate()

    mod = Model(specshape)
    mod.addcomponent(lin)

    s = Spectrum(specshape, data=np.ones(specshape.size))
    fit = Fitter(s, mod)
    deriv = fit.calculate_derivmatrix()
    assert deriv.shape == (2, 2048)


def test_deriv_matrix_exclude():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    lin.calculate()

    mod = Model(specshape)
    mod.addcomponent(lin)

    s = Spectrum(specshape, data=np.ones(specshape.size))
    s.set_exclude_region_energy(300, 400)
    fit = Fitter(s, mod)
    deriv = fit.calculate_derivmatrix()
    assert deriv.shape == (2, 1948)


def test_degreesoffreedom():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 1, 1)
    gaus = Gaussian(specshape, A=100, centre=500, fwhm=50)

    mod = Model(specshape, components=[lin, gaus])

    s = Spectrum(specshape, data=np.ones(specshape.size))
    s.set_exclude_region_energy(300, 400)
    fit = Fitter(s, mod)
    assert fit.degreesoffreedom() == 1943


def test_param_index():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 1, 1)
    gaus = Gaussian(specshape, A=100, centre=500, fwhm=50)

    mod = Model(specshape, components=[lin, gaus])

    s = Spectrum(specshape, data=np.ones(specshape.size))
    fit = Fitter(s, mod)
    assert fit.get_param_index(gaus.parameters[0]) == 2
    assert fit.get_param_index(lin.parameters[0]) == 0


def test_information_matrix():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 1, 1)
    lin.calculate()
    mod = Model(specshape)
    mod.addcomponent(lin)

    s = Spectrum(specshape, data=np.ones(specshape.size))
    s.set_exclude_region_energy(300, 400)
    fit = Fitter(s, mod)
    fit.set_information_matrix()
    assert fit.information_matrix.shape == (2, 2)

    # theoretical determined fischer matrix for f(x) = x + 1
    theo_fischer = np.zeros((2, 2))
    E = s.energy_axis[np.invert(s.exclude)]
    theo_fischer[0, 0] = np.sum(E ** 2 / (E + 1))
    theo_fischer[1, 1] = np.sum(1 / (E + 1))
    theo_fischer[1, 0] = np.sum(E / (E + 1))
    theo_fischer[0, 1] = np.sum(E / (E + 1))

    fischer = fit.information_matrix
    # test: the theoretical and calculatd fischer matrix should be similar
    assert np.all(
        np.abs((fischer - theo_fischer) / (fischer + theo_fischer)) < 0.01)


def main():
    test_numerical_derivative()
    test_use_gradients()
    test_deriv_matrix()
    test_deriv_matrix_exclude()
    test_degreesoffreedom()
    test_param_index()
    test_information_matrix()


if __name__ == "__main__":
    main()
