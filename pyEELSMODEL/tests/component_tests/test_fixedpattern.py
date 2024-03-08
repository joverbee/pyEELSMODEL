# -*- coding: utf-8 -*-
"""
This script calculates shows as small script on how to use the fixed pattern
and perform a linear least squares fit
@author: joverbee
"""

import sys
from pyEELSMODEL.components.fixedpattern import FixedPattern
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.core.spectrum import Spectrumshape
from pyEELSMODEL.core.spectrum import Spectrum
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
import numpy as np

sys.path.append("..")  # Adds higher directory to python modules path.


def test_fixed_pattern():
    """
    Test where a linear fit is performed on the sum of gaussian which both
    will be a fixedpattern.
    :return:
    """
    specshape = Spectrumshape(1, 100, 1024)
    ga1 = Gaussian(specshape, 100, 500, 100)
    ga2 = Gaussian(specshape, 250, 800, 50)
    ga1.calculate()
    ga2.calculate()

    ndata = ga1.data + ga2.data + np.random.normal(0, 1, ga1.data.size)
    s = Spectrum(specshape, data=ndata)

    specshape1 = Spectrumshape(0.33, 50, 3*1500)

    ga1 = Gaussian(specshape1, 1, 500, 100)
    ga1.calculate()

    pat1 = Spectrum(specshape1, data=ga1.data)
    comp1 = FixedPattern(specshape, pat1)
    comp1.calculate()

    specshape2 = Spectrumshape(2, 200, int(626/2))

    ga2 = Gaussian(specshape2, 1, 800, 50)
    ga2.calculate()

    pat2 = Spectrum(specshape2, data=ga2.data)
    comp2 = FixedPattern(specshape, pat2)
    comp2.calculate()

    mod = Model(specshape)
    mod.addcomponent(comp1)
    mod.addcomponent(comp2)

    s.set_exclude_region(825, 1023)

    fit = LinearFitter(s, mod)
    fit.perform_fit()
    fit.set_fit_values()

    assert (np.abs(fit.coeff[0]-100)/100) < 1e-2
    assert (np.abs(fit.coeff[1]-250) / 250) < 1e-2


def test_fixed_pattern_gradient():
    specshape = Spectrumshape(1, 100, 1024)
    specshape1 = Spectrumshape(0.33, 50, 3 * 1500)

    ga1 = Gaussian(specshape1, 1, 500, 100)
    ga1.calculate()

    pat1 = Spectrum(specshape1, data=ga1.data)
    comp1 = FixedPattern(specshape, pat1)
    comp1.parameters[0].setvalue(10)
    comp1.calculate()

    grad = comp1.getgradient(comp1.parameters[0])
    assert np.all(np.abs(comp1.data/10 - grad) < 1e-3)


def main():
    test_fixed_pattern()
    test_fixed_pattern_gradient()


if __name__ == "__main__":
    main()
