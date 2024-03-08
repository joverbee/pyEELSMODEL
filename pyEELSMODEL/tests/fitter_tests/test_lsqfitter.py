import sys

import numpy as np
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model

from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.fitters.lsqfitter import LSQFitter

sys.path.append("..")  # Adds higher directory to python modules path.


def test_perform_fit():
    specshape = Spectrumshape(1, 100, 1024)
    pol = Gaussian(specshape, 10000, 400, 50)
    mod = Model(specshape, components=[pol])
    mod.calculate()
    s = Spectrum(specshape, data=np.random.normal(mod.data, 1))

    # values = [1,110,1]
    values = [10000, 400, 50]

    for param, value in zip(pol.parameters, values):
        param.setvalue(value)

    fit = LSQFitter(s, mod)
    fit.perform_fit()

    for value, coeff in zip(values, fit.coeff):
        rel_er = np.abs((coeff - value) / value)
        assert rel_er < 1e-2


def test_use_bounds():
    specshape = Spectrumshape(1, 100, 1024)
    pol = Gaussian(specshape, 10000, 400, 50)
    mod = Model(specshape, components=[pol])
    mod.calculate()
    s = Spectrum(specshape, data=np.random.normal(mod.data, 1))

    fit = LSQFitter(s, mod, use_bounds=True)
    assert not fit.use_bounds

    fit = LSQFitter(s, mod, method='trf', use_bounds=True)
    assert fit.use_bounds


def main():
    test_perform_fit()
    test_use_bounds()


if __name__ == "__main__":
    main()
