import sys

import numpy as np
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model

from pyEELSMODEL.components.linear import Linear
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.fitters.minimizefitter import MinimizeFitter
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT

sys.path.append("..")  # Adds higher directory to python modules path.


def test_MLE_perform_fit():
    specshape = Spectrumshape(1, 100, 1024)
    pol = Gaussian(specshape, 10000, 400, 50)
    mod = Model(specshape, components=[pol])
    mod.calculate()
    s = Spectrum(specshape, data=np.random.poisson(mod.data))

    # values = [1,110,1]
    values = [10000, 400, 50]

    for param, value in zip(pol.parameters, values):
        param.setvalue(value)

    fit = MinimizeFitter(s, mod)
    fit.perform_fit()

    for value, coeff in zip(values, fit.coeff):
        rel_er = np.abs((coeff - value) / value)
        assert rel_er < 1e-2


def test_LSQ_perform_fit():
    specshape = Spectrumshape(1, 100, 1024)
    pol = Gaussian(specshape, 10000, 400, 50)
    mod = Model(specshape, components=[pol])
    mod.calculate()
    s = Spectrum(specshape, data=np.random.normal(mod.data, 1))

    values = [10000, 400, 50]
    for param, value in zip(pol.parameters, values):
        param.setvalue(value)

    fit = MinimizeFitter(s, mod, estimator='LSQ')
    fit.perform_fit()

    for value, coeff in zip(values, fit.coeff):
        rel_er = np.abs((coeff - value) / value)
        assert rel_er < 1e-2


def test_jacobian():
    specshape = Spectrumshape(1, 100, 1024)
    pol = Gaussian(specshape, 10000, 400, 50)
    gaus = Gaussian(specshape, 100, 300, 50)
    gaus.calculate()
    ll = MscatterFFT(specshape, gaus)

    mod = Model(specshape, components=[pol])
    mod.calculate()
    s = Spectrum(specshape, data=np.random.poisson(mod.data))
    fit = MinimizeFitter(s, mod)
    assert fit.jacobian != '2-point'

    mod1 = Model(specshape, components=[pol, ll])
    mod1.calculate()
    fit = MinimizeFitter(s, mod1)
    assert fit.jacobian == '2-point'

    fit = MinimizeFitter(s, mod, estimator='LSQ')
    assert fit.jacobian == '2-point'


def test_CRLB():
    specshape = Spectrumshape(1, 100, 1024)
    pol = Linear(specshape, m=1, q=1)
    mod = Model(specshape, components=[pol])
    mod.calculate()
    sig = np.copy(mod.data)

    nsamples = 100
    coeff_matrix = np.zeros((2, nsamples))

    for i in range(nsamples):
        pol.parameters[0].setvalue(1)
        pol.parameters[1].setvalue(1)

        ndata = np.random.poisson(sig)
        s = Spectrum(specshape, data=ndata)
        fit = MinimizeFitter(s, mod)
        fit.usegradients = False
        fit.perform_fit()
        coeff_matrix[:, i] = fit.coeff

    pol.parameters[0].setvalue(1)
    pol.parameters[1].setvalue(1)
    fit.set_information_matrix()

    std_a = np.std(coeff_matrix[0])
    crlb_a = fit.CRLB(pol.parameters[0])
    std_b = np.std(coeff_matrix[1])
    crlb_b = fit.CRLB(pol.parameters[1])

    assert np.abs(std_a - crlb_a) / (std_a + crlb_a) < 0.1
    assert np.abs(std_b - crlb_b) / (std_b + crlb_b) < 0.1


def main():
    test_MLE_perform_fit()
    test_LSQ_perform_fit()
    test_CRLB()
    test_jacobian()


if __name__ == "__main__":
    main()
