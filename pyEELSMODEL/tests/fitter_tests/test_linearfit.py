import sys

from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.model import Model
import numpy as np
import pytest

from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.components.linear import Linear
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT

sys.path.append("..")  # Adds higher directory to python modules path.


def make_multispectrum(specshape, ll=None):
    shape = (20, 15)
    m, q = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    data = np.zeros((shape[0], shape[1], specshape.size))
    lin = Linear(specshape, 1, 1)

    for index in (np.ndindex(shape)):
        islice = np.s_[index]
        if ll is not None:
            mod = Model(specshape, components=[lin, ll])
        else:
            mod = Model(specshape, components=[lin])

        lin.parameters[0].setvalue(m[islice])
        lin.parameters[1].setvalue(q[islice])
        mod.calculate()
        data[islice] = np.random.normal(mod.data, 0.01)

    msh = MultiSpectrumshape(specshape.dispersion, specshape.offset,
                             specshape.size, shape[0], shape[1])
    s = MultiSpectrum(msh, data=data)
    return s, m, q


def make_multispectrum_convolutor(specshape):
    shape = (20, 15)
    m, q = np.meshgrid(np.arange(shape[1]) + 1, np.arange(shape[0]) + 1)
    data = np.zeros((shape[0], shape[1], specshape.size))
    lldata = np.zeros(data.shape)
    lin = Linear(specshape, 1, 1)

    gaus1 = Gaussian(specshape, A=10, centre=500, fwhm=100)

    for index in (np.ndindex(shape)):
        islice = np.s_[index]
        gaus1.parameters[2].setvalue(50 * m[islice])
        gaus1.calculate()
        lin.parameters[0].setvalue(m[islice])
        lin.parameters[1].setvalue(q[islice])
        ll = MscatterFFT(specshape, gaus1)
        mod = Model(specshape, components=[lin, ll])
        mod.calculate()
        data[islice] = np.random.normal(mod.data, 0.01)
        lldata[islice] = gaus1.data

    msh = MultiSpectrumshape(specshape.dispersion, specshape.offset,
                             specshape.size, shape[0], shape[1])
    s = MultiSpectrum(msh, data=data)
    ll = MscatterFFT(specshape, llspectrum=MultiSpectrum(msh, lldata))
    return s, m, q, ll


def test_nonlinear():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    gaus = Gaussian(specshape, A=1000, centre=500, fwhm=50)

    mod = Model(specshape, components=[lin, gaus])
    mod.calculate()

    s = Spectrum(specshape, data=np.random.normal(mod.data, 10))
    with pytest.raises(ValueError):
        LinearFitter(s, mod)


def test_A_matrix():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    # gaus = Gaussian(specshape, A=1000, centre=500, fwhm=50)

    mod = Model(specshape, components=[lin])
    mod.calculate()

    s = Spectrum(specshape, data=np.random.normal(mod.data, 10))
    fit = LinearFitter(s, mod)
    fit.calculate_A_matrix()
    assert fit.A_matrix.shape == (2048, 2)

    # the excluding does not modify the A matrix size
    # s = Spectrum(specshape, data = np.random.normal(mod.data,10))
    # s.set_exclude_region_energy(300,500)
    # fit = LinearFitter(s, mod)
    # fit.calculate_A_matrix()
    # assert fit.A_matrix.shape == (1848,2)


def test_perform_fit():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    mod = Model(specshape, components=[lin])
    mod.calculate()

    s = Spectrum(specshape, data=np.random.normal(mod.data, 0.01))
    lin.parameters[0].setvalue(1.)
    lin.parameters[1].setvalue(1.)

    s.set_exclude_region_energy(300, 500)
    fit = LinearFitter(s, mod)
    fit.perform_fit()
    assert np.abs(fit.coeff[0] - 2.5) < 2.5*0.1
    assert np.abs(fit.coeff[1] - 5) < 5*0.1


def test_perform_fit_convolutor():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    gaus1 = Gaussian(specshape, A=10, centre=500, fwhm=100)
    gaus1.calculate()
    ll = MscatterFFT(specshape, gaus1)

    mod = Model(specshape, components=[lin, ll])
    mod.calculate()

    s = Spectrum(specshape, data=np.random.normal(mod.data, 1))
    fit = LinearFitter(s, mod)
    fit.perform_fit()

    index = fit.get_param_index(lin.parameters[0])
    assert np.abs(fit.coeff[index] - 2.5) < 1e-1*2.5
    index = fit.get_param_index(lin.parameters[1])
    assert np.abs(fit.coeff[index] - 5) < 1e-1*5


def test_multifit():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    mod = Model(specshape, components=[lin])
    s, m, q = make_multispectrum(specshape)

    fit = LinearFitter(s, mod)
    fit.multi_fit()

    index = fit.get_param_index(lin.parameters[0])
    assert np.all(np.abs(fit.coeff_matrix[:, :, index] - m) < 0.01)
    index = fit.get_param_index(lin.parameters[1])
    assert np.all(np.abs(fit.coeff_matrix[:, :, index] - q) < 0.01)


def test_multifit_convolutor():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    gaus1 = Gaussian(specshape, A=10, centre=500, fwhm=100)
    gaus1.calculate()
    ll = MscatterFFT(specshape, gaus1)

    mod = Model(specshape, components=[lin, ll])
    s, m, q = make_multispectrum(specshape, ll=ll)

    fit = LinearFitter(s, mod)
    fit.multi_fit()

    index = fit.get_param_index(lin.parameters[0])
    assert np.all(np.abs(fit.coeff_matrix[:, :, index] - m) < 0.01)
    index = fit.get_param_index(lin.parameters[1])
    assert np.all(np.abs(fit.coeff_matrix[:, :, index] - q) < 0.01)


def test_multifit_multiconvolutor():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    s, m, q, ll = make_multispectrum_convolutor(specshape)
    mod = Model(specshape, components=[lin, ll])
    fit = LinearFitter(s, mod)
    fit.multi_fit()

    index = fit.get_param_index(lin.parameters[0])
    assert np.all(np.abs(fit.coeff_matrix[:, :, index] - m) < 0.01)
    index = fit.get_param_index(lin.parameters[1])
    assert np.all(np.abs(fit.coeff_matrix[:, :, index] - q) < 0.1)


def main():
    test_nonlinear()
    test_A_matrix()
    test_perform_fit()
    test_perform_fit_convolutor()
    test_multifit()
    test_multifit_convolutor()
    test_multifit_multiconvolutor()


if __name__ == "__main__":
    main()
