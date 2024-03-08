import sys

import pytest

from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.exponential import Exponential
from pyEELSMODEL.components.fast_background import FastBG2
from pyEELSMODEL.components.polynomial import Polynomial

from pyEELSMODEL.core.spectrum import Spectrumshape, Spectrum
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval
import numpy as np
from pyEELSMODEL.fitters.lsqfitter import LSQFitter
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.fitters.minimizefitter import MinimizeFitter
sys.path.append("..")  # Adds higher directory to python modules path.


def make_multispectrum():
    specshape = Spectrumshape(1, 100, 1024)
    shape = (10, 10)
    A_ar = np.reshape(np.linspace(1000, 10000, shape[0] * shape[1]), shape)
    r_ar = np.reshape(np.linspace(2, 5, shape[0] * shape[1]), shape)
    data = np.empty((shape[0], shape[1], specshape.size))
    for index in (np.ndindex(shape)):
        islice = np.s_[index]
        pw = PowerLaw(specshape, A=A_ar[islice], r=r_ar[islice])
        pw.calculate()
        data[islice] = pw.data

    mshape = MultiSpectrumshape(specshape.dispersion, specshape.offset,
                                specshape.size, shape[0], shape[1])
    s_mult = MultiSpectrum(mshape, data=data)
    return s_mult


def test_init():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape)

    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(5., 20.))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(5., 120.))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 120.))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=((50., 60.), (20., 30.)))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=((70., 60.), (20., 30.)))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), model_type='Hello')
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.),
                          non_linear_fitter='Hello')
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), linear_fitting=True)
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), order=-5)
    with pytest.raises(TypeError):
        BackgroundRemoval(s, signal_range=(50., 60.), order=2.1)

    back = BackgroundRemoval(s, signal_range=((20., 30.), (40., 50.)))
    assert back.two_area is True


def test_set_indices():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape)
    back = BackgroundRemoval(s, signal_range=(50., 60.))
    assert back.indices[0] == 40
    assert back.indices[1] == 50

    back = BackgroundRemoval(s, signal_range=((20., 30.), (50., 60.)))
    assert back.indices[0][0] == 10
    assert back.indices[0][1] == 20
    assert back.indices[1][0] == 40
    assert back.indices[1][1] == 50


def test_include_areas():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape)
    back = BackgroundRemoval(s, signal_range=(50., 60.))
    back.include_areas()
    test = np.ones(s.size, dtype=bool)
    test[40:50] = False
    comparison = test == s.exclude
    assert comparison.all()

    back = BackgroundRemoval(s, signal_range=((20., 30.), (50., 60.)))
    back.include_areas()
    test = np.ones(s.size, dtype=bool)
    test[40:50] = False
    test[10:20] = False
    comparison = test == s.exclude
    assert comparison.all()


def test_make_model_type():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape, data=np.ones(specshape.size))

    models = ['Polynomial', 'Exponential', 'Powerlaw', 'FastBG']
    comp_list = [Polynomial, Exponential, PowerLaw, FastBG2]
    for index, model_type in enumerate(models):
        back = BackgroundRemoval(s, signal_range=(50., 60.),
                                 model_type=model_type)
        back.make_background_model()
        assert type(back.model.components[0]) is comp_list[index]


def test_set_fit_type():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape, data=np.ones(specshape.size))
    back = BackgroundRemoval(s, signal_range=(50., 60.), model_type='FastBG',
                             linear_fitting=True)
    back.make_background_model()
    back.set_fit_type()
    assert type(back.fitter) is LinearFitter

    back = BackgroundRemoval(s, signal_range=(50., 60.))
    back.make_background_model()
    back.set_fit_type()
    assert type(back.fitter) is LSQFitter

    back = BackgroundRemoval(s, signal_range=(50., 60.),
                             non_linear_fitter='ML')
    back.make_background_model()
    back.set_fit_type()
    assert type(back.fitter) is MinimizeFitter


def test_reset_exclude_after_fit():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape, data=np.ones(specshape.size))
    back = BackgroundRemoval(s, signal_range=(50., 60.))
    back.calculate()

    assert not s.exclude.all()


def test_background_subtraction():
    specshape = Spectrumshape(1, 175, 2048)
    fbg = PowerLaw(specshape, A=10000, r=3.)
    fbg.calculate()
    s = Spectrum(specshape, data=fbg.data)
    back = BackgroundRemoval(s, signal_range=(175., 275.))
    result = back.calculate()
    assert np.allclose(result.data, np.zeros(s.size), atol=1e-3)


def test_fast_fit():
    s = make_multispectrum()
    back = BackgroundRemoval(s, signal_range=(100., 200.))
    fast_back = back.fast_calculate_multi()
    check = np.zeros((s.xsize, s.ysize, s.size))
    assert np.allclose(fast_back.multidata, check, atol=1e-3)


def test_model_fit():
    s = make_multispectrum()
    back = BackgroundRemoval(s, signal_range=(100., 200.))
    s_back = back.calculate_multi()
    check = np.zeros((s.xsize, s.ysize, s.size))
    assert np.allclose(s_back.multidata, check, atol=1e-3)


def main():
    test_init()
    test_set_indices()
    test_include_areas()
    test_make_model_type()
    test_set_fit_type()
    test_reset_exclude_after_fit()
    test_background_subtraction()
    test_fast_fit()
    test_model_fit()


if __name__ == "__main__":
    main()
