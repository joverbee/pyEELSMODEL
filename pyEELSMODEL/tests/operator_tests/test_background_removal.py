import sys
sys.path.append("..") # Adds higher directory to python modules path.
import matplotlib.pyplot as plt
import pytest

from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.exponential import Exponential
from pyEELSMODEL.components.fast_background import FastBG2
from pyEELSMODEL.components.polynomial import Polynomial

from pyEELSMODEL.core.spectrum import Spectrumshape, Spectrum
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.components.CLedge.hs_coreloss_edgecombined import HSCoreLossEdgeCombined
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval
import numpy as np
from pyEELSMODEL.fitters.lsqfitter import LSQFitter
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.fitters.minimizefitter import MinimizeFitter

def make_multispectrum():
    specshape = Spectrumshape(1, 100, 1024)
    shape = (10,10)
    A_ar = np.reshape(np.linspace(1000,10000,shape[0]*shape[1]), shape)
    r_ar = np.reshape(np.linspace(2,5,shape[0]*shape[1]), shape)
    data = np.empty((shape[0], shape[1], specshape.size))
    for index in (np.ndindex(shape)):
        islice = np.s_[index]
        pw = PowerLaw(specshape, A=A_ar[islice], r=r_ar[islice])
        pw.calculate()
        data[islice] = pw.data

    mshape = MultiSpectrumshape(specshape.dispersion, specshape.offset, specshape.size, shape[0], shape[1])
    s_mult = MultiSpectrum(mshape, data=data)
    return s_mult


def test_init():
    specshape = Spectrumshape(1,10,100)
    s = Spectrum(specshape)

    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(5., 20.))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(5., 120.))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 120.))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=((50., 60.), (20.,30.)))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=((70., 60.), (20.,30.)))
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), model_type='Hello')
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), non_linear_fitter='Hello')
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), linear_fitting=True)
    with pytest.raises(ValueError):
        BackgroundRemoval(s, signal_range=(50., 60.), order=-5)
    with pytest.raises(TypeError):
        BackgroundRemoval(s, signal_range=(50., 60.), order=2.1)

    back = BackgroundRemoval(s, signal_range=((20., 30.), (40.,50.)))
    assert back.two_area is True

def test_set_indices():
    specshape = Spectrumshape(1,10,100)
    s = Spectrum(specshape)
    back= BackgroundRemoval(s, signal_range=(50.,60.))
    assert back.indices[0] == 40
    assert back.indices[1] == 50

    back= BackgroundRemoval(s, signal_range=((20.,30.),(50.,60.)))
    assert back.indices[0][0] == 10
    assert back.indices[0][1] == 20
    assert back.indices[1][0] == 40
    assert back.indices[1][1] == 50

def test_include_areas():
    specshape = Spectrumshape(1,10,100)
    s = Spectrum(specshape)
    back= BackgroundRemoval(s, signal_range=(50.,60.))
    back.include_areas()
    test = np.ones(s.size, dtype=bool)
    test[40:50] = False
    comparison = test == s.exclude
    assert comparison.all()

    back= BackgroundRemoval(s, signal_range=((20.,30.),(50.,60.)))
    back.include_areas()
    test = np.ones(s.size, dtype=bool)
    test[40:50] = False
    test[10:20] = False
    comparison = test == s.exclude
    assert comparison.all()

def test_make_model_type():
    specshape = Spectrumshape(1,10,100)
    s = Spectrum(specshape, data=np.ones(specshape.size))

    models = ['Polynomial', 'Exponential', 'Powerlaw', 'FastBG']
    comp_list = [Polynomial, Exponential, PowerLaw, FastBG2]
    for index, model_type in enumerate(models):
        back = BackgroundRemoval(s, signal_range=(50., 60.), model_type=model_type)
        back.make_background_model()
        assert type(back.model.components[0]) is comp_list[index]

def test_set_fit_type():
    specshape = Spectrumshape(1,10,100)
    s = Spectrum(specshape, data=np.ones(specshape.size))
    back= BackgroundRemoval(s, signal_range=(50.,60.), model_type='FastBG',
                            linear_fitting=True)
    back.make_background_model()
    back.set_fit_type()
    assert type(back.fitter) is LinearFitter

    back= BackgroundRemoval(s, signal_range=(50.,60.))
    back.make_background_model()
    back.set_fit_type()
    assert type(back.fitter) is LSQFitter

    back= BackgroundRemoval(s, signal_range=(50.,60.), non_linear_fitter='ML')
    back.make_background_model()
    back.set_fit_type()
    assert type(back.fitter) is MinimizeFitter


def test_reset_exclude_after_fit():
    specshape = Spectrumshape(1, 10, 100)
    s = Spectrum(specshape, data=np.ones(specshape.size))
    back= BackgroundRemoval(s, signal_range=(50.,60.))
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
    s =  make_multispectrum()
    back = BackgroundRemoval(s, signal_range=(100., 200.))
    fast_back = back.fast_calculate_multi()
    check = np.zeros((s.xsize, s.ysize, s.size))
    assert np.allclose(fast_back.multidata, check, atol=1e-3)

def test_model_fit():
    s =  make_multispectrum()
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

    #test if the exclude is resetted after the fit

# def test_reset_exclude_after_multifit():








# E0 = 300e3
# alpha = 5e-3
# beta = 20e-3
# specshape = Spectrumshape(1, 175, 2048)
# N_Ti = 5
#
# Ti_L = HSCoreLossEdgeCombined(specshape, A=N_Ti, E0=E0, alpha=alpha, beta=beta, element='Ti', edge='L')
# fbg = PowerLaw(specshape, A=10014, r=3.257)
#
# mod = Model(specshape)
# mod.addcomponent(Ti_L)
# mod.addcomponent(fbg)
#
# ### Fitting with one spectrum
# s = Spectrum(specshape, data = np.random.poisson(mod.data))
#
# method_list = ['Powerlaw', 'FastBG', 'Polynomial', 'Exponential']
# order = 3
#
# model_data = np.zeros((len(method_list), s.size))
# lin_l = [False, True, True, False]
# for i in range(len(method_list)):
#     backTi = BackgroundRemoval(s, signal_range=(380,450), model_type=method_list[i],
#                                order=3, linear_fitting=lin_l[i], non_linear_fitter='LSQ')
#     specTi = backTi.calculate()
#     mTi = backTi.model
#     model_data[i] = mTi.data
#
# plt.figure()
# plt.plot(s.energy_axis, s.data)
# for i in range(model_data.shape[0]):
#     plt.plot(s.energy_axis, model_data[i], label=method_list[i])
# plt.legend()
#
# #Comparing the ML fit with the LSQ fit
# backLSQ = BackgroundRemoval(s, signal_range=(380, 450), model_type=method_list[0],
#                            order=3, linear_fitting=lin_l[0], non_linear_fitter='LSQ')
# specLSQ = backLSQ.calculate()
# mLSQ = backLSQ.model
#
# backML = BackgroundRemoval(s, signal_range=(380, 450), model_type=method_list[0],
#                            order=3, linear_fitting=lin_l[0], non_linear_fitter='ML')
# specML = backML.calculate()
# mML = backML.model
#
# print('For ML A: '+str(np.round(mML.getfreeparameters()[0].getvalue(),2))+', r: '+str(np.round(mML.getfreeparameters()[1].getvalue(),2)))
# print('For LSQ A: '+str(np.round(mLSQ.getfreeparameters()[0].getvalue(),2))+', r: '+str(np.round(mLSQ.getfreeparameters()[1].getvalue(),2)))
#
# plt.figure()
# plt.plot(s.energy_axis, s.data)
# plt.plot(s.energy_axis, mML.data, label='ML')
# plt.plot(s.energy_axis, mLSQ.data, label='LSQ')
#
# ### Fitting using a multispectrum
# multi_data = np.zeros((5,5,s.size))
# count = 3
# delta = 3
# plt.figure()
# for i in range(multi_data.shape[0]):
#     for j in range(multi_data.shape[1]):
#          parTi = mod.getfreeparameters()[0]
#          parTi.setvalue(count)
#
#          count += delta
#          mod.calculate()
#          multi_data[i,j] = np.random.poisson(mod.data)
#
#          plt.plot(s.energy_axis, multi_data[i,j])
#
#
# multi_specshape = MultiSpectrumshape(specshape.dispersion,specshape.offset, specshape.size,
#                                      multi_data.shape[0], multi_data.shape[1])
#
# s_mult = MultiSpectrum(multi_specshape, data=multi_data)
#
# #Background removal from the multispectrum
# back_obj = BackgroundRemoval(s_mult, signal_range=(380, 450), model_type=method_list[0], linear_fitting=lin_l[0], non_linear_fitter='ML')
# back_obj.determine_fast_fit_parameters()
#
# fast_back = back_obj.fast_calculate_multi()
#
# s_back = back_obj.calculate_multi()
#
# fast_back.sum().plot()
#
# s_back.sum().plot()








