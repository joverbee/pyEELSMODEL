import sys

sys.path.append("..")  # Adds higher directory to python modules path.
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.model import Model
import numpy as np
import pytest

from pyEELSMODEL.components.fast_background import FastBG2
import matplotlib.pyplot as plt
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.components.linear import Linear
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT

def make_multispectrum(specshape, ll=None):
    shape = (20,15)
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

    msh = MultiSpectrumshape(specshape.dispersion, specshape.offset, specshape.size, shape[0], shape[1])
    s = MultiSpectrum(msh, data=data)
    return s, m, q

def make_multispectrum_convolutor(specshape):
    shape = (20,15)
    m, q = np.meshgrid(np.arange(shape[1])+1, np.arange(shape[0])+1)
    data = np.zeros((shape[0], shape[1], specshape.size))
    lldata = np.zeros(data.shape)
    lin = Linear(specshape, 1, 1)

    gaus1 = Gaussian(specshape, A=10, centre=500, fwhm=100)

    for index in (np.ndindex(shape)):
        islice = np.s_[index]
        gaus1.parameters[2].setvalue(50*m[islice])
        gaus1.calculate()
        lin.parameters[0].setvalue(m[islice])
        lin.parameters[1].setvalue(q[islice])
        ll = MscatterFFT(specshape, gaus1)
        mod = Model(specshape, components=[lin, ll])
        mod.calculate()
        data[islice] = np.random.normal(mod.data, 0.01)
        lldata[islice] = gaus1.data

    msh = MultiSpectrumshape(specshape.dispersion, specshape.offset, specshape.size, shape[0], shape[1])
    s = MultiSpectrum(msh, data=data)
    ll = MscatterFFT(specshape, llspectrum=MultiSpectrum(msh, lldata))
    return s, m, q, ll


def test_nonlinear():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    gaus = Gaussian(specshape, A=1000, centre=500, fwhm=50)

    mod = Model(specshape, components=[lin, gaus])
    mod.calculate()

    s = Spectrum(specshape, data = np.random.normal(mod.data,10))
    with pytest.raises(ValueError):
        LinearFitter(s, mod)

def test_A_matrix():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)
    # gaus = Gaussian(specshape, A=1000, centre=500, fwhm=50)

    mod = Model(specshape, components=[lin])
    mod.calculate()

    s = Spectrum(specshape, data = np.random.normal(mod.data,10))
    fit = LinearFitter(s, mod)
    fit.calculate_A_matrix()
    assert fit.A_matrix.shape == (2048,2)

    #the excluding does not modify the A matrix size
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

    s = Spectrum(specshape, data = np.random.normal(mod.data,0.01))
    lin.parameters[0].setvalue(1.)
    lin.parameters[1].setvalue(1.)

    s.set_exclude_region_energy(300,500)
    fit = LinearFitter(s, mod)
    fit.perform_fit()
    assert np.abs(fit.coeff[0]-2.5) < 1e-1
    assert np.abs(fit.coeff[1]-5) < 1e-1

def test_perform_fit_convolutor():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    gaus1 = Gaussian(specshape, A=10, centre=500, fwhm=100)
    gaus1.calculate()
    ll = MscatterFFT(specshape, gaus1)


    mod = Model(specshape, components=[lin,ll])
    mod.calculate()

    s = Spectrum(specshape, data = np.random.normal(mod.data,1))
    fit = LinearFitter(s, mod)
    fit.perform_fit()

    print(fit.coeff)
    index = fit.get_param_index(lin.parameters[0])
    assert np.abs(fit.coeff[index]-2.5) < 1e-1
    index = fit.get_param_index(lin.parameters[1])
    assert np.abs(fit.coeff[index]-5) < 1e-1

def test_multifit():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    mod = Model(specshape, components=[lin])
    s, m, q = make_multispectrum(specshape)

    fit = LinearFitter(s, mod)
    fit.multi_fit()

    index = fit.get_param_index(lin.parameters[0])
    assert np.all(np.abs(fit.coeff_matrix[:,:,index]-m)<0.01)
    index = fit.get_param_index(lin.parameters[1])
    assert np.all(np.abs(fit.coeff_matrix[:,:,index]-q)<0.01)

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
    assert np.all(np.abs(fit.coeff_matrix[:,:,index]-m)<0.01)
    index = fit.get_param_index(lin.parameters[1])
    assert np.all(np.abs(fit.coeff_matrix[:,:,index]-q)<0.01)

def test_multifit_multiconvolutor():
    specshape = Spectrumshape(1, 100, 2048)
    lin = Linear(specshape, 2.5, 5)

    s, m, q, ll = make_multispectrum_convolutor(specshape)
    mod = Model(specshape, components=[lin, ll])
    fit = LinearFitter(s, mod)
    fit.multi_fit()

    index = fit.get_param_index(lin.parameters[0])
    assert np.all(np.abs(fit.coeff_matrix[:,:,index]-m)<0.01)
    index = fit.get_param_index(lin.parameters[1])
    assert np.all(np.abs(fit.coeff_matrix[:,:,index]-q)<0.1)




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










#
#
#
#
#
#
# E0 = 300e3
# alpha = 5e-3
# beta = 20e-3
# specshape = Spectrumshape(1, 100, 2048)
# Aa = 100
# # Comparison between the hydrogen and hartree slater l edge
# Ti_L1 = HSCoreLossEdge(specshape, A=Aa, E0=E0, alpha=alpha, beta=beta, element='Ti', edge='L1')
# Ti_L3 = HSCoreLossEdge(specshape, A=1.1*Aa, E0=E0, alpha=alpha, beta=beta, element='Ti', edge='L3')
# Nd_M4 = HSCoreLossEdge(specshape, A=0.5*Aa, E0=E0, alpha=alpha, beta=beta, element='Nd', edge='M4')
# Nd_M5 = HSCoreLossEdge(specshape, A=0.6*Aa, E0=E0, alpha=alpha, beta=beta, element='Nd', edge='M5')
# O_K = HSCoreLossEdge(specshape, A=1.5*Aa, E0=E0, alpha=alpha, beta=beta, element='O', edge='K')
# Si_K = HSCoreLossEdge(specshape, A=2*Aa, E0=E0, alpha=alpha, beta=beta, element='Si', edge='K')
# Ni_L1 = HSCoreLossEdge(specshape, A=1.8*Aa, E0=E0, alpha=alpha, beta=beta, element='Ni', edge='L1')
# Ni_L3 = HSCoreLossEdge(specshape, A=3.5*Aa, E0=E0, alpha=alpha, beta=beta, element='Ni', edge='L3')
# pwbg = fbg = FastBG(specshape, A1=50000, A2=50000)
#
#
# mod = Model(specshape)
# mod.addcomponent(Ti_L1)
# mod.addcomponent(Ti_L3)
# mod.addcomponent(Nd_M4)
# mod.addcomponent(Nd_M5)
# mod.addcomponent(Si_K)
# mod.addcomponent(Ni_L1)
# mod.addcomponent(Ni_L3)
# mod.addcomponent(O_K)
# mod.addcomponent(pwbg)
#
# s = Spectrum(specshape, data = (mod.data)) #this is a ground truth modelled spectrum for which we
# #know the parameters, lets see if we can find them back by fitting
# m=mod.copy()
#
# fit = LinearFitter(s, m)
# # cmod = m.copy()
# # for pars, j in enumerate(cmod.getfreelinparameters()):
# #     print(pars)
# A = fit.A_matrix
# y = fit.y_vector
#
# fit.perform_fit()
# fit.set_fit_values()
#
#
# fit.set_information_matrix()
# information_matrix = fit.information_matrix
#
# plt.figure()
# plt.imshow(information_matrix)
#
# #self implemented
# ATA = np.dot(np.transpose(A),A)
# B = np.dot(np.linalg.inv(ATA), np.transpose(A))
# result = np.dot(B,y)
#
# print('results')
# groundthruth=mod.getfreelinparameters()
# estimated=m.getfreelinparameters()
#
# maxdeviation=1e-10 #
# for i in range(mod.getnumfreeparameters()):
#     g=groundthruth[i].getvalue()
#     e=estimated[i].getvalue()
#     deviation=100*(g-e)/g
#     print(groundthruth[i].getname(),':',g, 'estimated:',e, 'deviation:',deviation,'%')
#     assert(deviation<maxdeviation)
# s.plot(plt)
# m.plot(plt)


























