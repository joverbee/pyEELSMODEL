import sys

sys.path.append("..")  # Adds higher directory to python modules path.
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.component import Component

from pyEELSMODEL.components.linear import Linear
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.plasmon import Plasmon
from pyEELSMODEL.components.fast_background import FastBG2
from pyEELSMODEL.components.lorentzian import Lorentzian
from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.CLedge.hydrogen_coreloss_edge import HydrogenicCoreLossEdge
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
import matplotlib.pyplot as plt
import pytest
import numpy as np

#note that changing the loglevel on anaconde requires at least a kernel restart
import logging
#logging.basicConfig(level=logging.DEBUG) #detailed debug reporting
#logging.basicConfig(level=logging.INFO) #show info on normal working of code
logging.basicConfig(level=logging.WARNING) #only show warning that help user to avoid an issue



def test_init_component():
    specshape = Spectrumshape(1, 200, 1024)
    mod = Model(specshape)
    assert mod.energy_axis[0] == 200
    assert mod.energy_axis[-1] == 1223
    assert mod.getnumparameters()==0
    assert mod.getnumfreeparameters()==0  

def test_add_component():
    specshape = Spectrumshape(1, 200, 1024)
    mod = Model(specshape)
    c1 = Component(specshape)
    c2 = Component(specshape)
    assert mod.getnumcomponents() == 0
    mod.addcomponent(c1)
    assert mod.getnumcomponents() == 1
    mod.addcomponent(c2)
    assert mod.getnumcomponents() == 2

    with pytest.raises(TypeError):
        mod.addcomponent('not valid')

def test_remove_component():
    specshape = Spectrumshape(1, 200, 1024)
    mod = Model(specshape)
    c1 = Component(specshape)
    c2 = Component(specshape)
    c3 = Component(specshape)  
    mod = Model(specshape)
    mod.addcomponent(c1)
    mod.addcomponent(c2)
    assert mod.getnumcomponents() == 2
    mod.removecomponent(c1)
    assert mod.getnumcomponents() == 1
    assert mod.components[0] == c2
    with pytest.raises(ValueError):
        mod.removecomponent(c3)
    with pytest.raises(TypeError):
        mod.removecomponent('not valid')

def test_is_linear():
    specshape = Spectrumshape(1, 200, 1024)
    mod = Model(specshape)
    c1 = Linear(specshape, 1, 1)
    c2 = Gaussian(specshape, 1,0,1)
    mod.addcomponent(c1)
    assert mod.islinear()
    mod.addcomponent(c2)
    assert not mod.islinear()

def test_get_num_parameters():
    specshape = Spectrumshape(1, 200, 1024)
    c1 = Linear(specshape, 1, 1)
    c2 = Gaussian(specshape, 1,0,1)
    mod = Model(specshape)
    mod.addcomponent(c1)
    assert mod.getnumparameters() == 2
    mod.addcomponent(c2)
    assert mod.getnumparameters() == 5
    mod.removecomponent(c1)
    assert mod.getnumparameters() == 3


def test_get_num_free_parameters():
    specshape = Spectrumshape(1, 200, 1024)
    c1 = Linear(specshape, 1, 1)
    mod = Model(specshape)
    mod.addcomponent(c1)
    assert mod.getnumfreeparameters() == 2

    c1.parameters[0].setchangeable(False)
    assert mod.getnumfreeparameters() == 1

    c1.parameters[1].setchangeable(False)
    assert mod.getnumfreeparameters() == 0

    c1.parameters[0].setchangeable(True)
    assert mod.getnumfreeparameters() == 1

def test_get_free_linear_parameters():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    c1 = Linear(specshape, 1 / 1024, 1)
    c2 = Gaussian(specshape, 1, 500, 100)
    mod = Model(specshape)
    mod.addcomponent(c1)
    assert len(mod.getfreelinparameters()) == 2
    assert len(mod.getfreenonlinparameters()) == 0
    mod.addcomponent(c2)
    assert len(mod.getfreelinparameters()) == 3
    assert len(mod.getfreenonlinparameters()) == 2

# def test_coupled_parameters():

def test_getgradient():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    c1 = Linear(specshape, 1/1024, 1)
    mod = Model(specshape)
    mod.addcomponent(c1)
    deriv = mod.getgradient(mod.getfreeparameters()[1])
    comparison = deriv == np.ones(deriv.size)
    assert comparison.all()


# def test_plotting():
#     specshape = Spectrumshape(1, 200, 1024)
#     s0 = Spectrum(specshape)
#     c1 = Linear(specshape, 1/1024, 1)
#     c2 = Gaussian(specshape, 1,500,100)
#     c4 = FastBG2(specshape,  A1=5, A2=2)
#     c6 = PowerLaw(specshape, A=10, r=3)
#     c7 = HydrogenicCoreLossEdge(specshape,A=1e27, E0=300e5, alpha=1e-3, beta=20e-3, element='C', edge='K')
#     mod = Model(specshape)
#     mod.addcomponent(c1)
#     mod.addcomponent(c2)
#     mod.addcomponent(c4)
#     mod.addcomponent(c6)
#     mod.addcomponent(c7)
#     mod.calculate()
#     mod.plot()
#     #test convolution with a low loss spectrum
#     specshapell = Spectrumshape(1, -20, 1024)
#     cll = Plasmon(specshapell,  A=10, Ep=15, Ew=4, n=3, tlambda=0.1, beta=10e-3, E0=300e3)
#     czl = Lorentzian(specshape, A=10, centre=0, fwhm=2)
#     modll = Model(specshapell)
#     modll.addcomponent(cll)
#     modll.addcomponent(czl)
#     plt.figure()
#     modll.plot(plt)
#
#     llspectrum=modll #note that a model is a spectrum so you can use it as a synthetic LL spectrum
#     #by lack of an experimental LL spectrum for now
#
#     c8= MscatterFFT(specshape, llspectrum)
#     mod.addcomponent(c8)
#     mod.calculate()
#     mod.plot()

def test_coupling_parameters():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    c1 = Linear(specshape, 0, 1)
    c2 = Linear(specshape, 0, 4.5)
    c3 = Linear(specshape, 0, 1.5)
    c4 = Linear(specshape, 0, 2.)

    c4.parameters[1].couple(c1.parameters[1], fraction=2.)
    c3.parameters[1].couple(c2.parameters[1], fraction=1/3)


    mod = Model(specshape, components=[c1,c2,c3,c4])
    mod.order_coupled_components()

    coo = [c3,c4,c1,c2]
    for comp0, comp1 in zip(mod.components, coo):
        assert comp0.parameters[1].getvalue() == comp1.parameters[1].getvalue()

def test_calculate_coupling_parameters():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    c1 = Linear(specshape, 0, 1)
    c2 = Linear(specshape, 0, 4.5)
    c3 = Linear(specshape, 0, 1.5)
    c4 = Linear(specshape, 0, 2.)

    c4.parameters[1].couple(c1.parameters[1], fraction=2.)
    c3.parameters[1].couple(c2.parameters[1], fraction=1/3)


    mod = Model(specshape, components=[c1,c2,c3,c4])
    c1.parameters[1].setvalue(4.)
    c2.parameters[1].setvalue(3.)
    mod.calculate()

    assert c3.parameters[1].getvalue() == 1.
    assert c4.parameters[1].getvalue() == 8.

    # plt.figure()
    # plt.plot(c1.data)
    # plt.plot(c2.data)
    # plt.plot(c3.data)
    # plt.plot(c4.data)



def main():
    test_init_component()
    test_add_component()
    test_remove_component()
    test_is_linear()
    test_get_num_parameters()
    test_get_num_free_parameters()
    test_get_free_linear_parameters()
    test_getgradient()
    # test_plotting()
    test_coupling_parameters()
    test_calculate_coupling_parameters()

if __name__ == "__main__":
    main()