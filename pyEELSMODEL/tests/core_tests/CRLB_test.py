"""
The scripts tests the CRLB with a linear function
"""
from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.spectrum import Spectrumshape,Spectrum
from pyEELSMODEL.fitters.minimizefitter import MinimizeFitter
from pyEELSMODEL.components.linear import Linear
from pyEELSMODEL.components.CLedge.dummymodel import DummyEdge

import numpy as np


def test_CRLB_linear():
    """
    Test to validate if the CRLB is found for a linear function (x+1).
    For this it is possible to determine easily the theoretical fisher 
    information matrix. 
    The validity of the test does not ensure that no bugs are present. 

    """
    specshape = Spectrumshape(1, 100, 1024)
    pol = Linear(specshape,m=1,q=1)
    mod = Model(specshape, components=[pol])
    mod.calculate()
    sig = np.copy(mod.data)
    
    nsamples = 100
    coeff_matrix = np.zeros((2, nsamples))
    std_matrix = np.zeros(coeff_matrix.shape)
    
    for i in range(nsamples):
        pol.parameters[0].setvalue(1)
        pol.parameters[1].setvalue(1)
    
        ndata = np.random.poisson(sig)
        s = Spectrum(specshape, data=ndata)
        fit = MinimizeFitter(s, mod)
        fit.usegradients = False
        fit.perform_fit()
        coeff_matrix[:,i] = fit.coeff
        fit.set_information_matrix()
        std_matrix[:,i] = np.sqrt(np.linalg.inv(fit.information_matrix).diagonal())
    
    
    
    #theoretical fischer matrix for a linear function f(x) = x + 1
    theo_fischer = np.zeros((2,2))
    theo_fischer[0,0] = np.sum(s.energy_axis**2/(s.energy_axis+1))
    theo_fischer[1,1] = np.sum(1/(s.energy_axis+1))
    theo_fischer[1,0] = np.sum(s.energy_axis/(s.energy_axis+1))
    theo_fischer[0,1] = np.sum(s.energy_axis/(s.energy_axis+1))
    
    #the calculated fischermatrix using model
    pol.parameters[0].setvalue(1)
    pol.parameters[1].setvalue(1)
    fischer = fit.information_matrix
    
    
    
    #test: the theoretical and calculated fischer matrix should be similar
    assert np.all(np.abs((fischer - theo_fischer)/(fischer+theo_fischer))<0.01)
    
    #test: the standard deviation on the poisson simulated noise should be larger
    # then the fischer information results
    std_a = np.std(coeff_matrix[0])
    crlb_a = np.sqrt(fit.covariance_matrix[0,0])
    std_b = np.std(coeff_matrix[1])
    crlb_b = np.sqrt(fit.covariance_matrix[1,1])
    
    assert np.abs(std_a-crlb_a)/(std_a+crlb_a)<0.1
    assert np.abs(std_b-crlb_b)/(std_b+crlb_b)<0.1


def test_CRLB_coreloss():
    specshape = Spectrumshape(1, 100, 1024)
    edge = DummyEdge(specshape, A=50, element='C', edge='K1')
    bg = PowerLaw(specshape, A=1e3, r=2.5)
    mod = Model(specshape, components=[bg, edge])
    mod.calculate()
    sig = np.copy(mod.data)


    nsamples = 100
    coeff_matrix = np.zeros((3, nsamples))
    
    for i in range(nsamples):
        bg.parameters[0].setvalue(1e3)
        bg.parameters[1].setvalue(2.5)
        edge.parameters[0].setvalue(50)
    
        ndata = np.random.poisson(sig)
        s = Spectrum(specshape, data=ndata)
        fit = MinimizeFitter(s, mod)
        fit.usegradients = False
        fit.perform_fit()
        coeff_matrix[:,i] = fit.coeff
    fit.set_information_matrix()


    std_a = np.std(coeff_matrix[0])
    crlb_a = np.sqrt(fit.covariance_matrix[0,0])
    std_b = np.std(coeff_matrix[1])
    crlb_b = np.sqrt(fit.covariance_matrix[1,1])
    std_c = np.std(coeff_matrix[2])
    crlb_c = np.sqrt(fit.covariance_matrix[2,2])
    
    
    assert np.abs(std_a-crlb_a)/(std_a+crlb_a)<0.1
    assert np.abs(std_b-crlb_b)/(std_b+crlb_b)<0.1
    assert np.abs(std_c-crlb_c)/(std_c+crlb_c)<0.1

def main():
    test_CRLB_linear()
    test_CRLB_coreloss()


if __name__ == "__main__":
    main()


