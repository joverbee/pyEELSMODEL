"""
The scripts tests the CRLB with a linear function
"""
from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.components.CLedge.hs_coreloss_edge import HSCoreLossEdge
from pyEELSMODEL.components.polynomial import Polynomial
from pyEELSMODEL.components.gdoslin import GDOSLin
from pyEELSMODEL.core.spectrum import Spectrumshape,Spectrum
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval
from pyEELSMODEL.fitters.minimizefitter import MinimizeFitter

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import time
import os

#check the CRLB for linear fit
specshape = Spectrumshape(1, 100, 1024)
pol = Polynomial(specshape,order=2)
mod = Model(specshape, components=[pol])
mod.calculate()
sig = np.copy(mod.data)

nsamples = 100
spec = []
confidence = np.zeros(nsamples)
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
    std_matrix[:,i] = np.sqrt(np.linalg.inv(fit.get_information_matrix()).diagonal())



#theoretical fischer matrix for a linear function f(x) = x + 1
theo_fischer = np.zeros((2,2))
theo_fischer[0,0] = np.sum(s.energy_axis**2/(s.energy_axis+1))
theo_fischer[1,1] = np.sum(1/(s.energy_axis+1))
theo_fischer[1,0] = np.sum(s.energy_axis/(s.energy_axis+1))
theo_fischer[0,1] = np.sum(s.energy_axis/(s.energy_axis+1))

#the calculated fischermatrix using model
pol.parameters[0].setvalue(1)
pol.parameters[1].setvalue(1)
fischer = fit.get_information_matrix()



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

#
# #check the CRLB for the GDOSlin
# specshape = Spectrumshape(1, 100, 1024)
# gdos = GDOSLin(specshape, 200, 500, 20)
# values = []
# for param in gdos.parameters[2:]:
#     value = np.random.randint(1000)
#     param.setvalue(value)
#     values.append(value)
#
# mod = Model(specshape, components=[gdos])
# mod.calculate()
# sig = np.copy(mod.data)
#
#
# nsamples = 100
# spec = []
# confidence = np.zeros(nsamples)
# coeff_matrix = np.zeros((len(gdos.parameters[2:]), nsamples))
# std_matrix = np.zeros(coeff_matrix.shape)
#
# for i in range(nsamples):
#     for param, value in zip(gdos.parameters[2:], values):
#         param.setvalue(value)
#
#     ndata = np.random.poisson(sig)
#     s = Spectrum(specshape, data=ndata)
#     fit = MinimizeFitter(s, mod)
#     # fit.usegradients = False
#     fit.perform_fit()
#     coeff_matrix[:,i] = fit.coeff
#     fit.set_information_matrix()
#     std_matrix[:,i] = np.sqrt(1/fit.get_information_matrix().diagonal())


