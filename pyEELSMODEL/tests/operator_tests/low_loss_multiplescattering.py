import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import matplotlib.pyplot as plt
import pytest



from pyEELSMODEL.components.CLedge.dummymodel import DummyEdge
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.plasmon import Plasmon
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

from pyEELSMODEL.components.fixedpattern import FixedPattern
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model
from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.voigt import Voigt

from pyEELSMODEL.operators.deconvolutions.richardson_lucy import RichardsonLucyDeconvolution
from pyEELSMODEL.operators.deconvolutions.gaussianmodifier import GaussianModifier
from pyEELSMODEL.operators.deconvolutions.wienerfilter import WienerFilter
from scipy import signal

def fft_phys(data):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data)))

def ifft_phys(data):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data)))

def convolution(zlp, se):
    co1 = np.argmax(se)
    J_ = signal.fftconvolve(zlp/zlp.sum(), se, mode='full')
    co2 = np.argmax(J_)
    J = np.roll(J_, -co1)

    return J[:se.size]
    

alpha = 1e-9
beta = 150e-3
E0 = 200e3
specshape = Spectrumshape(0.25, -50, 2048)

I0 = 1e4
t_lambda = 0.3
A_zlp = I0 * np.exp(-t_lambda)
A_1 = A_zlp * t_lambda


sigm= 3


g1 = Gaussian(specshape, 2*A_1/3 , 30, sigm)
g2 =  Gaussian(specshape, A_1/3, 48, sigm)
g1.calculate()
g2.calculate()

S_e = g1 + g2
S_e.plot()

zlp = Gaussian(specshape, A_zlp, 0,sigm)
zlp.calculate()
zlp.plot()


print(S_e.data.sum()/zlp.data.sum())

co = np.argmax(zlp.data)
J1_e = convolution(zlp.data, S_e.data)
print(J1_e.sum()/zlp.data.sum())

J2_e = convolution(J1_e, S_e.data)*(0.5*t_lambda**2)
J3_e = convolution(J2_e, S_e.data)*(1/6*t_lambda**3)

plt.figure()
plt.plot(S_e.data)
plt.plot(J1_e)
plt.plot(J2_e)
plt.plot(J3_e)
plt.plot(J1_e + J2_e + J3_e)
plt.plot(zlp.data)


print(J2_e.sum()/zlp.data.sum())

J_e = np.roll(J1_e + J2_e + J3_e, co) + zlp.data
plt.figure()
plt.plot(J1_e + J2_e + J3_e)
plt.plot(zlp.data)
plt.plot(J_e)

J_f = fft_phys(J_e)

fft_se = fft_phys(S_e.data)

plt.figure()
plt.plot(np.abs(fft_phys(zlp.data)))

j_f = fft_phys(zlp.data) * np.exp(fft_phys(S_e.data)/I0)
j_f_ = fft_phys(zlp.data) * (1 + fft_se/I0 + fft_se**2/(2*I0**2) + fft_se**3/(6*I0**3))
# j_f_ = fft_phys(zlp.data) * (1 + fft_se/I0)

plt.figure()
plt.plot(np.abs(j_f_))
plt.plot(np.abs(j_f))
plt.plot(np.abs(J_f))


plt.plot(np.abs(J_f))

co= np.argmax(zlp.data)
j_e = np.roll(ifft_phys(j_f), -co)
plt.figure()
plt.plot(j_e/j_e.max())
plt.plot(J_e/J_e.max())

plt.plot(zlp.data + S_e.data)


plt.figure()
plt.plot(np.abs(ifft_phys(j_f_)))








































