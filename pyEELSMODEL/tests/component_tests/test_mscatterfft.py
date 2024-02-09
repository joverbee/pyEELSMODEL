import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.components.CLedge.hs_coreloss_edge import HSCoreLossEdge
from pyEELSMODEL.core.spectrum import Spectrumshape, Spectrum
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.components.gaussian import Gaussian
import pytest


specshape = Spectrumshape(1, -100, 1024)
C = HSCoreLossEdge(specshape, A=1, E0=300, alpha=1e-9, beta=1e-3, element='C', edge='K')
C.calculate()

spec0 = Spectrum(specshape, data=C.data)

gaus = Gaussian(specshape, A=1, centre=spec0.energy_axis[int(spec0.size/2)], fwhm=20)
gaus.calculate()
spec1 = Spectrum(specshape, data=gaus.data)

llcomp = MscatterFFT(specshape, spec1)
llcomp.data = C.data
llcomp.calculate()

plt.figure()
plt.plot(llcomp.data)

plt.figure()
plt.plot(tophat)








