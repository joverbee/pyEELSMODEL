import sys

sys.path.append("..")  # Adds higher directory to python modules path.
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model
import numpy as np

import matplotlib.pyplot as plt
from pyEELSMODEL.components.CLedge.hs_coreloss_edge import HSCoreLossEdge
from pyEELSMODEL.components.CLedge.hydrogen_coreloss_edge import HydrogenicCoreLossEdge
E0 = 300e3
alpha = 0.02
beta = 100e-3
specshape = Spectrumshape(1, 100, 2048)
Aa = 100
# Comparison between the hydrogen and hartree slater l edge
O_K = HSCoreLossEdge(specshape, A=1, E0=E0, alpha=alpha, beta=beta, element='O', edge='K')
O_K_hy = HydrogenicCoreLossEdge(specshape,A=1,E0=E0, alpha=alpha, beta=beta, element='O', edge='K')

O_K.calculate()
O_K_hy.calculate()

plt.figure()
plt.plot(O_K.energy_axis, O_K.data)
plt.plot(O_K.energy_axis, O_K_hy.data)









