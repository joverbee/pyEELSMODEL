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



alpha = 1e-9
beta = 150e-3
E0 = 200e3
specshape = Spectrumshape(0.5, -10, 2048)

plasmon = Plasmon(specshape, A=1, Ep=24, Ew=5, n=3, tlambda=1., beta=beta, E0=E0)
plasmon.calculate()
plasmon.plot()

# specshape0 = Spectrumshape(np.diff(energy_axis)[0],-10, energy_axis.size)
# zlp = ZeroLoss(specshape, 1, 0, 2)
# zlp = Gaussian(specshape, 1, 0, 2)
zlp = Voigt(specshape, 1, 0, 0.2, 1)
zlp.calculate()
zlp.plot()


hl_shape = Spectrumshape(specshape.dispersion, 100, 2048)
compCK = DummyEdge(hl_shape, 1e2, E0, alpha, beta, 'C', 'K1')
compCK.calculate()


bkg = PowerLaw(hl_shape, A=1e3, r=3)

cte = 1e-5
SE = Spectrum(specshape, data = zlp.data+plasmon.data+cte*compCK.data)
SE.data = SE.data*5e2
SE.plot()



ll = SE.get_interval((-100,120))

llcomp = MscatterFFT(hl_shape, ll)
mod = Model(hl_shape, components=[compCK, llcomp])
mod.calculate()
mod.plot()



rich = RichardsonLucyDeconvolution(mod, ll)
rich.iterations = 10
r_res = rich.deconvolve()

gm = GaussianModifier(mod, ll, factor=4)
gm_res = gm.deconvolve()

wf = WienerFilter(mod, ll, iterations=5)
wf_res = wf.deconvolve()



fig, ax = plt.subplots()
ax.plot(compCK.energy_axis, compCK.data)
ax.plot(r_res.energy_axis, r_res.data)
ax.plot(gm_res.energy_axis, gm_res.data)













































































