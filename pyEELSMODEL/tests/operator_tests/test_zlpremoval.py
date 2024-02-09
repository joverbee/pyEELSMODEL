from pyEELSMODEL.operators.zlpremoval import ZLPRemoval
from pyEELSMODEL.operators.multispectrumvisualizer import MultiSpectrumVisualizer

from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.model import Model

from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.lorentzian import Lorentzian
from pyEELSMODEL.components.voigt import Voigt
from pyEELSMODEL.components.plasmon import Plasmon

import matplotlib.pyplot as plt
import numpy as np

specshape = Spectrumshape(0.32, -20, 1024)
zlp = Lorentzian(specshape, A=1, centre=0, fwhm=1)
zlp.calculate()
plasmon = Plasmon(specshape, A=50, Ep=20, Ew=10, n=5, tlambda=0.3, E0=80e3, beta=50e-3)
plasmon.calculate()

spectrum = zlp + plasmon
spectrum.data += np.random.normal(0, 0.005, size=spectrum.size)
spectrum.plot()

# rem = ZLPRemoval(spectrum, model_type='Lorentzian', signal_range=(-12.5,3))
rem = ZLPRemoval(spectrum, model_type='Lorentzian')

rem.mirrored_zlp()
rem.fit_zlp()

rem.mirror_inelastic.plot()
rem.mirror_zlp.plot()

rem.show_mirror_result()

rem.show_fit_result()
rem.inelastic.plot()


Eps = [20, 25, 30]
tlambdas = [0.3, 0.6, 0.9]
nper = 10

ndata = np.zeros((len(Eps)*nper, zlp.size))
cc = 0
for ii in range(len(Eps)):
    plasmon = Plasmon(specshape, A=50, Ep=Eps[ii], Ew=10, n=5, 
                      tlambda=tlambdas[ii], E0=80e3, beta=50e-3)
    plasmon.calculate()
    for jj in range(nper):
        spectrum = zlp + plasmon
        ndata[cc] = spectrum.data + np.random.normal(0, 0.005, size=spectrum.size)
        cc += 1

mspec = MultiSpectrum.from_numpy(ndata[:,np.newaxis,:], zlp.energy_axis)

MultiSpectrumVisualizer([mspec])


rem = ZLPRemoval(mspec, model_type='Lorentzian', signal_range=(-12.5,3))
rem.fit_zlp()
rem.mirrored_zlp()

rem.show_fit_result()
rem.inelastic.plot()

plt.figure()
plt.imshow(np.squeeze(rem.inelastic.multidata), aspect='auto')

plt.figure()
plt.imshow(np.squeeze(rem.mirror_inelastic.multidata), aspect='auto')

plt.figure()
plt.imshow(np.squeeze(rem.mirror_zlp.multidata), aspect='auto')


plt.figure()
plt.plot(rem.inelastic.integrate((0,1024)))















