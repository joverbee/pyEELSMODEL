from pyEELSMODEL.operators.simulator.coreloss_simulator import CoreLossSimulator
from pyEELSMODEL.operators.quantification.elemental_quantification import ElementalQuantification
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval

from pyEELSMODEL.core.multispectrum import MultiSpectrumshape, MultiSpectrum
import pyEELSMODEL.api as em

import numpy as np
import matplotlib.pyplot as plt
import os

from pyEELSMODEL.components.CLedge.kohl_coreloss_edgecombined import KohlLossEdgeCombined
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.gdoslin import GDOSLin

specshape = em.Spectrumshape(1, 200, 1024)


savedir = r'C:\Users\DJannis\PycharmProjects\project\pyEELSMODEL\examples\data'

Ce3 =  em.Spectrum.load(r'Z:\emattitan3\Daen\2023\2023_04_13\Ce3+4+\3+ref.msa')
Ce3.offset += 8-1-1
Ce3.data = Ce3.data/Ce3.data.max()
Ce3.name='Ce 3+'
Ce3.dispersion = Ce3.dispersion*1.
Ce3.exclude = np.zeros(Ce3.size, dtype='bool')
Ce4 =  em.Spectrum.load(r'Z:\emattitan3\Daen\2023\2023_04_13\Ce3+4+\4+ref.msa')
Ce4.dispersion = Ce4.dispersion*1.
Ce4.offset += 10.5-1
Ce4.data = Ce4.data/Ce4.data.max()
Ce4.name='Ce 4+'
Ce4.exclude = np.zeros(Ce4.size, dtype='bool')


Ce3.plot()
Ce4.plot()


plt.figure()
plt.plot(Ce3.energy_axis, Ce3.data)
plt.plot(Ce4.energy_axis, Ce4.data)

E0=300e3
alpha=1e-9
beta=20e3


comp = ZezhongCoreLossEdgeCombined(Ce3.get_spectrumshape(), 1, E0, alpha, beta, 'Ce', 'M')
fine = GDOSLin.gdoslin_from_edge(Ce3.get_spectrumshape(), comp, ewidth=35,degree=90,
                                 interpolationtype='cubic')

modCe3 = em.Model(Ce3.get_spectrumshape(), [comp, fine])

fitCe3 = em.LinearFitter(Ce3, modCe3)
fitCe3.perform_fit()
fitCe3.plot()

specshape = em.Spectrumshape(0.05, 860, 4096)


comp_ = ZezhongCoreLossEdgeCombined(specshape, 1, E0, alpha, beta, 'Ce', 'M')
comp_.plot()

fine_ = GDOSLin.gdoslin_from_edge(specshape, comp_, ewidth=30,degree=90,
                                 interpolationtype='cubic')

for para, para_ in zip(fine.parameters[2:], fine_.parameters[2:]):
    value = para.getvalue()/comp.parameters[0].getvalue()
    para_.setvalue(value)
    

fine_.calculate()

s = comp_ + fine_
s_ce3 = em.Spectrum.from_numpy(s.data, comp_.energy_axis)

# savename = os.path.join(savedir, 'ce3_edge.hdf5')
# s_ce3.save_hdf5(savename)


comp = ZezhongCoreLossEdgeCombined(Ce4.get_spectrumshape(), 1, E0, alpha, beta, 'Ce', 'M')
fine = GDOSLin.gdoslin_from_edge(Ce4.get_spectrumshape(), comp, ewidth=40,degree=90,
                                 interpolationtype='cubic')

modCe4 = em.Model(Ce4.get_spectrumshape(), [comp, fine])

fitCe4 = em.LinearFitter(Ce4, modCe4)
fitCe4.perform_fit()
fitCe4.plot()

comp_ = ZezhongCoreLossEdgeCombined(specshape, 1, E0, alpha, beta, 'Ce', 'M')
comp_.plot()

fine_ = GDOSLin.gdoslin_from_edge(specshape, comp_, ewidth=30,degree=90,
                                 interpolationtype='cubic')

for para, para_ in zip(fine.parameters[2:], fine_.parameters[2:]):
    value = para.getvalue()/comp.parameters[0].getvalue()
    para_.setvalue(3*value)
    

fine_.calculate()

s = comp_ + fine_
s_ce4 = em.Spectrum.from_numpy(s.data, comp_.energy_axis)

savename = os.path.join(savedir, 'ce4_edge.hdf5')
s_ce4.save_hdf5(savename, overwrite=True)

plt.figure()
plt.plot(s_ce4.energy_axis, s_ce4.data)
plt.plot(s_ce3.energy_axis, s_ce3.data)





















