# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:20:33 2024

@author: DJannis
"""

from pyEELSMODEL.operators.simulator.coreloss_simulator import CoreLossSimulator
from pyEELSMODEL.operators.quantification.elemental_quantification import ElementalQuantification
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval
from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.linear_background import LinearBG
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import ZezhongCoreLossEdgeCombined


from pyEELSMODEL.core.multispectrum import MultiSpectrumshape, MultiSpectrum
import pyEELSMODEL.api as em

import numpy as np
import matplotlib.pyplot as plt



elements = ['C', 'O']
edges = ['K', 'K']

xsize = 100
ysize = 100


msh = MultiSpectrumshape(0.5, 200, 2048, xsize, ysize)
sh = msh.getspectrumshape()
bg = LinearBG(sh, rlist=np.linspace(1,5,4))
E0 = 300e3 #V
alpha=1e-9 #rad
beta=20e-3 #rad

        # The components of the edges
comp_elements = []
As = [1,2]
for elem, edge, A in zip(elements, edges, As):
    cte = 0.01
    comp = ZezhongCoreLossEdgeCombined(sh, A*cte,E0,alpha,beta, elem, edge)
    comp_elements.append(comp)
    
mod = em.Model(sh, components=[bg]+comp_elements)

mod.calculate()
mod.plot()

eels_data = np.zeros((xsize, ysize, msh.Esize))
ne = np.exp(np.linspace(1, 10, xsize))
gain = 5
for ii in range(xsize):
    ndata = np.copy(mod.data) * ne[ii]
    for jj in range(ysize):
        eels_data[ii,jj] = gain*np.random.poisson(ndata)

s = MultiSpectrum(msh, data=eels_data)
s.pppc = 1/gain
em.MultiSpectrumVisualizer([s])




bg = LinearBG(sh, rlist=np.linspace(1,5,4))

comp_elements = []
for elem, edge in zip(elements, edges):
    comp = ZezhongCoreLossEdgeCombined(sh, 1,E0,alpha,beta, elem, edge)
    comp_elements.append(comp)
    

mod = em.Model(sh, components=[bg]+comp_elements)

fit = em.LinearFitter(s, mod)
fit.multi_fit()

fig, maps, names = fit.show_map_result(comp_elements)

crlb_C = fit.CRLB_map(comp_elements[0].parameters[0]) #comp_elements[3].parameters[0]: amplitude of iron edge
crlb_N = fit.CRLB_map(comp_elements[1].parameters[0]) #comp_elements[3].parameters[0]: amplitude of iron edge


fig, ax = plt.subplots(2,2)
ax[0,0].plot(ne/s.pppc, s.pppc*maps[0].mean(1)/ne, color='red')
ax[0,0].axhline(As[0]*cte, linestyle='dotted', color='black', label='Ground truth')
ax[0,0].set_ylabel('Average value/Total counts')

ax[0,1].plot(ne/s.pppc, s.pppc*maps[1].mean(1)/ne, color='blue')
ax[0,1].axhline(As[1]*cte, linestyle='dotted', color='black', label='Ground truth')
# ax[0,1].set_xscale('log')

ax[1,0].plot(ne/s.pppc, crlb_C.mean(1)/ne, color='black', label='CRLB')
ax[1,0].plot(ne/s.pppc, maps[0].std(1)/ne, color='red', label='std C')
# ax[1,0].set_xscale('log')
ax[1,1].plot(gain*ne, crlb_N.mean(1)/ne, color='black', label='CRLB')
ax[1,1].plot(gain*ne, maps[1].std(1)/ne, color='blue', label='std N')
# ax[1,1].set_xscale('log')
ax[1,0].set_ylabel('Standard deviation/Total counts')
ax[1,0].set_xlabel(r'Total counts')
ax[1,1].set_xlabel(r'Total counts')

for axe in ax.flatten():
    axe.set_xscale('log')
    axe.legend()


plt.figure()
plt.plot(s.multidata[0,0]*s.pppc)


plt.figure()
plt.plot(ne, maps[0].std(1)/crlb_C.mean(1), color='blue', label='std N')

plt.figure()
plt.plot(ne, maps[1].std(1)/crlb_N.mean(1), color='blue', label='std N')













