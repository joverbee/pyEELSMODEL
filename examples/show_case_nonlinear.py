# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:11:42 2024

@author: DJannis
"""

from pyEELSMODEL.operators.simulator.coreloss_simulator import CoreLossSimulator
from pyEELSMODEL.operators.quantification.elemental_quantification import ElementalQuantification
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval
from pyEELSMODEL.components.linear_background import LinearBG

from pyEELSMODEL.core.multispectrum import MultiSpectrumshape, MultiSpectrum
import pyEELSMODEL.api as em
from pyEELSMODEL.components.lorentzian import Lorentzian

import numpy as np
import matplotlib.pyplot as plt


def ti_4(specshape, As, cs, fwhms, settings):
    E0 = settings[0]
    alpha = settings[1]
    beta = settings[2]    
    
    element = 'Ti'
    edge = 'L'
    
    bg = LinearBG(specshape, rlist=np.linspace(1,5,4))

    comp = em.ZezhongCoreLossEdgeCombined(specshape, 0.5, E0, alpha, beta, element, edge)
    value = 1/comp.data.max()
    comp.parameters[0].setvalue(value)
    
    lorentz = []
    for ii in range(len(As)):
        lorentz.append(Lorentzian(specshape, As[ii], cs[ii], fwhms[ii]))
    
    components = [bg, comp] + lorentz
    mod = em.Model(specshape, components)
    mod.calculate()
    return mod


def change_params(mod, As, cs, fwhms):
    #first two components are the background and atomic edge
    for ii, comp in enumerate(mod.components[2:]):
        comp.parameters[0].setvalue(As[ii])
        comp.parameters[1].setvalue(cs[ii])
        comp.parameters[2].setvalue(fwhms[ii])
    return mod
    

xsize = 50
ysize = 50 

msh = MultiSpectrumshape(0.1, 420, 1024, xsize, ysize)
specshape = msh.getspectrumshape()
settings = (300e3, 1e-9, 20e-3)


As = [4,6,4,6]
fwhms = [0.5,0.8,1,1.3]

cs1 = [456, 459, 462, 464]
cs2 = [457, 459, 462.5, 464]

mod1 = ti_4(specshape, As, cs1, fwhms, settings)
mod2 = ti_4(specshape, As, cs2, fwhms, settings)

plt.figure()
plt.plot(mod1.data)
plt.plot(mod2.data)

scan_size = (xsize, ysize)
As_map = np.ones(scan_size+(4,))
fwhms_map = np.ones(scan_size+(4,))
cs_map = np.ones(scan_size+(4,))

start=10
end=40

XX, YY = np.meshgrid(np.arange(xsize)-start, np.arange(ysize)-start)
m1 = (cs2[0]-cs1[0])/(end-start)
m1 = (cs2[0]-cs1[0])/(end-start)

for ii in range(len(As)):
    As_map[:,:,ii] = As[ii]
    fwhms_map[:,:,ii] = fwhms[ii]
    cs_map[:start,:,ii] = cs1[ii]
    cs_map[end:,:,ii] = cs2[ii]
    
    m = (cs2[ii] - cs1[ii])/(end-start)
    cs_map[start:end,:,ii] = m*YY[start:end,:]+cs1[ii]
    

fig, ax = plt.subplots(3,4)
for ii in range(As_map.shape[-1]):
    ax[0,ii].imshow(As_map[:,:,ii])
    ax[1,ii].imshow(cs_map[:,:,ii])
    ax[2,ii].imshow(fwhms_map[:,:,ii])
    

cte = 1e2
eels_data = np.zeros((xsize, ysize, msh.Esize))
for index in np.ndindex(scan_size):
        islice = np.s_[index]
        mod = change_params(mod1, As_map[islice], cs_map[islice], fwhms_map[islice])
        
        mod.calculate()
        eels_data[islice] = np.random.poisson(mod.data*cte)
        
s = MultiSpectrum(msh, data=eels_data)

em.MultiSpectrumVisualizer([s.mean(1)])
        

fig, ax = plt.subplots()
ax.imshow(s.multidata.mean(1), aspect='auto')
        
        
back = em.BackgroundRemoval(s, (425,440))
rem = back.calculate_multi()

em.MultiSpectrumVisualizer([s, rem])
        
smean = rem[-1,:].mean()
specshape = s.get_spectrumshape()
E0 = settings[0]
alpha = settings[1]
beta = settings[2]    

element = 'Ti'
edge = 'L'

comp = em.ZezhongCoreLossEdgeCombined(specshape, 1, E0, alpha, beta, element, edge)
comp.parameters[0].setvalue(2)

As = [500,500,500,500]
cs = [456, 459,462,464]
fwhms = [1,1,1,1]
lorentz = []
for ii in range(len(As)):
    lorentz.append(Lorentzian(specshape, As[ii], cs[ii], fwhms[ii]))

for lor in lorentz:
    lor.parameters[0].setboundaries(100,2000)
    
    val = lor.parameters[1].getvalue()
    lor.parameters[1].setboundaries(val-1.5, val+1.5)

    lor.parameters[2].setboundaries(0.2, 3)


components = [comp] + lorentz
mod = em.Model(specshape, components)
mod.calculate()
mod.plot(spectrum=smean)

smean.set_exclude_region_energy(0, 455)
smean.show_excluded_region()

fit = em.LSQFitter(smean, mod, use_bounds=True, method='trf')
fit.perform_fit()        
fit.plot()        

fitm = em.LSQFitter(rem, mod, use_bounds=True, method='trf')
fitm.multi_fit()
        
multimodel=fitm.model_to_multispectrum()

em.MultiSpectrumVisualizer([rem, multimodel])
        
        
fig, ax = plt.subplots(1,fitm.coeff_matrix.shape[-1])
    
for ii in range(fitm.coeff_matrix.shape[-1]):
    ax[ii].imshow(fitm.coeff_matrix[:,:,ii])
    
    
    
    
    
    
    
    
    












