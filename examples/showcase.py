from pyEELSMODEL.operators.simulator.coreloss_simulator import CoreLossSimulator
from pyEELSMODEL.operators.quantification.elemental_quantification import ElementalQuantification
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval

from pyEELSMODEL.core.multispectrum import MultiSpectrumshape, MultiSpectrum
import pyEELSMODEL.api as em

import numpy as np
import matplotlib.pyplot as plt


def make_circular_mask(xco, yco, Rin, Rout, shape):
    XX, YY = np.meshgrid(np.arange(shape[1])-yco, np.arange(shape[0])-xco)
    
    R = np.sqrt(XX**2+YY**2)
    
    mask = np.zeros(shape)
    boolean = (R>=Rin) & (R<Rout)
    mask[boolean] = 1
    return mask

def make_rectangular_mask(xco, yco, width,height, shape):
    mask = np.zeros(shape)
    mask[xco:xco+width, yco:yco+height] = 1
    return mask
    


elements = ['C', 'N', 'O', 'Fe']
edges = ['K', 'K', 'K', 'L']
Zs = [6, 7, 8, 26]

xsize = 128
ysize = 128
maps = np.zeros((len(elements),xsize,ysize))


mask0 =make_rectangular_mask(5, 5, 20, 20, (xsize,ysize))
mask1 =  make_rectangular_mask(90, 90, 20, 30, (xsize,ysize))
mask2 = make_circular_mask(xsize//2, ysize//2, 20, 30, (xsize,ysize))
mask3 = make_circular_mask(xsize//2, ysize//2, 0, 20, (xsize,ysize))

maps[0] = 1
maps[1] = 2*mask0 + mask1
maps[2] = mask2
maps[3] = mask3+0.5*mask2

adf = np.zeros((xsize, ysize))
tlambda_map = np.zeros_like(adf)
for ii in range(maps.shape[0]):
    adf += (Zs[ii]*maps[ii])**2
    tlambda_map += Zs[ii]*maps[ii]

tlambda_map = tlambda_map/tlambda_map.max()

fig, ax = plt.subplots(1,2)
ax[0].imshow(adf, cmap='gray')
ax[1].imshow(tlambda_map, cmap='gray')


settings = (300e3, 1e-9, 20e-3)

msh = MultiSpectrumshape(0.5, 200, 2048, xsize, ysize)


sim = CoreLossSimulator(msh, elements, edges, maps, tlambda_map, settings)
sim.bg_method='convolved'
sim.simulate_multispectrum()


hl = sim.multispectrum
ll = sim.ll

quant = ElementalQuantification(hl, elements, edges, settings, ll=ll)
quant.n_bgterms = 4
quant.linear_fitter_method = 'ols'
quant.do_procedure()

multimodels = quant.get_multimodels()

em.MultiSpectrumVisualizer([ll.get_interval((-10,50))])

quantpw = ElementalQuantification(hl, elements, edges, settings, ll=ll)
quantpw.background_model = 'powerlaw'
quantpw.do_procedure()

multimodelspw = quantpw.get_multimodels()

#%% Ordinary backgroundremoval 
signal_ranges = [[220,280],[350,395],[440,520],[600,700]]
int_wins= [[285,350],[400,475],[525,600], [700,800]]
E0=settings[0]
alpha=settings[1]
beta=settings[2]
int_maps = np.zeros_like(maps)
for ii in range(len(signal_ranges)):
    back = BackgroundRemoval(hl, signal_ranges[ii])
    rem = back.calculate_multi()
    int_maps[ii] = back.quantify_from_edge(int_wins[ii], elements[ii], edges[ii],
                                           E0, alpha, beta, ll=ll)

fig, ax = plt.subplots(1, int_maps.shape[0])
for ii in range(int_maps.shape[0]):
    ax[ii].imshow(int_maps[ii])


#%% 
masks = [mask0, mask1, mask2, mask3]
mps = [quant.elemental_maps, quantpw.elemental_maps, int_maps]
lbs = ['Linear', 'Powerlaw', 'Conventional']
# fig, ax = plt.subplots(len(masks), len(elements))
# bins = np.linspace(-0.5,3,100)
# xx = 0.5*(bins[:-1]+bins[1:])
rn=2
for jj, mask in enumerate(masks):
    
# mask = masks[0]
    
    for ii in range(mps[0].shape[0]):
        print('Mask: '+str(jj)+', element: '+str(elements[ii]))
        theo = maps[ii][mask.astype('bool')].flatten().mean()
        print('Theoretical value: ' + str(theo))
        for kk in range(len(mps)):
            ndata = mps[kk][ii][mask.astype('bool')].flatten()
            avg = ndata.mean()
            std = ndata.std()
            print(lbs[kk] + ' method: '+str(np.round(avg,rn))+' +- ' + str(np.round(std, rn)))
            
        print(' ')


mths = [maps, int_maps, quantpw.elemental_maps, quant.elemental_maps]
mths_nms = ['Ground truth', 'Conventional', 'Power-law Method', 'Linear Method']

fig, ax = plt.subplots(4, len(elements))
for jj in range(len(mths)):
    for ii in range(len(elements)):
        if ii == 0:
            vmin = -0.1
            vmax = 1.5*mths[jj][ii].max()
        
        ax[jj,ii].imshow(mths[jj][ii], vmin=vmin, vmax=vmax)

        if ii == 0:
            ax[jj,ii].set_ylabel(mths_nms[jj])
        if jj == 0:
            ax[jj,ii].set_title(elements[ii])























