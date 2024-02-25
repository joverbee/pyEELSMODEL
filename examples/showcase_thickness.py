import pyEELSMODEL.api as em
import os 
import numpy as np
import matplotlib.pyplot as plt
from pyEELSMODEL.components.fixedpattern import FixedPattern
from pyEELSMODEL.components.linear_background import LinearBG
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT

os.chdir(r'C:\Users\DJannis\PycharmProjects\project\pyEELSMODEL\examples')
cdir = os.getcwd()


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
    


xsize = 128
ysize = 128
maps = np.zeros((2,xsize,ysize))


mask0 =make_rectangular_mask(5, 5, 20, 20, (xsize,ysize))
mask1 =  make_rectangular_mask(90, 90, 20, 30, (xsize,ysize))
mask2 = make_circular_mask(xsize//2, ysize//2, 20, 30, (xsize,ysize))
mask3 = make_circular_mask(xsize//2, ysize//2, 0, 20, (xsize,ysize))


cte=1
tlambda_map = np.ones_like(mask0)*0.1
tlambda_map[mask0==1] = cte*0.5
tlambda_map[mask1==1] = cte*0.7
tlambda_map[mask2==1] = cte*1
tlambda_map[mask3==1] = cte*1.5

plt.figure()
plt.imshow(tlambda_map)



settings = (300e3, 1e-9, 20e-3)
msh = em.MultiSpectrumshape(0.05, 840, 2048, xsize, ysize)
sh = msh.getspectrumshape()

sim = em.CoreLossSimulator(msh, [], [], maps, tlambda_map, settings)
sim.fwhm=0.3
sim.n_plasmon = 3
sim.make_lowloss()

em.MultiSpectrumVisualizer([sim.multispectrum])



from pyEELSMODEL.operators.estimate_thickness import ThicknessEstimator

thick = ThicknessEstimator(sim.ll, model_type='Mirrored')

thick.log_ratio_method()

em.MultiSpectrumVisualizer([thick.zlpremoval.zlp, sim.ll])



fig, ax = plt.subplots()
ax.imshow(thick.tlambda)











