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

maps[0] = 1*mask0  + 0.5*mask2 + 0.7*mask3#ce 3+
maps[1] = 1*mask1 + 0.5*mask2 + 0.3*mask3#ce 4+

    
fig, ax = plt.subplots()
ax.imshow(maps[0])

file_ce3 = os.path.join(cdir, r'data\ce3_edge.hdf5')
ce3 = em.Spectrum.load(file_ce3)


file_ce4 = os.path.join(cdir, r'data\ce4_edge.hdf5')
ce4 = em.Spectrum.load(file_ce4)
ce4.data = ce4.data*ce3.data.sum()/ce4.data.sum()

plt.figure()
plt.plot(ce3.data)
plt.plot(ce4.data)

cte=1
# tlambda_map = np.ones_like(mask0)*0.3*cte
tlambda_map = np.ones_like(mask0)*0.0003*cte

# tlambda_map[mask0==1] = 0.2
# tlambda_map[mask1==1] = 0.3
# tlambda_map[mask2==1] = 0.5
# tlambda_map[mask3==1] = 0.4


plt.figure()
plt.imshow(tlambda_map)



settings = (300e3, 1e-9, 20e-3)
msh = em.MultiSpectrumshape(0.05, 840, 4096, xsize, ysize)
sh = msh.getspectrumshape()

sim = em.CoreLossSimulator(msh, [], [], maps, tlambda_map, settings)
sim.fwhm=0.3
sim.n_plasmon = 3
sim.make_lowloss()

sim.element_components = []
sim.element_components.append(FixedPattern(sh, ce3))
sim.element_components.append(FixedPattern(sh, ce4))

sim.make_coreloss()

em.MultiSpectrumVisualizer([sim.ll])
em.MultiSpectrumVisualizer([sim.multispectrum])


hl = sim.multispectrum
ll = sim.ll


from pyEELSMODEL.operators.quantification.extract_experimental_edge import ExperimentalEdgeExtractor


exp = ExperimentalEdgeExtractor(hl, settings, ll=ll)

# exp.define_new_region()
exp.define_new_region(max_points=4, coords = [[5,5,25,25],[5,25,25,5]])
exp.define_new_region(max_points=4, coords = [[90,90,120,120],[90,110,110,90]])


exp.show_regions()


exp.calculate_spectra()
exp.show_average_spectra()


fixs0 = exp.extract_edge(0, ['Ce'], ['M'], [35], [5])
fixs0[0].setname('Ce 3+')
fixs1 = exp.extract_edge(1, ['Ce'], ['M'], [35], [5])
fixs1[0].setname('Ce 4+')

fig, ax = plt.subplots()
ax.plot(fixs0[0].energy_axis, fixs0[0].data)
ax.plot(fixs1[0].energy_axis, fixs1[0].data)

fig, ax = plt.subplots()
ax.plot(fixs0[0].energy_axis, fixs0[0].data/fixs0[0].integrate((950,1000)), label='Ce3+', color='blue')
ax.plot(fixs1[0].energy_axis, fixs1[0].data/fixs1[0].integrate((950,1000)), label='Ce4+', color='red')

ax.plot(ce3.energy_axis, ce3.data/ce3.integrate((950,1000)), color='blue', linestyle='dotted', label='Simulted Ce3+')
ax.plot(ce4.energy_axis, ce4.data/ce4.integrate((950,1000)), color='red', linestyle='dotted', label='Simulted Ce4+')

n = 4
bg = LinearBG(specshape=hl.get_spectrumshape(), rlist=np.linspace(1,5,n))

comps = [fixs0[0], fixs1[0]]

llcomp  = MscatterFFT(hl.get_spectrumshape(), ll)

components = [bg]+comps+[llcomp]
mod = em.Model(hl.get_spectrumshape(), components)
fit = em.LinearFitter(hl, mod, use_weights=True)

fit.multi_fit()

fig, maps, names = fit.show_map_result(comps)


multimodel = fit.model_to_multispectrum()

em.MultiSpectrumVisualizer([hl, multimodel])












