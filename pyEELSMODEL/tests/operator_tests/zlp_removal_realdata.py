from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.operators.zlpremoval import ZLPRemoval
from pyEELSMODEL.operators.multispectrumvisualizer import MultiSpectrumVisualizer
from pyEELSMODEL.operators.fastalignzeroloss import FastAlignZeroLoss
from pyEELSMODEL.operators.alignzeroloss import AlignZeroLoss
import os
import matplotlib.pyplot as plt

dir_ = r'C:\Users\daen_\OneDrive - Universiteit Antwerpen\TFS\paper_background'
name = 'lowloss.hdf5'

s = MultiSpectrum.load(os.path.join(dir_, name))



align = FastAlignZeroLoss(s)
align.perform_alignment()
align.show_shift()

s_al = align.aligned[:10,:10]


rem = ZLPRemoval(s_al, model_type='Lorentzian', signal_range=(-20,6))
rem.fit_zlp()
rem.mirrored_zlp()

rem.spectrum.show_excluded_region()
MultiSpectrumVisualizer([s_al, rem.zlp])

MultiSpectrumVisualizer([s_al, rem.mirror_zlp])





