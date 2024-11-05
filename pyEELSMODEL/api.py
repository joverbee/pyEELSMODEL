# the classes in the cores
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.model import Model

#the classes from the components
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined \
    import ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.CLedge.kohl_coreloss_edgecombined \
    import KohlLossEdgeCombined
from pyEELSMODEL.components.CLedge.hydrogen_coreloss_edge \
    import HydrogenicCoreLossEdge
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT

#the fitters, mainly linear and least squares optimization
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from pyEELSMODEL.fitters.lsqfitter import LSQFitter
from pyEELSMODEL.fitters.quadratic_fitter import QuadraticFitter

#operators which are useful
from pyEELSMODEL.operators.backgroundremoval import BackgroundRemoval
from pyEELSMODEL.operators.aligns.fastalignzeroloss import FastAlignZeroLoss
from pyEELSMODEL.operators.aligns.alignzeroloss import AlignZeroLoss
from pyEELSMODEL.operators.aligns.aligncrosscorrelation import\
    AlignCrossCorrelation
from pyEELSMODEL.operators.multispectrumvisualizer import \
    MultiSpectrumVisualizer
from pyEELSMODEL.operators.areaselection import \
    AreaSelection
from pyEELSMODEL.operators.calibratespectrum import \
    CalibrateSpectrum



from pyEELSMODEL.operators.simulator.coreloss_simulator import\
    CoreLossSimulator
from pyEELSMODEL.operators.quantification.elemental_quantification import\
    ElementalQuantification

