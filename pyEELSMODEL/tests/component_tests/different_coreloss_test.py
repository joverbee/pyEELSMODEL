#test for coreloss 
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape

from pyEELSMODEL.components.CLedge.hs_coreloss_edgecombined import HSCoreLossEdgeCombined
from pyEELSMODEL.components.CLedge.hs_coreloss_edge import HSCoreLossEdge
from pyEELSMODEL.components.CLedge.hydrogen_coreloss_edge import HydrogenicCoreLossEdge
from pyEELSMODEL.components.CLedge.kohl_coreloss_edge import KohlLossEdge
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edge import ZezhongCoreLossEdge


alpha = 20e-3 #convergence angle [rad]
beta = 40e-3 #collection angle [rad]
E0 = 300e3 #acceleration voltage [V]
elements = ['Ti','O','Mn','Ba', 'La']
edges = ['L','K','L','M','M']


specshape = Spectrumshape(1, 50, 3000)

edge = 'K1'
element = 'O'


comp = HSCoreLossEdge(specshape, 1, E0, alpha, beta, element, edge)
comp = HSCoreLossEdgeCombined(specshape, 1, E0, alpha, beta, element, 'K')
comp = HydrogenicCoreLossEdge(specshape, 1, E0, alpha, beta, element, 'K')
comp = KohlLossEdge(specshape, 1, E0, alpha, beta, element, edge)
comp = ZezhongCoreLossEdge(specshape, 1, E0, alpha, beta, element, edge)




comp.calculate()










