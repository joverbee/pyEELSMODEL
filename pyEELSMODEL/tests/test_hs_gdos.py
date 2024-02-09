import numpy as np
import matplotlib.pyplot as plt
import pyEELSMODEL.misc.hs_gdos as hsdos



#Calcium edge
Z = 20
filename_L23 =  r'.\H-S GOS Tables\Ca.L3'
filename_L1 =  r'.\H-S GOS Tables\Ca.L1'

e_L3 = 346
e_L2 = 350
e_L1 = 438


energy_axis = np.arange(300, 1200, 1)
E0 = 200e3
beta = 0.005
dsigma_dE_L3 = 1*hsdos.dsigma_dE_HS(energy_axis, Z, e_L3, E0, beta, 0, filename_L23, q_steps=100)
dsigma_dE_L2 = 0.5*hsdos.dsigma_dE_HS(energy_axis, Z, e_L2, E0, beta, 0, filename_L23, q_steps=100)
dsigma_dE_L1 = 1*hsdos.dsigma_dE_HS(energy_axis, Z, e_L1, E0, beta, 0, filename_L1, q_steps=100)



plt.figure()
plt.plot(energy_axis, (dsigma_dE_L3+dsigma_dE_L2+dsigma_dE_L1)*1e28)
plt.plot(energy_axis, (dsigma_dE_L3)*1e28)
plt.plot(energy_axis, (dsigma_dE_L2)*1e28)
plt.plot(energy_axis, (dsigma_dE_L1)*1e28)

#Cerium edge
Z = 58
filename_M45 =  r'.\H-S GOS Tables\Ce.M5'
filename_M23 =  r'.\H-S GOS Tables\Ce.M3'

e_M5 = 883.0
e_M4 = 901.0
e_M3 = 1185.0
e_M2 = 1273.0

energy_axis = np.arange(700, 2000, 1)
E0 = 200e3
beta = 0.005
dsigma_dE_M5 = 1*hsdos.dsigma_dE_HS(energy_axis, Z, e_M5, E0, beta, 0, filename_M45, q_steps=100)
dsigma_dE_M4 = (2/3)*hsdos.dsigma_dE_HS(energy_axis, Z, e_M4, E0, beta, 0, filename_M45, q_steps=100)
dsigma_dE_M3 = 1*hsdos.dsigma_dE_HS(energy_axis, Z, e_M3, E0, beta, 0, filename_M23, q_steps=100)
dsigma_dE_M2 = 0.5*hsdos.dsigma_dE_HS(energy_axis, Z, e_M2, E0, beta, 0, filename_M23, q_steps=100)



plt.figure()
plt.plot(energy_axis, (dsigma_dE_M5+dsigma_dE_M4+dsigma_dE_M3+dsigma_dE_M2)*1e28)
# plt.plot(energy_axis, (dsigma_dE_L3)*1e28)
# plt.plot(energy_axis, (dsigma_dE_L2)*1e28)
# plt.plot(energy_axis, (dsigma_dE_L1)*1e28)

