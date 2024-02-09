
import sys
sys.path.append(r'C:\Users\DJannis\PycharmProjects\pyEELSmodel')
from pyEELSMODEL.dmread.dm_ncempy import dmReader
import matplotlib.pyplot as plt


#filename = r'C:\Users\DJannis\PycharmProjects\pyEELSmodel\Examples\SrTiO3_example\hl.dm3'

dmfile = dmReader(filename)

data = dmfile['data'][0,:]
e_axis = dmfile['coords'][1]

plt.figure()
plt.plot(data)





