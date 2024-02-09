"""
Makes a .hdf5 file containing the relevant information obtained from the
database of Zezhong Zhang.
"""


import h5py
import os
from glob import glob
from mendeleev import element


database_path = r'C:\Users\DJannis\PycharmProjects\pyEELSmodel\database\Dirac'

os.chdir(database_path)
files = glob('*.hdf5')


file_path = r'C:\Users\DJannis\PycharmProjects\pyEELSmodel\pyEELSMODEL'

newfile='element_info.hdf5'
new = h5py.File(os.path.join(file_path, newfile), 'w')


for file in files:
    grp = new.create_group(file[:-5])
    
    org = h5py.File(file, 'r')

    edges = list(org['metadata']['edges_info'].keys())

    el = element(file[:-5])
    at_n = el.atomic_number

    for edge in edges:
        atrs_l = list(org['metadata']['edges_info'][edge].attrs.keys())
        
        if 'onset_energy' in atrs_l:
            onset_energy = org['metadata']['edges_info'][edge].attrs['onset_energy']
        else:
            onset_energy = org['metadata']['edges_info'][edge].attrs['onset_energy_guess']

        
        occupancy = org['metadata']['edges_info'][edge].attrs['occupancy_ratio']

        grp_e = grp.create_group(edge)
        grp_e.attrs['onset_energy'] = onset_energy
        grp_e.attrs['occupancy_ratio'] = occupancy
        grp.attrs['Z'] = at_n



    org.close()
        
new.close()
        

        
        
        
        
        
        
        
        
        