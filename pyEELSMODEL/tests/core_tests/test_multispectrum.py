'''
copyright University of Antwerp 2021
author: Jo Verbeeck and Daen Jannis
'''
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
import matplotlib.pyplot as plt
import pytest
import numpy as np

#note that changing the loglevel on anaconde requires at least a kernel restart
import logging
#logging.basicConfig(level=logging.DEBUG) #detailed debug reporting
#logging.basicConfig(level=logging.INFO) #show info on normal working of code
logging.basicConfig(level=logging.WARNING) #only show warning that help user to avoid an issue



def test_init_multispectrum():
    mshape=MultiSpectrumshape(1,100,1024,4,4)
    mspec=MultiSpectrum(mshape)   
    #mspec.plot()
    assert mspec.size==1024
    assert mspec.xsize==4
    assert mspec.ysize==4   
    
def test_indexing():
    mshape=MultiSpectrumshape(1,100,1024,4,4)
    mspec=MultiSpectrum(mshape)   
    
    assert mspec[0,:,:].size==1024

    with pytest.raises(IndexError):
        assert mspec[16,:,:].size==1024
        
def test_init_from_data():
    mshape=MultiSpectrumshape(1,100,1024,4,4)
    mdata=np.zeros((4,4,1024))
    for i in range(4):
        mdata[i,0,]=i
    mspec=MultiSpectrum(mshape,mdata)
   
    assert mspec[0,0].data[0]==0
    assert mspec[1,0].data[0]==1
    assert mspec[2,0].data[0]==2
    assert mspec[3,0].data[0]==3 

# def test_iterator():
#     mshape=MultiSpectrumshape(1,100,1024,4,4)
#     mspec=MultiSpectrum(mshape) 
    
#     i=0
#     for s in mspec:
#        i+=1
#        print(s.get_max())
#     assert i==16
    
# def test_set_pppc():
#     specshape = MultiSpectrumshape(1, 200, 1024,4,4)
#     data = np.ones((4,4,1024))
#     s0 =MultiSpectrum(specshape, data=data)
#     s0.pppc = 4
    
#     assert np.all(s0.multidata == 4*data)
#     assert s0.error.sum() == np.sqrt(4*data[0,0]).sum()



def main():
    test_init_multispectrum()
    test_indexing()
    test_init_from_data()
    # test_set_pppc()

if __name__ == "__main__":
    main()