# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
(c) University of Antwerp 2021
"""

import sys
import pytest
sys.path.append("..") # Adds higher directory to python modules path.
from pyEELSMODEL.core.parameter import Parameter

#note that changing the loglevel on anaconde requires at least a kernel restart
import logging
#logging.basicConfig(level=logging.DEBUG) #detailed debug reporting
#logging.basicConfig(level=logging.INFO) #show info on normal working of code
logging.basicConfig(level=logging.WARNING) #only show warning that help user to avoid an issue


def test_creation():
    p1=Parameter('p1',1.0) 
    assert p1.getvalue()== 1
    
def test_boundaries():
    p1=Parameter('p1',1.0) 
    p1.setboundaries(0,5)
    assert p1.getvalue()== 1
    p1.setvalue(10.0)
    assert p1.getvalue()== 1
    p1.setvalue(2.0)
    assert p1.getvalue()== 2
    p1.setboundaries(0, 10)
    p1.setvalue(10.0)
    assert p1.getvalue()== 10

def test_coupling():
    p1=Parameter('p1',1.0) 
    p2=Parameter('p2',2.0) 
    assert p1.getvalue()== 1
    assert p2.getvalue()== 2
    p2.couple(p1,3)
    assert p2.getvalue()== 3
    # do some erasing of coupled ones
    p1.release() #always call release if a parameter might dissapear
    del p1
    assert p2.getvalue()== 2 #back to its own uncoupled value

def test_operators():
    p1=Parameter('p1',1.0)
    p2=Parameter('p2',2.0)
    #operators between 2 parameters
    psum=p1+p2
    psub=p1-p2
    pmul=p1*p2
    pdiv=p1/p2
    assert psum.getvalue()==3
    assert psub.getvalue()==-1
    assert pmul.getvalue()==2
    assert pdiv.getvalue()==0.5
    
    #operator between parameter and float
    pfsum=p1+3.5
    pfsub=p1-3.5
    pfmul=p1*3.5
    pfdiv=p1/0.5
    assert pfsum.getvalue()==4.5
    assert pfsub.getvalue()==-2.5
    assert pfmul.getvalue()==3.5
    assert pfdiv.getvalue()==2.0
    
    #operator between parameter and float
    pisum=p1+1024
    pisub=p1-1024
    pimul=p1*1024
    pidiv=p1/10
    assert pisum.getvalue()==1025
    assert pisub.getvalue()==-1023
    assert pimul.getvalue()==1024
    assert pidiv.getvalue()==0.1
    
    #and try with a forbidden type
    with pytest.raises(TypeError):
        pssum=p1+'not_a_valid_type'
    with pytest.raises(TypeError):
        pssub=p1-'not_a_valid_type'
    with pytest.raises(TypeError):
        psmul=p1*'not_a_valid_type'
    with pytest.raises(TypeError):
        psdiv=p1/'not_a_valid_type'
        

def main():
    test_creation()
    test_boundaries()
    test_coupling()
    test_operators()

if __name__ == "__main__":
    main()
    
