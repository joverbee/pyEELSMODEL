# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("../..") # Adds higher directory to python modules path.
from pyEELSMODEL.components import HSCoreLossEdge
from core.spectrum import Spectrumshape
import pytest


def test_set_edge():
    specshape=Spectrumshape(1,100,1024)
    OK= HSCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3, element= 'O', edge='K')
    assert OK.edge == 'K'

    with pytest.raises(TypeError):
        HSCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3, element= 'O', edge=5)

    with pytest.raises(ValueError):
        HSCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3, element= 'O', edge='J')

    with pytest.raises(ValueError):
        HSCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3, element= 'O', edge='M5')


def test_set_onset_energy():
    specshape=Spectrumshape(1,100,1024)
    OK= HSCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3, element= 'O', edge='K')
    assert OK.onset_energy == 532.
    CaL= HSCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3, element= 'Ca', edge='L3')
    assert CaL.onset_energy == 346.


def main():
    test_set_edge()
    test_set_onset_energy()


if __name__ == "__main__":
    main()