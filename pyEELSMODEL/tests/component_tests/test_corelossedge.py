# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.core.spectrum import Spectrumshape
import pytest
sys.path.append("..")  # Adds higher directory to python modules path.


def test_set_element():
    specshape = Spectrumshape(1, 100, 1024)
    OK = CoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3,
                      element='O', edge='K1')
    assert OK.element == 'O'

    with pytest.raises(ValueError):
        CoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3,
                     element='OK', edge='K1')


def test_set_Z():
    specshape = Spectrumshape(1, 100, 1024)
    OK = CoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3,
                      element='O', edge='K1')
    CK = CoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3,
                      element='C', edge='K1')
    TiL = CoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3, beta=20e-3,
                       element='Ti', edge='L3')

    assert OK.Z == 8
    assert CK.Z == 6
    assert TiL.Z == 22


def main():
    test_set_element()
    test_set_Z()


if __name__ == "__main__":
    main()
