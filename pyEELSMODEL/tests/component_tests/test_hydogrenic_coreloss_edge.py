# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:25:55 2021

@author: joverbee
"""

import sys
from pyEELSMODEL.components.CLedge.hydrogen_coreloss_edge import \
    HydrogenicCoreLossEdge
from pyEELSMODEL.core.spectrum import Spectrumshape
import pytest
import numpy as np
sys.path.append("..")  # Adds higher directory to python modules path.


def test_set_edge():
    specshape = Spectrumshape(1, 100, 1024)
    OK = HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                                beta=20e-3, element='O', edge='K')
    assert OK.edge == 'K'

    TiL = HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                                 beta=20e-3, element='Ti', edge='L')
    assert TiL.edge == 'L'

    with pytest.raises(ValueError):
        HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                               beta=20e-3, element='Ti', edge='M')
    with pytest.raises(ValueError):
        HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                               beta=20e-3, element='C', edge='L')
    with pytest.raises(ValueError):
        HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                               beta=20e-3, element='Ag', edge='L')


def test_set_onset_energy():
    specshape = Spectrumshape(1, 100, 1024)
    OK = HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                                beta=20e-3, element='O', edge='K')
    assert OK.onset_energy == 532.
    CaL = HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                                 beta=20e-3, element='Ca', edge='L')
    assert CaL.onset_energy == 347.


def test_calculate():
    specshape = Spectrumshape(1, 100, 1024)
    OK = HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                                beta=20e-3, element='O', edge='K')
    OK.calculate()
    # give some interval to have a least the right order of magnitude right
    # assert OK.data.sum() >= 1e-22
    # assert OK.data.sum() <= 1e-21

    # check if by changing parameter A new data is calculated
    prev_data = np.copy(OK.data)
    A = OK.getparameter(0)
    A.setvalue(1e5)
    OK.calculate()
    np.testing.assert_allclose(OK.data*1e-2, prev_data)


def test_gradient():
    specshape = Spectrumshape(1, 100, 1024)
    OK = HydrogenicCoreLossEdge(specshape, A=1e3, E0=300e3, alpha=5e-3,
                                beta=20e-3, element='O', edge='K')
    OK.calculate()
    OK.getgradient(OK.getparameter(0))
    assert (OK.gradient[0]*1e3 == OK.data).all()


def main():
    test_set_edge()
    test_set_onset_energy()
    test_calculate()
    test_gradient()


if __name__ == "__main__":
    main()
