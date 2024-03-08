import sys
from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.spectrum import Spectrumshape
from pyEELSMODEL.core.parameter import Parameter

# note that changing the loglevel on anaconde requires at least a kernel
# restart
import logging

# logging.basicConfig(level=logging.DEBUG) #detailed debug reporting
# logging.basicConfig(level=logging.INFO) #show info on normal working of code
# only show warning that help user to avoid an issue
logging.basicConfig(level=logging.WARNING)

sys.path.append("..")  # Adds higher directory to python modules path.


def test_add_parameters():
    spec_shape = Spectrumshape(1, 300, 1024)
    comp = Component(spec_shape)

    p0 = Parameter('A', 1)
    p1 = Parameter('B', 2)

    params = [p0, p1]
    comp._setparameters(params)
    assert len(comp.parameters) == 2


def test_setcanconvolute():
    spec_shape = Spectrumshape(1, 300, 1024)
    comp = Component(spec_shape)
    assert comp.canconvolute is True
    comp._setcanconvolute(False)
    assert comp.canconvolute is False


def test_changed():
    spec_shape = Spectrumshape(1, 300, 1024)
    comp = Component(spec_shape)

    p0 = Parameter('A', 1)
    p1 = Parameter('B', 2)

    params = [p0, p1]
    comp._setparameters(params)
    assert comp.parameters[0].changed
    assert comp.parameters[1].changed

    comp.setunchanged()
    assert not comp.parameters[0].changed
    assert not comp.parameters[1].changed


def test_has_parameter():
    spec_shape = Spectrumshape(1, 300, 1024)
    comp = Component(spec_shape)

    p0 = Parameter('A', 1)
    p1 = Parameter('B', 2)
    p2 = Parameter('C', 20)

    params = [p0, p1]
    comp._setparameters(params)

    assert comp.has_parameter(p0)
    assert comp.has_parameter(p1)
    assert not comp.has_parameter(p2)


def main():
    test_add_parameters()
    test_setcanconvolute()
    test_changed()
    test_has_parameter()


if __name__ == "__main__":
    main()
