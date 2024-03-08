import sys
import os
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.components.gaussian import Gaussian
import numpy as np
import pytest
# note that changing the loglevel on anaconde requires at least a
# kernel restart
import logging

# logging.basicConfig(level=logging.DEBUG) #detailed debug reporting
# logging.basicConfig(level=logging.INFO) #show info on normal working of code
logging.basicConfig(level=logging.WARNING)
sys.path.append("..")  # Adds higher directory to python modules path.


def test_copy_spectrum():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    s1 = s0.copy()
    assert s0 != s1


def test_check_same_settings():
    specshape0 = Spectrumshape(1, 200, 1024)
    specshape1 = Spectrumshape(1, 200, 1024)
    specshape2 = Spectrumshape(2, 200, 1024)
    specshape3 = Spectrumshape(1, 400, 1024)
    specshape4 = Spectrumshape(1, 200, 512)

    s0 = Spectrum(specshape0)
    s1 = Spectrum(specshape1)
    s2 = Spectrum(specshape2)
    s3 = Spectrum(specshape3)
    s4 = Spectrum(specshape4)

    assert s0._check_not_same_settings(s1) is False
    assert s0._check_not_same_settings(s2) is True
    assert s0._check_not_same_settings(s3) is True
    assert s0._check_not_same_settings(s4) is True


def test_multiply_spectrum():
    # Still check when ValueError is thrown

    specshape = Spectrumshape(1, 200, 1024)
    differentshape = Spectrumshape(1, 201, 1025)
    s0 = Spectrum(specshape, data=np.ones(specshape.size) * 2)
    s1 = Spectrum(specshape, data=np.ones(specshape.size) * 4)
    s2 = Spectrum(differentshape, data=np.ones(differentshape.size) * 4)
    b = 5
    c = 10.

    smult = s0 * s1
    smult1 = s0 * b
    smult2 = s0 * c

    assert smult.data.sum() == 8 * 1024
    assert smult1.data.sum() == (2 * 5) * 1024
    assert smult2.data.sum() == (2 * 10) * 1024

    with pytest.raises(ValueError):
        s0 * s2
    with pytest.raises(TypeError):
        s0 * 'not_valid'


def test_add_spectrum():
    specshape = Spectrumshape(1, 200, 1024)
    differentshape = Spectrumshape(1, 201, 1025)
    s0 = Spectrum(specshape, data=np.ones(specshape.size) * 2)
    s1 = Spectrum(specshape, data=np.ones(specshape.size) * 4)
    s2 = Spectrum(differentshape, data=np.ones(differentshape.size) * 4)
    # b = 5
    # c = 10.

    sadd = s0 + s1
    # sadd1 = s0 + b
    # sadd2 = s0 + c

    assert sadd.data.sum() == 6 * 1024
    assert sadd.dispersion == s0.dispersion
    assert sadd.offset == s0.offset
    assert sadd.size == s0.size

    with pytest.raises(ValueError):
        s0 + s2
    with pytest.raises(TypeError):
        s0 + 'not_valid'


def test_subtract_spectrum():
    specshape = Spectrumshape(1, 200, 1024)
    differentshape = Spectrumshape(1, 201, 1025)
    s0 = Spectrum(specshape, data=np.ones(specshape.size) * 2)
    s1 = Spectrum(specshape, data=np.ones(specshape.size) * 4)
    s2 = Spectrum(differentshape, data=np.ones(differentshape.size) * 4)

    ssub = s1 - s0
    assert ssub.data.sum() == 1024 * (4 - 2)
    assert ssub.dispersion == s0.dispersion
    assert ssub.offset == s0.offset
    assert ssub.size == s0.size

    with pytest.raises(ValueError):
        s0 - s2
    with pytest.raises(TypeError):
        s0 - 'not_valid'


def test_divide_spectrum():
    specshape = Spectrumshape(1, 200, 1024)
    differentshape = Spectrumshape(1, 201, 1025)
    s0 = Spectrum(specshape, data=np.ones(specshape.size) * 2)
    s1 = Spectrum(specshape, data=np.ones(specshape.size) * 4)
    s2 = Spectrum(differentshape, data=np.ones(differentshape.size) * 4)

    ssub = s1 / s0
    assert ssub.data.sum() == 1024 * (4 / 2)
    assert ssub.dispersion == s0.dispersion
    assert ssub.offset == s0.offset
    assert ssub.size == s0.size

    with pytest.raises(ValueError):
        s0 / s2
    with pytest.raises(TypeError):
        s0 / 'not_valid'


def test_bad_index():
    # Still add when ValueError is thrown
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    s0.bad_index(1)


def test_set_exclude_region():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    s0.set_exclude_region(10, 110)
    assert s0.exclude[10:110].all()
    assert not s0.exclude[:10].all()
    assert not s0.exclude[110:].all()


def test_reset_exclude_region():
    specshape = Spectrumshape(1, 200, 1024)
    s0 = Spectrum(specshape)
    s0.set_exclude_region(10, 110)
    s0.reset_exclude_region(10, 90)
    assert not s0.exclude[10:90].all()
    assert s0.exclude[90:110].all()


def test_get_max_index():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.zeros(specshape.size)
    data[10] = 1
    s0 = Spectrum(specshape, data=data)
    max_index = s0.getmaxindex()
    assert max_index == 10


def test_get_first_higher_then():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.zeros(specshape.size)
    data[10] = 2
    data[100] = 2

    s0 = Spectrum(specshape, data=data)
    index = s0.get_first_higher_then(10)
    index1 = s0.get_first_higher_then(1)

    assert index is None
    assert index1 == 10


def test_get_max():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.zeros(specshape.size)
    data[10] = 2
    data[100] = 20
    s0 = Spectrum(specshape, data=data)
    assert s0.get_max() == 20


def test_get_min():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.ones(specshape.size)
    data[10] = 2
    data[100] = 20
    data[5] = 0.5
    s0 = Spectrum(specshape, data=data)
    assert s0.get_min() == 0.5


def test_set_pppc():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.arange(specshape.size)
    s0 = Spectrum(specshape, data=data)
    s0.pppc = 4

    # assert np.all(s0.data == 4*data)
    assert s0.error.sum() == np.sqrt(4 * data).sum()


def test_get_energy_index():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.arange(specshape.size)
    s0 = Spectrum(specshape, data=data)
    assert s0.get_energy_index(199) == 0
    assert s0.get_energy_index(1224) == 1024
    assert s0.get_energy_index(300) == 100


def test_save_and_load_hdf5():
    specshape = Spectrumshape(1, 200, 1024)
    data = np.arange(specshape.size)
    s0 = Spectrum(specshape, data=data)
    filename = 'test.hdf5'
    s0.save_hdf5(filename)
    s1 = Spectrum.load(filename)
    os.remove(filename)

    a = s0.data - s1.data
    assert np.all(a == 0)
    assert s0.offset == s1.offset
    assert s0.dispersion == s1.dispersion


def test_inter_to_other_energy_axis():
    shape = Spectrumshape(1, 100, 512)
    ga1 = Gaussian(shape, 100, 300, 100)
    ga1.calculate()
    ndata = ga1.data
    s = Spectrum(shape, data=ndata)

    shape0 = Spectrumshape(0.33, 200, 1024)
    s0 = Spectrum(shape0)

    sint = s.interp_to_other_energy_axis(s0)

    # plt.figure()
    # plt.plot(s.energy_axis, s.data)
    # plt.plot(sint.energy_axis, sint.data)
    #
    # print(s.energy_axis[np.argmax(s.data)])
    # print(sint.energy_axis[np.argmax(sint.data)])

    # the energy axis of the interpolated data is the same as to which we map
    assert (s0.energy_axis == sint.energy_axis).all()
    assert np.argmax(s.data) - np.argmax(sint.data) == -103

    dif = s.energy_axis[np.argmax(s.data)] - sint.energy_axis[
        np.argmax(sint.data)]
    assert np.abs(dif) < 0.1


def test_rescale_spectrum():
    shape = Spectrumshape(1, 100, 512)
    ga1 = Gaussian(shape, 100, 300, 100)
    ga1.calculate()
    ndata = ga1.data
    s = Spectrum(shape, data=ndata)
    s0 = s.rescale_spectrum(1, -50)

    # plt.figure()
    # plt.plot(s.energy_axis, s.data)
    # plt.plot(s0.energy_axis, s0.data)

    dif = s.energy_axis[np.argmax(s.data)] - s0.energy_axis[
        np.argmax(s0.data)] - 50
    assert np.abs(dif) < 0.1

    s1 = s.rescale_spectrum(0.5, 0)
    dif = s.energy_axis[np.argmax(s.data)] - s1.energy_axis[
        np.argmax(s1.data)] - 100
    assert np.abs(dif) < 0.1


def main():
    test_copy_spectrum()
    test_check_same_settings()
    test_multiply_spectrum()
    test_add_spectrum()
    test_subtract_spectrum()
    test_divide_spectrum()
    test_bad_index()
    test_set_exclude_region()
    test_reset_exclude_region()
    test_get_max_index()
    test_get_first_higher_then()
    test_get_max()
    test_get_min()
    test_set_pppc()
    test_get_energy_index()
    test_save_and_load_hdf5()
    test_inter_to_other_energy_axis()
    test_rescale_spectrum()


if __name__ == "__main__":
    main()
