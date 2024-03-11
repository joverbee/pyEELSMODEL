"""
Testing the functionality of the multiple zero loss alignements.
"""

import numpy as np
import pytest

from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrumshape
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.lorentzian import Lorentzian
from pyEELSMODEL.operators.aligns.alignzeroloss import AlignZeroLoss
from pyEELSMODEL.operators.aligns.fastalignzeroloss import FastAlignZeroLoss
from pyEELSMODEL.operators.aligns.align import Align
from pyEELSMODEL.operators.aligns.aligncrosscorrelation import \
    AlignCrossCorrelation


# Make a dataset where the centre changes.
# Every column has a incremental increase and between every row there is
# a random offset

def make_artificial_dataset(start, end, model_type="Gaussian",
                            return_shift=False):
    """
    Make artificial datasets used to test the performance of the zero loss
    alignment
    """

    ncol = 100  # number of columns
    nrow = 12  # number of rows
    width = 10  # widht of the centre misalignement
    fwhm = 3  # fwhm of gaussian or lorentzian
    amp = 100  # amplitude
    specshape = Spectrumshape(0.5, -200, 1024)

    centre_array = np.linspace(start * width, end * width, ncol)
    offs_array = 10 * np.random.rand(nrow)

    sim_data = np.zeros((nrow, ncol, specshape.size))
    sim1_data = np.zeros(
        sim_data.shape)  # extra spectra which will be aligned using zl
    sim2_data = np.zeros(
        sim_data.shape)  # extra spectra which will be aligned using zl
    real_shift = np.zeros((nrow, ncol))
    nind = 200  # the number of indices which are high
    for i in range(nrow):
        for j in range(ncol):
            if model_type == 'Gaussian':
                zl = Gaussian(specshape, amp, centre_array[j] + offs_array[i],
                              fwhm)
                zl1 = Gaussian(specshape, amp,
                               50 + centre_array[j] + offs_array[i], fwhm)
                zl2 = Gaussian(specshape, amp,
                               100 + centre_array[j] + offs_array[i], fwhm)

            else:
                zl = Lorentzian(specshape, amp,
                                centre_array[j] + offs_array[i], fwhm)
                zl1 = Lorentzian(specshape, amp,
                                 50 + centre_array[j] + offs_array[i], fwhm)
                zl2 = Lorentzian(specshape, amp,
                                 100 + centre_array[j] + offs_array[i], fwhm)

            zl.calculate()
            zl1.calculate()
            zl2.calculate()

            sim_data[i, j] = zl.data

            sim_data[i, j, :nind] = 1
            sim_data[i, j, -nind:] = 5
            sim1_data[i, j] = zl1.data
            sim1_data[i, j, :nind] = 1
            sim1_data[i, j, -nind:] = 5

            sim2_data[i, j] = zl2.data
            sim2_data[i, j, :nind] = 1
            sim2_data[i, j, -nind:] = 5
            real_shift[i, j] = centre_array[j] + offs_array[i]

    mshape = MultiSpectrumshape(specshape.dispersion, specshape.offset,
                                specshape.size, nrow, ncol)
    s = MultiSpectrum(mshape, data=sim_data)
    s1 = MultiSpectrum(mshape, data=sim1_data)
    s2 = MultiSpectrum(mshape, data=sim2_data)
    if return_shift:
        return s, s1, s2, real_shift
    else:
        return s, s1, s2


def test_init_other():
    s, _, _ = make_artificial_dataset(-2, 2)

    # wrong xsize
    mshape = MultiSpectrumshape(1, 10, 1024, s.xsize + 20, s.ysize)
    s1 = MultiSpectrum(mshape)
    with pytest.raises(ValueError):
        Align(s, other_spectra=[s1], cropping=True)

    # wrong ysize
    mshape = MultiSpectrumshape(1, 10, 1024, s.xsize, s.ysize + 20)
    s1 = MultiSpectrum(mshape)
    with pytest.raises(ValueError):
        Align(s, other_spectra=[s1], cropping=True)


def test_fast_align():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
        align = FastAlignZeroLoss(s)
        align.perform_alignment()
        # every maximum value is at zero energy
        assert np.all(align.aligned.energy_axis[
                          np.argmax(align.aligned.multidata, axis=2)].astype(
            'int') == 0)


def test_fast_align_1():
    """
    Align the fast zero loss using the align function which does the
    interpolation instead of roll
    """
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
        align = FastAlignZeroLoss(s, other_spectra=[s1, s2])
        align.determine_fast_shift()
        align.align()
        # every maximum value is at zero energy
        alg = align.aligned
        alg0 = align.aligned_others[0]
        alg1 = align.aligned_others[1]

        Em = alg.energy_axis[np.argmax(alg.multidata, axis=2)]
        Em0 = alg0.energy_axis[np.argmax(alg0.multidata, axis=2)]
        Em1 = alg1.energy_axis[np.argmax(alg1.multidata, axis=2)]

        assert np.all(np.abs(Em)) <= 1
        assert np.all(np.abs(Em0-50)) <= 1
        assert np.all(np.abs(Em1-100)) <= 1


def test_fast_align_crop():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
        align = FastAlignZeroLoss(s, other_spectra=[s1, s2], cropping=True)
        align.perform_alignment()

        assert np.min(align.aligned.multidata[:, :, 0]) == 1
        assert np.max(align.aligned.multidata[:, :, 0]) == 1
        assert np.min(align.aligned.multidata[:, :, -1]) == 5
        assert np.max(align.aligned.multidata[:, :, -1]) == 5

        assert np.min(align.aligned_others[0].multidata[:, :, 0]) == 1
        assert np.max(align.aligned_others[0].multidata[:, :, 0]) == 1
        assert np.min(align.aligned_others[0].multidata[:, :, -1]) == 5
        assert np.max(align.aligned_others[0].multidata[:, :, -1]) == 5

        assert np.min(align.aligned_others[1].multidata[:, :, 0]) == 1
        assert np.max(align.aligned_others[1].multidata[:, :, 0]) == 1
        assert np.min(align.aligned_others[1].multidata[:, :, -1]) == 5
        assert np.max(align.aligned_others[1].multidata[:, :, -1]) == 5


def test_fit_align():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
        align = AlignZeroLoss(s)
        align.perform_alignment()

        # every maximum value is almost at zero
        assert np.all(np.abs(
            s.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)


def test_fit_align_lorentz():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1],
                                            model_type='Lorentzian')
        align = AlignZeroLoss(s, model_type='Lorentzian')
        align.perform_alignment()

        # every maximum value is almost at zero
        assert np.all(np.abs(
            s.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)


def fit_align_other():
    s, s1, s2 = make_artificial_dataset(-2, 2)
    align = AlignZeroLoss(s, other_spectra=[s1, s2])
    align.perform_alignment()

    # every maximum value is almost at zero
    assert np.all(
        np.abs(s.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)
    assert np.all(np.abs((s1.energy_axis[
        np.argmax(align.aligned_others[0].multidata, axis=2)]) - 50) < 2)
    assert np.all(np.abs((s2.energy_axis[
        np.argmax(align.aligned_others[1].multidata, axis=2)]) - 100) < 2)


def fit_align_crop():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
        align = AlignZeroLoss(s, other_spectra=[s1, s2], cropping=True)
        align.perform_alignment()

        assert np.min(align.aligned.multidata[:, :, 0]) == 1
        assert np.max(align.aligned.multidata[:, :, 0]) == 1
        assert np.min(align.aligned.multidata[:, :, -1]) == 5
        assert np.max(align.aligned.multidata[:, :, -1]) == 5

        assert np.min(align.aligned_others[0].multidata[:, :, 0]) == 1
        assert np.max(align.aligned_others[0].multidata[:, :, 0]) == 1
        assert np.min(align.aligned_others[0].multidata[:, :, -1]) == 5
        assert np.max(align.aligned_others[0].multidata[:, :, -1]) == 5

        assert np.min(align.aligned_others[1].multidata[:, :, 0]) == 1
        assert np.max(align.aligned_others[1].multidata[:, :, 0]) == 1
        assert np.min(align.aligned_others[1].multidata[:, :, -1]) == 5
        assert np.max(align.aligned_others[1].multidata[:, :, -1]) == 5


def test_cross_correlation():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2, real_shift = make_artificial_dataset(shf_l[i][0],
                                                        shf_l[i][1],
                                                        return_shift=True)
        align = AlignCrossCorrelation(s, other_spectra=[s1, s2])
        align.perform_alignment()

        diff = np.abs(align.shift - (real_shift - real_shift[0, 0]))
        assert np.all(diff < 1)


def test_cross_correlation_signal_range():
    shf_l = [[-2, 2], [-4, -2], [2, 4]]
    for i in range(len(shf_l)):
        s, s1, s2, real_shift = make_artificial_dataset(shf_l[i][0],
                                                        shf_l[i][1],
                                                        return_shift=True)
        align = AlignCrossCorrelation(s, other_spectra=[s1, s2],
                                      signal_range=(-50, 50))
        align.perform_alignment()

        diff = np.abs(align.shift - (real_shift - real_shift[0, 0]))
        print(diff)
        assert np.all(diff < 1)


def main():
    test_init_other()
    test_fast_align()
    test_fast_align_1()
    test_fast_align_crop()
    test_fit_align()
    test_fit_align_lorentz()
    fit_align_other()
    fit_align_crop()
    test_cross_correlation()
    test_cross_correlation_signal_range()


if __name__ == "__main__":
    main()
