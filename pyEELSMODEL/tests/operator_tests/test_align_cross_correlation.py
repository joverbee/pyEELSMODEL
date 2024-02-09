"""
Testing the functionality of the zero loss alignment
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyEELSMODEL.core.multispectrum import MultiSpectrum, MultiSpectrumshape
from pyEELSMODEL.core.spectrum import Spectrum, Spectrumshape
from pyEELSMODEL.components.gaussian import Gaussian
from pyEELSMODEL.components.lorentzian import Lorentzian
from pyEELSMODEL.components.powerlaw import PowerLaw
from pyEELSMODEL.components.CLedge.hs_coreloss_edgecombined import HSCoreLossEdgeCombined
from pyEELSMODEL.components.CLedge.hydrogen_coreloss_edge import HydrogenicCoreLossEdge
from pyEELSMODEL.operators.alignzeroloss import AlignZeroLoss
from pyEELSMODEL.operators.fastalignzeroloss import FastAlignZeroLoss
from pyEELSMODEL.operators.aligncrosscorrelation import AlignCrossCorrelation

#Make a dataset where the centre changes.
#Every column has a incremental increase and between every row there is a random offset



def make_artificial_dataset(start, end, model_type="Gaussian", return_shift=False):
    """
    Make artificial datasets used to test the performance of the zero loss alignment
    :param start: The start value of the centering
    :param end:  The end value of the center
    :return:
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
    sim1_data = np.zeros(sim_data.shape)  # extra spectra which will be aligned using zl
    sim2_data = np.zeros(sim_data.shape)  # extra spectra which will be aligned using zl
    real_shift = np.zeros((nrow, ncol))
    nind = 200  # the number of indices which are high
    for i in range(nrow):
        for j in range(ncol):
            if model_type == 'Gaussian':
                zl = Gaussian(specshape, amp, centre_array[j] + offs_array[i], fwhm)
                zl1 = Gaussian(specshape, amp, 50 + centre_array[j] + offs_array[i], fwhm)
                zl2 = Gaussian(specshape, amp, 100 + centre_array[j] + offs_array[i], fwhm)

            else:
                zl = Lorentzian(specshape, amp, centre_array[j] + offs_array[i], fwhm)
                zl1 = Lorentzian(specshape, amp, 50 + centre_array[j] + offs_array[i], fwhm)
                zl2 = Lorentzian(specshape, amp, 100 + centre_array[j] + offs_array[i], fwhm)

            zl.calculate()
            zl1.calculate()
            zl2.calculate()

            sim_data[i, j] = np.random.poisson(zl.data)
            sim1_data[i, j] = np.random.poisson(zl1.data)
            sim2_data[i, j] = np.random.poisson(zl2.data)

            real_shift[i, j] = centre_array[j] + offs_array[i]

    mshape = MultiSpectrumshape(specshape.dispersion, specshape.offset, specshape.size, nrow, ncol)
    s = MultiSpectrum(mshape, data=sim_data)
    s1 = MultiSpectrum(mshape, data=sim1_data)
    s2 = MultiSpectrum(mshape, data=sim2_data)
    if return_shift:
        return s, s1, s2, real_shift
    else:
        return s, s1, s2




s, s1, s2, real_shift = make_artificial_dataset(-2, 2, return_shift=True)
s.plot()
align = AlignCrossCorrelation(s, cropping=True, signal_range=(-50, 50),interp=3)
# align = AlignCrossCorrelation(s, cropping=True)
# align.reference = s.multidata[0,50]
# align.perform_alignment()
align.determine_shift()
align.align()



plt.figure()
plt.plot(s.energy_axis, s.mean().data)
plt.plot(align.aligned.energy_axis, align.aligned.mean().data)
plt.plot(s.energy_axis, s.multidata[0,50])

plt.figure()
plt.plot(s.mean().get_interval((-50,50)).data)
plt.plot(align.aligned.mean().data)
plt.plot(align.reference)

fast = FastAlignZeroLoss(s, cropping=True)
fast.perform_alignment()
fast.aligned.sum().plot()
fast.show_shift()

plt.figure()
plt.plot(align.shift[0])
plt.plot(fast.shift[0])


# def test_init_other():
#     s, _, _ = make_artificial_dataset(-2, 2)
#
#     #wrong dispersion
#     mshape = MultiSpectrumshape(0.1,10,1024, s.xsize, s.ysize)
#     s1 = MultiSpectrum(mshape)
#     with pytest.raises(ValueError):
#         AlignZeroLoss(s, other_spectra=[s1])
#
#     #wrong xsize
#     mshape = MultiSpectrumshape(1,10,1024, s.xsize+20, s.ysize)
#     s1 = MultiSpectrum(mshape)
#     with pytest.raises(ValueError):
#         AlignZeroLoss(s, other_spectra=[s1])
#
#     #wrong ysize
#     mshape = MultiSpectrumshape(1,10,1024, s.xsize, s.ysize+20)
#     s1 = MultiSpectrum(mshape)
#     with pytest.raises(ValueError):
#         AlignZeroLoss(s, other_spectra=[s1])
#
#     #wrong Esize
#     mshape = MultiSpectrumshape(1,10,1024+20, s.xsize, s.ysize)
#     s1 = MultiSpectrum(mshape)
#     with pytest.raises(ValueError):
#         AlignZeroLoss(s, other_spectra=[s1])
#
# def test_fast_align():
#     shf_l = [[-2,2],[-4,-2],[2,4]]
#     for i in range(len(shf_l)):
#         s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
#         align = AlignZeroLoss(s)
#         align.fast_align()
#
#         # every maximum value is at zero energy
#         assert np.all(s.energy_axis[np.argmax(align.fast_aligned.multidata, axis=2)]== 0)
#
# def test_fast_align_other():
#     s, s1, s2 = make_artificial_dataset(-2, 2)
#     align = AlignZeroLoss(s, other_spectra=[s1,s2])
#     align.fast_align()
#
#     assert s.energy_axis[np.argmax(align.fast_aligned_others[0].sum().data)] == 50
#     assert s.energy_axis[np.argmax(align.fast_aligned_others[1].sum().data)] == 100
#
# def test_fast_align_crop():
#     shf_l = [[-2,2],[-4,-2],[2,4]]
#     for i in range(len(shf_l)):
#         s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
#         align = AlignZeroLoss(s, other_spectra=[s1,s2], cropping=True)
#         align.fast_align()
#
#         assert np.min(align.fast_aligned.multidata[:,:,0]) == 1
#         assert np.max(align.fast_aligned.multidata[:,:,0]) == 1
#         assert np.min(align.fast_aligned.multidata[:,:,-1]) == 5
#         assert np.max(align.fast_aligned.multidata[:,:,-1]) == 5
#
#         assert np.min(align.fast_aligned_others[0].multidata[:,:,0]) == 1
#         assert np.max(align.fast_aligned_others[0].multidata[:,:,0]) == 1
#         assert np.min(align.fast_aligned_others[0].multidata[:,:,-1]) == 5
#         assert np.max(align.fast_aligned_others[0].multidata[:,:,-1]) == 5
#
#         assert np.min(align.fast_aligned_others[1].multidata[:,:,0]) == 1
#         assert np.max(align.fast_aligned_others[1].multidata[:,:,0]) == 1
#         assert np.min(align.fast_aligned_others[1].multidata[:,:,-1]) == 5
#         assert np.max(align.fast_aligned_others[1].multidata[:,:,-1]) == 5
#
#
# def test_fit_align():
#     shf_l = [[-2,2],[-4,-2],[2,4]]
#     for i in range(len(shf_l)):
#         s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
#         align = AlignZeroLoss(s)
#         align.apply_shift()
#
#         # every maximum value is almost at zero
#         assert np.all(np.abs(s.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)
#
# def test_fit_align_lorentz():
#     shf_l = [[-2,2],[-4,-2],[2,4]]
#     for i in range(len(shf_l)):
#         s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1], model_type='Lorentzian')
#         align = AlignZeroLoss(s, model_type='Lorentzian')
#         align.apply_shift()
#
#         # every maximum value is almost at zero
#         assert np.all(np.abs(s.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)
#
#
# def fit_align_other():
#     s, s1, s2 = make_artificial_dataset(-2, 2)
#     align = AlignZeroLoss(s, other_spectra=[s1, s2])
#     align.apply_shift()
#
#     # every maximum value is almost at zero
#     assert np.all(np.abs(s.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)
#     assert np.all(np.abs((s1.energy_axis[np.argmax(align.aligned_others[0].multidata, axis=2)])-50) < 2)
#     assert np.all(np.abs((s2.energy_axis[np.argmax(align.aligned_others[1].multidata, axis=2)])-100) < 2)
#
#
# def fit_align_crop():
#     shf_l = [[-2,2],[-4,-2],[2,4]]
#     for i in range(len(shf_l)):
#         s, s1, s2 = make_artificial_dataset(shf_l[i][0], shf_l[i][1])
#         align = AlignZeroLoss(s, other_spectra=[s1,s2], cropping=True)
#         align.apply_shift()
#
#         assert np.min(align.aligned.multidata[:,:,0]) == 1
#         assert np.max(align.aligned.multidata[:,:,0]) == 1
#         assert np.min(align.aligned.multidata[:,:,-1]) == 5
#         assert np.max(align.aligned.multidata[:,:,-1]) == 5
#
#         assert np.min(align.aligned_others[0].multidata[:,:,0]) == 1
#         assert np.max(align.aligned_others[0].multidata[:,:,0]) == 1
#         assert np.min(align.aligned_others[0].multidata[:,:,-1]) == 5
#         assert np.max(align.aligned_others[0].multidata[:,:,-1]) == 5
#
#         assert np.min(align.aligned_others[1].multidata[:,:,0]) == 1
#         assert np.max(align.aligned_others[1].multidata[:,:,0]) == 1
#         assert np.min(align.aligned_others[1].multidata[:,:,-1]) == 5
#         assert np.max(align.aligned_others[1].multidata[:,:,-1]) == 5
#
# def fit_align_given_shift():
#     s, s1, s2, real_shift = make_artificial_dataset(-2, 2, return_shift=True)
#     align = AlignZeroLoss(s, other_spectra=[s1, s2], cropping=True)
#     align.apply_shift(shift=real_shift)
#
#     assert np.all(np.abs(align.aligned.energy_axis[np.argmax(align.aligned.multidata, axis=2)]) < 2)
#
#
# def main():
#     test_init_other()
#     test_fast_align()
#     test_fast_align_other()
#     test_fast_align_crop()
#     test_fit_align()
#     test_fit_align_lorentz()
#     fit_align_other()
#     fit_align_crop()
#     fit_align_given_shift()
#
# if __name__ == "__main__":
#     main()




