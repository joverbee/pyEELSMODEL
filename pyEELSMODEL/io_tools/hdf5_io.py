import h5py
import numpy as np


def load_h5py(filename):
    f = h5py.File(filename, 'r')
    d = f['Energy_axis']
    dispersion = d.attrs['Dispersion']
    size = d.attrs['Size']
    offset = d.attrs['Offset']
    data = f['Data'][:]
    f.close()
    return data, dispersion, offset, size


def load_elements_and_edges(filename):
    f = h5py.File(filename, 'r')
    d = f['Energy_axis']
    edges = d.attrs['edges']
    elements = d.attrs['elements']
    f.close()
    return elements, edges


def load_hspy(filename):
    f = h5py.File(filename, 'r')
    d = f['Experiments']
    params = []
    haadf = []
    df = []
    detector_type = []

    for key in d.keys():
        if 'EELS Spectrum' in key:
            data = d[key]['data'][:]
            axis_num = data.ndim - 1

            if data.ndim == 2:
                data = data[:, np.newaxis, :]
            data = data.astype(float)

            ax_key = 'axis-' + str(axis_num)
            offset = d[key][ax_key].attrs['offset']
            dispersion = d[key][ax_key].attrs['scale']
            size = data.shape[-1]
            stra = 'Acquisition_instrument'
            acq_time = d[key]['metadata'][stra]['TEM'].attrs['exposure']
            params.append([data, dispersion, offset, size, acq_time])
        elif 'HAADF' in key:
            data = d[key]['data'][:]
            haadf.append(data)
            detector_type.append(key)
        elif 'DF4' in key:
            data = d[key]['data'][:]
            df.append(data)
            detector_type.append(key)
    f.close()

    if len(haadf) > 0:
        return params, haadf, detector_type
    elif len(df) > 0:
        return params, df, detector_type
    else:
        return params, df, detector_type
