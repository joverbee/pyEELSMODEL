import json

import h5py
import numpy as np

def open_emd(fname):
    ## Open the emd file to read
    emdFile = h5py.File(fname,'r')
    # Check for STEM images:
    try:
        image_keys=list(emdFile['Data']['Image'].keys())
    except:
        print("No scan images")
        image_keys=[]

    max_frame = 0
    for image_key in image_keys:
        image = emdFile['Data']['Image'][image_key]['Data'][:]
        max_frame = max(max_frame, image.shape[2])

    print(f"The number of frames in the scan images is: {max_frame}")
    # Check if the file contains EELS Feature
    grp_features = emdFile['Features']
    features = list(grp_features.keys())
    if 'SIFeature' in features:
        has_SI=True
    else:
        has_SI=False
    if has_SI:
        try:
            groupCubes = emdFile['Data/EelsSpectrumImage']
        except:
            msg=r"No 'Data/EelsSpectrumImage' present in this file"
            exit()
        print(f"Found {len(list(groupCubes.keys()))} EELS spectrum images")
        si_cubes_data=[]
        si_cubes_metadata=[]
        cube_names=[]
        # List of cubes
        for cube_name in list(groupCubes.keys()):
            cube = groupCubes[cube_name]
            cube_names.append(cube_name)
            cube_shape=cube['Data'].shape
            print("Cube: "+str(cube_name)+", shape:"+str(cube_shape))
            cube_data = np.empty(cube_shape, dtype=np.float32)
            for chunk in cube['Data'].iter_chunks():
                cube_data[chunk] = cube['Data'][chunk]
            si_cubes_data.append(np.copy(cube_data)-max_frame*100)
            metadata={}
            metadata_array = np.array(cube['Metadata'])
            metadata_char = ''.join([chr(i[0]) for i in metadata_array if i != 0])
            metadata_json = json.loads(metadata_char)
            try:
                acqusitionMetadata_array = np.array(cube['AcquisitionMetadata'])
            except:
                msg=r"This .emd file has no 'AcquisitionMetadata'. This script doesn't support this file format"
                exit()
            a =  acqusitionMetadata_array[:, [0]]
            a.reshape(a.shape[0])
            acqusitionMetadata =  ''.join([chr(i[0]) for i in a if i != 0])
            acqusitionMetadata_json = json.loads(acqusitionMetadata)
            for key in list(acqusitionMetadata_json['Data'].keys()):
                metadata[key]=acqusitionMetadata_json['Data'][key]
            metadata['raw_object']=metadata_json
            collection_angle_begin = float(metadata_json["CustomProperties"]["EnergyFilter.CollectionAngleRange.Begin"]["value"])
            collection_angle_end = float(metadata_json["CustomProperties"]["EnergyFilter.CollectionAngleRange.End"]["value"])
            metadata["collection_angle_begin"]=collection_angle_begin
            metadata["collection_angle_end"]=collection_angle_end
            beam_voltage = int(metadata_json["Optics"]["AccelerationVoltage"])
            metadata["beam_voltage"]=beam_voltage
            si_cubes_metadata.append(metadata)
        emdFile.close()
        cubes_data = []
        for i in range(len(si_cubes_data)):
            shape_before=str(si_cubes_data[i].shape)
            reshaped=np.transpose(np.array(si_cubes_data[i]),axes=(0,2,1))
            cubes_data.append(reshaped)

        return cubes_data, si_cubes_metadata