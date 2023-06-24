import h5py
import os
import numpy as np

class Input:
    def extractRowFromHRIR(pos_idx, subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['hrir_l']['trunc_64']
            row = np.array(dset[pos_idx, :])
        
        return row
    
    def extractPos(pos_idx, subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['srcpos']['trunc_64']
            row = np.array(dset[pos_idx, :])
        
        return row
    
    subject_number = 3
    row_index = 0
    hrir = extractRowFromHRIR(row_index, subject_number)
    position = extractPos(row_index, subject_number)
    print(hrir)
    print(position)
        
