import h5py
import os
import numpy as np

class InputProcessing:
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
    
    def extractAnthro(subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]
            d = np.array(dset.attrs['D'])
            x = np.array(dset.attrs['X'])
            theta = np.array(dset.attrs['theta'])
            row = np.hstack((d, x, theta))
            
        
        return row
    
    subject_number = 3
    row_index = 0
    hrir = extractRowFromHRIR(row_index, subject_number)
    position = extractPos(row_index, subject_number)
    anthro = extractAnthro(subject_number)
    print(anthro)
        
