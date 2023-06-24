import h5py
import os
import numpy as np

class InputProcessing:
    def extractHRIR(subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['hrir_l']['trunc_64']
            # row = np.array(dset[pos_idx, :])
            row = np.array(dset)
        
        return row
    
    def extractPos(subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['srcpos']['trunc_64']
            #row = np.array(dset[pos_idx, :])
            row = np.array(dset)
        
        return row
    
    def extractAnthro(subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]
            #assume the first 8 meaurements are for left ear
            d = np.array(dset.attrs['D'])[:8]
            x = np.array(dset.attrs['X'])
            #assume the first 2 meaurements are for left ear
            theta = np.array(dset.attrs['theta'])[:2]
            row = np.hstack((d, x, theta))
        return row
    

        
