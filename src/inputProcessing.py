import h5py
import os
import numpy as np

class InputProcessing:
    # return an array representing the hrir from a single subject
    def extractHRIR(subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['hrir_l']['raw']
            row = np.array(dset)
        
        return row
    
    # return an array representing the positions of a single subject
    def extractPos(subject_num):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['srcpos']['raw']
            row = np.array(dset)
        
        return row
    
    # return an array representing the anthropometric data from a single subject
    def extractAnthro(subject_num, just_left_ear):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]
            if just_left_ear == True:
                # assume the first 8 meaurements are for left ear
                ear = np.array(dset.attrs['D'])[:8]
                # assume the first 2 meaurements are for left ear
                pinna = np.array(dset.attrs['theta'])[:2]
            else:
                ear = np.array(dset.attrs['D'])
                pinna = np.array(dset.attrs['theta'])
            row = np.hstack((ear, pinna))
        return row
    

        
