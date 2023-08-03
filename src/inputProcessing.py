import h5py
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

# Class to extract data from HDF5 file
class InputProcessing:

    # blank constructor
    def __init__(self):
        pass

    # return an array representing the hrir from a single subject
    def extractSingleHRIR(self, subject_num: int, dataType: str):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset_right = f[subject]['hrir_r']['raw'] if dataType == "raw" else f[subject]['hrir_r']['trunc_64'] 
            dset_left = f[subject]['hrir_l']['raw'] if dataType == "raw" else f[subject]['hrir_l']['trunc_64']
            print(len(dset_right), len(dset_right[0]))
            row_right = np.array(dset_right)
            row_left = np.array(dset_left)
            left_hrtf = self.FourierTransform(row_left)
            right_hrtf = self.FourierTransform(row_right)
        single_hrir = np.vstack((row_left, row_right))
        single_hrtf = np.vstack((left_hrtf, right_hrtf))

        # Plot hrir and hrtf 
        if dataType == "HRTF":
            return single_hrtf
        return single_hrir
    
    # return an array representing the positions of a single subject
    def extractSinglePos(self, subject_num: int):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['srcpos']['trunc_64']
            row = np.array(dset)
        doubled_row = np.vstack((row, row))
        
        return doubled_row
    
    # return an array with hrir and position of a single subject
    def extractSingleHrirAndPos(self, subject_num: int, dataType):
        hrir = self.extractSingleHRIR(subject_num, False, dataType)
        pos = self.extractSinglePos(subject_num)
        hrir_pos = np.hstack((hrir, pos))
        return hrir_pos
    
    # Make it zero mean unit variance normalization
    # return an array representing the anthropometric data from a single subject
    def extractSingleAnthro(self, subject_num, stack: bool):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]
            # assume the first 8 meaurements are for left ear
            left_ear = np.array(dset.attrs['D'])[:8]
            right_ear = np.array(dset.attrs['D'])[8:]
            # assume the first 2 meaurements are for left ear
            left_pinna = np.array(dset.attrs['theta'])[:2]
            right_pinna = np.array(dset.attrs['theta'])[2:]
            
            left_row = np.hstack((left_ear, left_pinna))
            right_row = np.hstack((right_ear, right_pinna))

            if stack:
                leftAnthro = np.tile(left_row, (1250, 1))
                rightAnthro = np.tile(right_row, (1250, 1)) 
                combinedAnthro = np.vstack((leftAnthro, rightAnthro))
                return combinedAnthro
            
            combinedAnthro = np.vstack((left_row, right_row))
            
        return combinedAnthro
     
    # extract HRIR and Pos for all subjects in subjects list
    def extractHrirPos(self, subjects, dataType):
        # get first hrir, pos vector
        hrir_pos = self.extractSingleHrirAndPos(subjects[0], dataType)
        for subject in subjects[1:]:
            currArray = self.extractSingleHrirAndPos(subject, dataType)
            hrir_pos = np.vstack((hrir_pos, currArray))
        return hrir_pos
    
    # extract Anthro data for all subjects in subjects
    def extractAnthro(self, subjects, stack: bool):
        # get first anthro vector
        anthro = self.extractSingleAnthro(subjects[0], stack)
        for subject in subjects[1:]:
            currArray = self.extractSingleAnthro(subject, stack)
            anthro = np.vstack((anthro, currArray))
        return anthro

    # extract both hrir_pos and anthro for all subjects in subjects
    def extractData(self, subjects, dataType):
        # get first hrir, pos vector
        hrir_pos = self.extractSingleHrirAndPos(subjects[0], dataType)
        # get first anthro vector
        anthro = self.extractSingleAnthro(subjects[0], True)
        for subject in subjects[1:]:
            currHrirPosArray = self.extractSingleHrirAndPos(subject, dataType)
            currAnthroArray = self.extractSingleAnthro(subject, True)
            hrir_pos = np.vstack((hrir_pos, currHrirPosArray))
            anthro = np.vstack((anthro, currAnthroArray))
        return hrir_pos, anthro

    # perform FFT on data
    def FourierTransform(self, data):
        #Get the HRTF
        outputs_fft = np.fft.rfft(data, axis=1)
        # outputs_complex = np.zeros(np.shape(outputs_fft), dtype=outputs_fft.dtype)
        # for (s, h) in enumerate(outputs_fft):
        outputs_complex = outputs_fft/np.max(np.abs(outputs_fft))
        outputs_mag = abs(outputs_complex) 
        outputs_mag = 20.0*np.log10(outputs_mag)
        return outputs_mag

# IP = InputProcessing()
# fft = IP.extractSingleHRIR(3, True)
