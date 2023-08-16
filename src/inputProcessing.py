import h5py
import os
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

# Class to extract data from HDF5 file
class InputProcessing:

    # blank constructor
    def __init__(self):
        pass

    # Todo: Normalize HRTF
    # Zero mean and unit variance
    # return an array representing the hrir from a single subject
    # Plot normalized HRTF and show channel before procedding
    def extractSingleHRIR(self, subject_num: int, dataType: str, normalize=False):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            # dset_right = f[subject]['hrir_r']['raw'] if dataType == "raw" else f[subject]['hrir_r']['trunc_64'] 
            dset_left = f[subject]['hrir_l']['raw'] if dataType == "raw" else f[subject]['hrir_l']['trunc_64']
            # row_right = np.array(dset_right)
            row_left = np.array(dset_left)
            left_hrtf = self.FourierTransform(row_left)
            if normalize:
                left_hrtf = self.normalize(left_hrtf)
        #     right_hrtf = self.FourierTransform(row_right)
        # single_hrir = np.vstack((row_left, row_right))
        # single_hrtf = np.vstack((left_hrtf, right_hrtf))
        if dataType == "HRTF":
            return left_hrtf
        else:
            return row_left


    
    # return an array representing the positions of a single subject
    def extractSinglePos(self, subject_num: int, cart=False, normalize=False):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset = f[subject]['srcpos']['trunc_64']
            pos = np.array(dset)
        # doubled_row = np.vstack((row, row))
        if cart:
            pos = self.sphericalToCartesian(pos)
        if normalize:
            pos = self.normalize(pos) 
        return pos
    
    # return an array with hrir and position of a single subject
    def extractSingleHrirAndPos(self, subject_num: int, dataType):
        hrir = self.extractSingleHRIR(subject_num, dataType, True)
        pos = self.extractSinglePos(subject_num, True)
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
            # right_ear = np.array(dset.attrs['D'])[8:]
            # assume the first 2 meaurements are for left ear
            left_pinna = np.array(dset.attrs['theta'])[:2]
            # right_pinna = np.array(dset.attrs['theta'])[2:]
            
            left_row = np.hstack((left_ear, left_pinna))
            # right_row = np.hstack((right_ear, right_pinna))

            if stack:
                leftAnthro = np.tile(left_row, (1250, 1))
                # rightAnthro = np.tile(right_row, (1250, 1)) 
                # combinedAnthro = np.vstack((leftAnthro, rightAnthro))
                return leftAnthro
            
            # combinedAnthro = np.vstack((left_row, right_row))
            
        return left_row
     
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

    def normalize(self, data):
        norm_data = data
        for i in range(len(data)):
            mean = np.mean(data[i])
            std = np.std(data[i])
            norm_data[i] = (data[i] - mean) / std
        return norm_data

    # taken from lab's code base
    def sphericalToCartesian(self, positions):
        "pos should be (#subjs, #positions, [azi, ele, r])"
        for position in positions:
            pos_cart = [0]*3
            pos_cart[0] = np.multiply(position[2], np.multiply(np.cos(position[1]/180 * math.pi), np.cos(position[0]/180 * math.pi)))
            pos_cart[1] = np.multiply(position[2], np.multiply(np.cos(position[1]/180 * math.pi), np.sin(position[0]/180 * math.pi)))
            pos_cart[2] = np.multiply(position[2], np.sin(position[1]/180 * math.pi))
            position = pos_cart
        return positions
    
    def plotInput(lines, labels):
        for i in range(len(lines)):
            plt.plot(range(i), lines(i), label = labels(i))
        plt.legend(loc="upper right")
        plt.ylabel("HRIR")
        plt.xlabel("Time")
        plt.title(f"Center HRIR Plot for subject")
        plt.show()
        plt.close()