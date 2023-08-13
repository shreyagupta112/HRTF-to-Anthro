import h5py
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

# Class to extract data from HDF5 file
class InputProcessing:

    # blank constructor
    def __init__(self):
        self.validSubjects = [3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
                         60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
                         148, 152, 153, 154, 155, 156, 162, 163, 165]
        self.calcMeanSTD()
        
        # Extract initial data for the first subject
    def calcMeanSTD(self):
        HRTF = self.extractSingleHRIR(self.validSubjects[0], False, "HRTF")
        leftHRTF = HRTF[0:1250]
        rightHRTF = HRTF[1250:]
        POS = self.extractSinglePos(self.validSubjects[0])
        ANTHRO = self.extractSingleAnthro(self.validSubjects[0], False)
        leftAnthro = ANTHRO[0]
        rightAnthro = ANTHRO[1]

        # Loop through remaining subjects and accumulate data
        for subject in self.validSubjects[1:]:
            currHRTF = self.extractSingleHRIR(subject, False, "HRTF")
            currLeftHRTF = currHRTF[0:1250]
            currRightHRTF = currHRTF[1250:]
            leftHRTF = np.vstack((leftHRTF, currLeftHRTF))
            rightHRTF = np.vstack((rightHRTF, currRightHRTF))
            
            currPOS = self.extractSinglePos(subject)
            POS = np.vstack((POS, currPOS))
            
            currAnthro = self.extractSingleAnthro(subject, False)
            leftAnthro = np.vstack((leftAnthro, currAnthro[0]))
            rightAnthro = np.vstack((rightAnthro, currAnthro[1]))

        # Calculate means and standard deviations
        self.left_hrtf_mean = np.mean(leftHRTF, axis=0)
        self.left_hrtf_std = np.std(leftHRTF, axis=0)
        self.right_hrtf_mean = np.mean(rightHRTF, axis=0)
        self.right_hrtf_std = np.std(rightHRTF, axis=0)
        self.pos_mean = np.mean(POS, axis=0)
        self.pos_std = np.std(POS, axis=0)
        self.left_anthro_mean = np.mean(leftAnthro, axis=0)
        self.left_anthro_std = np.std(leftAnthro, axis=0)
        self.right_anthro_mean = np.mean(rightAnthro, axis=0)
        self.right_anthro_std = np.std(rightAnthro, axis=0)

    # Todo: Normalize HRTF
    # Zero mean and unit variance
    # return an array representing the hrir from a single subject
    # Plot normalized HRTF and show channel before procedding
    def extractSingleHRIR(self, subject_num: int, plot: bool, dataType: str):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            dset_right = f[subject]['hrir_r']['raw'] if dataType == "raw" else f[subject]['hrir_r']['trunc_64'] 
            dset_left = f[subject]['hrir_l']['raw'] if dataType == "raw" else f[subject]['hrir_l']['trunc_64']
            row_right = np.array(dset_right)
            row_left = np.array(dset_left)
            left_hrtf = self.FourierTransform(row_left)
            right_hrtf = self.FourierTransform(row_right)
        single_hrir = np.vstack((row_left, row_right))
        single_hrtf = np.vstack((left_hrtf, right_hrtf))
        if dataType == "HRTF":
            return single_hrtf
        else:
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
    def extractAnthro(self, subjects, stack, normalize=False):
        # get first anthro vector
        anthro = self.extractSingleAnthro(subjects[0], stack)
        if normalize == False:
            for subject in subjects[1:]:
                currArray = self.extractSingleAnthro(subject, stack)
                anthro = np.vstack((anthro, currArray))
        else:
            left_anthro, right_anthro = anthro[:1250], anthro[1250:]
            left_anthro = self.normalize(left_anthro, "leftAnthro")
            right_anthro = self.normalize(right_anthro, "rightAnthro")
            anthro = np.vstack((left_anthro, right_anthro))
            for subject in subjects[1:]:
                currArray = self.extractSingleAnthro(subject, stack)
                left_anthro, right_anthro = currArray[:1250], currArray[1250:]
                left_anthro = self.normalize(left_anthro, "leftAnthro")
                right_anthro = self.normalize(right_anthro, "rightAnthro")
                currArray = np.vstack((left_anthro, right_anthro))

                anthro = np.vstack((anthro, currArray))
            

        return anthro

    # extract both hrir_pos and anthro for all subjects in subjects
    def extractData(self, subjects, dataType, doNormalize=False):
        if doNormalize:
            comb_hrir = 0
            hrir_pos = 0
            anthro = 0

            for subject in subjects:
                currHrirArray = self.extractSingleHRIR(subject, False, dataType)
                currPosArray = self.extractSinglePos(subjects[0])
                currAnthroArray = self.extractSingleAnthro(subject, True)
                curr_left_anthro, curr_right_anthro = currAnthroArray[:1250], currAnthroArray[1250:]
                curr_left_hrir, curr_right_hrir = currHrirArray[:1250], currHrirArray[1250:]

                curr_left_anthro = self.normalize(curr_left_anthro, "leftAnthro")
                curr_right_anthro = self.normalize(curr_right_anthro, "rightAnthro")
                curr_left_hrir = self.normalize(curr_left_hrir, "leftHRTF")
                curr_right_hrir = self.normalize(curr_right_hrir, "rightHRTF")
                # currPosArray = self.normalize(currPosArray, "pos")

                new_comb_hrir = np.vstack((curr_left_hrir, curr_right_hrir))
                new_hrir_pos = np.hstack((new_comb_hrir, currPosArray))
                new_anthro = np.vstack((curr_left_anthro, curr_right_anthro))
                
                comb_hrir = new_comb_hrir if type(comb_hrir) == int else np.vstack((new_comb_hrir, comb_hrir))
                hrir_pos = new_hrir_pos if type(hrir_pos) == int else np.vstack((new_hrir_pos, hrir_pos))
                anthro = new_anthro if type(anthro) == int else np.vstack((new_anthro, anthro))

        else:
            anthro = self.extractSingleAnthro(subjects[0], True)
            hrir_pos = self.extractSingleHrirAndPos(subjects[0], dataType)
            # get first anthro vector
            for subject in subjects[1:]:
                currHrirPosArray = self.extractSingleHrirAndPos(subject, dataType)
                currAnthroArray = self.extractSingleAnthro(subject, True)
                hrir_pos = np.vstack((hrir_pos, currHrirPosArray))
                anthro = np.vstack((anthro, currAnthroArray))
        print("hrir", len(hrir_pos), len(hrir_pos[0]))
        print("anthro", len(anthro), len(anthro[0]))
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

    # normalize the data

    def normalize(self, data, data_type):
        type_mappings = {
            "pos": (self.pos_mean, self.pos_std),
            "leftHRTF": (self.left_hrtf_mean, self.left_hrtf_std),
            "rightHRTF": (self.right_hrtf_mean, self.right_hrtf_std),
            "leftAnthro": (self.left_anthro_mean, self.left_anthro_std),
            "rightAnthro": (self.right_anthro_mean, self.right_anthro_std)
        }
        mean, std = type_mappings.get(data_type, (1, 1))
        normalized_data = (data - mean) / std
        return normalized_data


# IP = InputProcessing()
# normalized_data = IP.extractSingleAnthro(3, False, True)
# data = IP.extractSingleAnthro(3, False, False)
# print(normalized_data)
# print(data)
