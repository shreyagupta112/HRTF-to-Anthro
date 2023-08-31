import h5py
import os
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd

# Class to extract data from HDF5 file
class InputProcessing:

    # blank constructor
    def __init__(self):
        pass

    # Todo: Normalize HRTF
    # Zero mean and unit variance
    # return an array representing the hrir from a single subject
    # Plot normalized HRTF and show channel before procedding
    def extractSingleHRIR(self, subject_num: int, dataType: str, normalize=True):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            # dset_right = f[subject]['hrir_r']['raw'] if dataType == "raw" else f[subject]['hrir_r']['trunc_64'] 
            dset_left = f[subject]['hrir_l']['raw'] if dataType == "raw" else f[subject]['hrir_l']['trunc_64']
            # row_right = np.array(dset_right)
            row_left = np.array(dset_left)
            left_hrtf, freq = self.FourierTransform(row_left)
            if normalize:
                left_hrtf = self.normalize(left_hrtf)
        #     right_hrtf = self.FourierTransform(row_right)
        # single_hrir = np.vstack((row_left, row_right))
        # single_hrtf = np.vstack((left_hrtf, right_hrtf))
        if dataType == "HRTF":
            return left_hrtf, freq
        else:
            return row_left


    
    # return an array representing the positions of a single subject
    def extractSinglePos(self, subject_num: int, cart=True):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')
        with h5py.File(file_path, "r") as f:
            dset = f[subject]['srcpos']['trunc_64']
            pos = np.array(dset)
        if cart:
            pos = self.sphericalToCartesian(pos)
        return pos
    
    # return an array with hrir and position of a single subject
    def extractSingleHrirAndPos(self, subject_num: int, dataType):
        hrir = self.extractSingleHRIR(subject_num, dataType, True)
        pos = self.extractSinglePos(subject_num, True)
        hrir_pos = np.hstack((hrir, pos))
        hrir_total = np.vstack((hrir_pos[0], hrir_pos[8]))
        # hrir_avg = np.mean(hrir_total, axis = 0)
        return hrir_total
    
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
                leftAnthro = np.tile(left_row, (2, 1))
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
        pos = self.extractSinglePos(3, True)
        anthro = np.hstack((anthro, pos))
        # pos = self.extractSinglePos(subjects[0])
        # anthro_pos = np.hstack((anthro, pos))
        for subject in subjects[1:]:
            currAnthro = self.extractSingleAnthro(subject, stack)

            pos = self.extractSinglePos(subject, True)
            currAnthro = np.hstack((currAnthro, pos))
            # currPos = self.extractSinglePos(subject)
            # currArray = np.hstack((currAnthro, currPos))
            anthro = np.vstack((anthro, currAnthro))
        return anthro

    # extract both hrir_pos and anthro for all subjects in subjects
    def extractData(self, subjects, dataType):
        # get first hrir, pos vector
        self.plotInput(subjects[0], [0, 8], dataType)
        hrir_pos = self.extractSingleHrirAndPos(subjects[0], dataType)
        # get first anthro vector
        anthro = self.extractSingleAnthro(subjects[0], True)
        # pos = self.extractSinglePos(subjects[0])
        # anthro_pos = np.hstack((anthro, pos))
        for subject in subjects[1:]:
            currHrirPosArray = self.extractSingleHrirAndPos(subject, dataType)
            currAnthroArray = self.extractSingleAnthro(subject, True)
            # currPosArray = self.extractSinglePos(subject)
            hrir_pos = np.vstack((hrir_pos, currHrirPosArray))
            # anthro = np.vstack((anthro, currAnthroArray))
            # curr_anthro_pos = np.hstack((currAnthroArray, currPosArray))
            anthro = np.vstack((anthro, currAnthroArray))
        
        
        return hrir_pos, anthro

    # perform FFT on data
    def FourierTransform(self, data):
        #Get the HRTF
        outputs_fft = np.fft.rfft(data, axis=1)
        n = len(data[0])
        sample_rate = 44.1
        print(n)
        freq = np.fft.rfftfreq(n, d=1/sample_rate)
        print(freq)
        # outputs_complex = np.zeros(np.shape(outputs_fft), dtype=outputs_fft.dtype)
        # for (s, h) in enumerate(outputs_fft):
        outputs_complex = outputs_fft/np.max(np.abs(outputs_fft))
        outputs_mag = abs(outputs_complex) 
        outputs_mag = 20.0*np.log10(outputs_mag)
        return outputs_mag, freq

    def normalize(self, data):
        norm_data = data
        for i in range(len(data)):
            mean = np.mean(data[i])
            std = np.std(data[i])
            norm_data[i] = (data[i] - mean) / std
        return norm_data

    # mostly taken from lab's code base
    def sphericalToCartesian(self, positions):
        "pos should be (#subjs, #positions, [azi, ele, r])"
        for i in range(len(positions)):
            pos_cart = [0]*3
            position = positions[i]
            pos_cart[0] = np.multiply(position[2], np.multiply(np.cos(position[1]/180 * math.pi), np.cos(position[0]/180 * math.pi)))
            pos_cart[1] = np.multiply(position[2], np.multiply(np.cos(position[1]/180 * math.pi), np.sin(position[0]/180 * math.pi)))
            pos_cart[2] = np.multiply(position[2], np.sin(position[1]/180 * math.pi))
            positions[i] = pos_cart
        return positions
    
    def plotInput(self, subject, positions, datatype):
        hrir, freq = self.extractSingleHRIR(subject, datatype, False)
        # freq = self.FourierTransform(hrir)[1]
        print(hrir)
        # hrir2 = self.extractSingleHrirAndPos(subject, datatype)
        hrir_plot = plt.figure()
        # plt.plot(range(len(hrir2[:1250])), hrir2[:1250], label = f"left hrtf average")
        for position in positions:
            plt.plot(freq, hrir[position], label = f"left hrtf at position {position}")
            #plt.plot(range(len(hrir[1250:][position])), hrir[1250:][position], label = f"right hrir at position {position}")
        plt.legend(loc="lower left")
        plt.ylabel("Magnitude")
        plt.xlabel("Frequency")
        plt.title(f"Center HRTF Plot for subject {subject} ")
        if not os.path.exists(f'../figures/inputs'):
            os.makedirs(f'../figures/inputs')
        hrir_plot.savefig(f'../figures/inputs/HRTF_subject_{subject}.png')
        plt.close()

        pos = self.extractSinglePos(subject)
        hrir_plot = plt.figure()
        for position in positions:
            plt.plot(range(len(pos[position])), pos[position], label = f" position {position}")
        plt.legend(loc="upper right")
        plt.ylabel("Pos")
        plt.xlabel("Time")
        plt.title(f"Center Pos Plot for subject {subject}")
        if not os.path.exists(f'../figures/inputs'):
            os.makedirs(f'../figures/inputs')
        hrir_plot.savefig(f'../figures/inputs/Pos_subject_{subject}.png')
        plt.close()

    def plotPositions(self):
        subject = 3
        cart = self.extractSinglePos(subject, True)
        sph = self.extractSinglePos(subject, False)
        dict = {}
        dict["spherical"] = []
        dict["cartesian"] = []
        for i in range(50):
            dict["spherical"].append(sph[i])
            dict["cartesian"].append(cart[i])
        df = pd.DataFrame(data=dict)
        df.to_csv('../figures/inputs/positions.csv', index=False)

IP = InputProcessing()
# hrtf, anthro = IP.extractData([3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
#                          60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
#                           148, 152, 153, 154, 155, 156, 162, 163, 165], "HRTF")

# # prediction = plt.plot()
# # plt.scatter(hrtf[::1250, 33], anthro[::1250, 0])
# # plt.ylabel("Anthro 0")
# # plt.xlabel("Hrtf 0")
# # plt.show()
# # plt.close()
# print(np.shape(hrtf))
# print(np.shape(anthro))

IP.plotInput(3, [0], "HRTF")