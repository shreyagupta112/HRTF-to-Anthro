import h5py
import os
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

# Class to extract data from HDF5 file
class InputProcessing:

    # blank constructor
    def __init__(self):
        anthroValues = self.extractAnthro([3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
                         60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
                          148, 152, 153, 154, 155, 156, 162, 163, 165], False)
        self.anthroMean = np.mean(anthroValues, axis=0)

    # Todo: Normalize HRTF
    # Zero mean and unit variance
    # return an array representing the hrir from a single subject
    # Plot normalized HRTF and show channel before procedding
    def extractSingleHRIR(self, subject_num: int, dataType: str, normalize=True, peaks=True):
        subject = 'subject_' + str(subject_num).zfill(3)
        file_path =  os.path.join('..','data','cipic.hdf5')

        with h5py.File(file_path, "r") as f:
            # dset_right = f[subject]['hrir_r']['raw'] if dataType == "raw" else f[subject]['hrir_r']['trunc_64'] 
            dset_left = f[subject]['hrir_l']['raw'] if dataType == "raw" else f[subject]['hrir_l']['trunc_64']
            # row_right = np.array(dset_right)
            row_left = np.array(dset_left)
            left_hrtf, freq = self.FourierTransform(row_left)
            minHrtf, maxHrtf = self.getHrtfRange(10, freq)
            if normalize:
                left_hrtf = self.normalize(left_hrtf)
            if peaks:
                peak_array = 0
                for row in left_hrtf:
                    peaks = self.getPeaks(row, freq)
                    valleys = self.getPeaks(-row, freq)
                    currArray = np.zeros(33)
                    for i in peaks:
                        currArray[i] = row[i]
                    for i in valleys:
                        currArray[i] = row[i]
                    # if isinstance(peak_array, int):
                    #     print(currArray)
                    #     print(peaks)
                    #     print(freq[peaks])
                    peak_array = currArray if isinstance(peak_array, int) else np.vstack((peak_array, currArray))
                return peak_array[:,minHrtf:maxHrtf]
        #     right_hrtf = self.FourierTransform(row_right)
        # single_hrir = np.vstack((row_left, row_right))
        # single_hrtf = np.vstack((left_hrtf, right_hrtf))
        if dataType == "HRTF":
            return left_hrtf[:,minHrtf:maxHrtf], freq
        else:
            return row_left
    
    def getHrtfRange(self, measurement, freq):
        if measurement == 10:
            return 0, 33
        
        meanValue = self.anthroMean
        maxRange = (345 / (meanValue)) / 10
        minRange = (345 / ((meanValue * 2))) / 10
        ind = []
        for i in range(len(freq)):
            if freq[i] >= minRange and freq[i] <= maxRange:
                ind.append(i)
        
        return min(ind), max(ind) + 1

    def getPeaks(self, row, freq):
        peaks, _ = find_peaks(row, height=0)
        if len(peaks) == 1:
            return peaks
        frequencies = freq[peaks]
        currArray = []
        for i in range(len(frequencies)):
            if i == 0:
                if frequencies[i+1] - frequencies[i] > 1:
                    currArray.append(peaks[i])
            elif i == len(frequencies) - 1:
                if frequencies[i] - frequencies[i-1] > 1:
                    currArray.append(peaks[i])
            else:
                if frequencies[i+1] - frequencies[i] > 1 and frequencies[i] - frequencies[i-1] > 1:
                    currArray.append(peaks[i])
        return currArray


    
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
        hrir_total = np.vstack((hrir_pos[0], hrir_pos[8], hrir_pos[16], hrir_pos[24]))
        # hrir_avg = np.mean(hrir_total, axis = 0)
        return hrir_pos[0:50]
    
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

            left_row = np.array(dset.attrs['D'])[2]

            if stack:
                leftAnthro = np.tile(left_row, (50, 1))
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
        # pos = self.extractSinglePos(3, True)
        # anthro = np.hstack((anthro, pos))
        # pos = self.extractSinglePos(subjects[0])
        # anthro_pos = np.hstack((anthro, pos))
        for subject in subjects[1:]:
            currAnthro = self.extractSingleAnthro(subject, stack)

            # pos = self.extractSinglePos(subject, True)
            # currAnthro = np.hstack((currAnthro, pos))
            # currPos = self.extractSinglePos(subject)
            # currArray = np.hstack((currAnthro, currPos))
            anthro = np.vstack((anthro, currAnthro))
        return anthro

    # extract both hrir_pos and anthro for all subjects in subjects
    def extractData(self, subjects, dataType):
        # get first hrir, pos vector
        # self.plotInput(subjects[0], [0, 8, 16, 24], dataType)
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
        freq = np.fft.rfftfreq(n, d=1/sample_rate)
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
        hrir, freq = self.extractSingleHRIR(subject, datatype, True, False)
        hrir_plot = plt.figure()
        for position in positions:
            single_hrir = hrir[position]
            plt.plot(freq, single_hrir, label = f"left hrtf at position {position}")
            # peaks, _ = find_peaks(single_hrir, height=0)
            # dips, _ = find_peaks(-single_hrir, height=0)
            peaks = self.getPeaks(single_hrir, freq)
            dips = self.getPeaks(-single_hrir, freq)
            plt.plot(freq[peaks], single_hrir[peaks], "x")
            plt.plot(freq[dips], single_hrir[dips], "o")
            #plt.plot(range(len(hrir[1250:][position])), hrir[1250:][position], label = f"right hrir at position {position}")
        # single_hrir = hrir[0]
        # plt.plot(freq, single_hrir, label = f"left hrtf at position {0}")
        # # peaks, _ = find_peaks(single_hrir, height=0)
        # # dips, _ = find_peaks(-single_hrir, height=0)
        # peaks = self.getPeaks(single_hrir, freq)
        # dips = self.getPeaks(-single_hrir, freq)
        # plt.plot(freq[peaks], single_hrir[peaks], "x")
        # plt.plot(freq[dips], single_hrir[dips], "o")
        plt.plot(freq, np.zeros_like(hrir[0]), "--", color="gray")
        plt.legend(loc="lower left")
        plt.ylabel("Magnitude")
        plt.xlabel("Frequency")
        plt.title(f"HRTF Plot for subject {subject} at positions 0, 8, 16, 24")
        if not os.path.exists(f'../figures/inputs'):
            os.makedirs(f'../figures/inputs')
        hrir_plot.savefig(f'../figures/inputs/HRTF_subject_{subject}_4_Pos.png')
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

        peak = self.extractSingleHRIR(subject, datatype)

        peak_plot = plt.figure()
        for position in positions:
            currPeak = peak[position]
            plt.plot(freq, currPeak, label = f"left HRTF peaks at position {position}")
        plt.legend(loc="lower left")
        plt.ylabel("Peak exists")
        plt.xlabel("Frequency")
        plt.title(f"Frequnecy at which peak exists plot for subject {subject} at positions 0, 8, 16, 24")
        if not os.path.exists(f'../figures/inputs'):
            os.makedirs(f'../figures/inputs')
        peak_plot.savefig(f'../figures/inputs/Peaks_subject_{subject}_4_Pos.png')
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

#IP.plotInput(3, [0], "HRTF")

# IP.extractSingleHRIR(3, "HRTF")

# IP = InputProcessing()
# # anthro = IP.extractAnthro([3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
# #                          60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
# #                           148, 152, 153, 154, 155, 156, 162, 163, 165], False)
# # meanValue = np.mean(anthro, axis=0)
# # print(meanValue)

# # m, mx = IP.getHrtfRange(4, 2)
# # print(m)
# # print(mx)

# Hrtf = IP.extractSingleHRIR(3, "HRTF")
# print(np.shape(Hrtf))
