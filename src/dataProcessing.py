import numpy as np
import torch
from sklearn.model_selection import train_test_split
from inputProcessing import *

class DataProcessing:

    def __init__(self):
        self.IP = InputProcessing()
        self.validSubjects = [3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
                         60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
                         148, 152, 153, 154, 155, 156, 162, 163, 165]
    
    # Split data by subjects: 70% of subjects for train, 20% of subjects for validation, 10% for testing
    def dataSplitTypeOne(self):
        total_subjects = len(self.validSubjects)

        trainSize = int(0.7 * total_subjects)
        validSize = int(0.2 * total_subjects)

        shuffled_data = np.random.permutation(self.validSubjects)
        
        trainSubjects = shuffled_data[:trainSize]
        validSubjects = shuffled_data[trainSize:validSize+trainSize]
        testSubjects = shuffled_data[trainSize + validSize:]

        X_train, Y_train = self.IP.extractData(trainSubjects)
        X_valid, Y_valid = self.IP.extractData(validSubjects)
        X_test, Y_test = self.IP.extractData(testSubjects)

        return X_train, Y_valid, X_test, Y_train, Y_valid, Y_test

DP = DataProcessing()
DP.crossValidationTypeOne()


    
