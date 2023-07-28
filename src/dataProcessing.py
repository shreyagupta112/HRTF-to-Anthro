import numpy as np
import torch
from sklearn.model_selection import train_test_split
from inputProcessing import *

# Class to deal with data processing
class DataProcessing:

    def __init__(self):
        self.IP = InputProcessing()
        self.validSubjects = [3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
                         60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
                         148, 152, 153, 154, 155, 156, 162, 163, 165]
    
    # Split data by subjects: 70% of subjects for train, 20% of subjects for validation, 10% for testing
    def dataSplitTypeOne(self, dataType):
        total_subjects = len(self.validSubjects)

        trainSize = int(0.7 * total_subjects)
        validSize = int(0.2 * total_subjects)

        shuffled_data = np.random.permutation(self.validSubjects)
        
        trainSubjects = shuffled_data[:trainSize]
        validSubjects = shuffled_data[trainSize:validSize+trainSize]
        testSubjects = shuffled_data[trainSize + validSize:]

        X_train, Y_train = self.IP.extractData(trainSubjects, dataType)
        X_valid, Y_valid = self.IP.extractData(validSubjects, dataType)
        X_test, Y_test = self.IP.extractData(testSubjects, dataType)
        
        X_train = torch.tensor(X_train).to(torch.float32)
        Y_train = torch.tensor(Y_train).to(torch.float32)
        X_valid = torch.tensor(X_valid).to(torch.float32)
        Y_valid = torch.tensor(Y_valid).to(torch.float32)
        X_test = torch.tensor(X_test).to(torch.float32)
        Y_test = torch.tensor(Y_test).to(torch.float32)

        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    
    # Split data by individual subject data: For each subject take 70% data for train, 20% for validation, 10% for test
    def dataSplitTypeTwo(self, dataType):
        # Get first split
        first_X = self.IP.extractSingleHrirAndPos(self.validSubjects[0], dataType)
        first_Y = self.IP.extractSingleAnthro(self.validSubjects[0], True)
        X_train, X_test, Y_train, Y_test = train_test_split(first_X, first_Y, test_size=0.3, random_state=41)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.33, random_state=41)
        # Get all other subject data
        for subject in self.validSubjects[1:]:
            curr_X = self.IP.extractSingleHrirAndPos(subject, dataType)
            curr_Y = self.IP.extractSingleAnthro(subject, True)
            currX_train, currX_test, currY_train, currY_test = train_test_split(curr_X, curr_Y, test_size=0.3, random_state=41)
            currX_valid, currX_test, currY_valid, currY_test = train_test_split(currX_test, currY_test, test_size=0.33, random_state=41) 
            X_train = np.vstack((X_train, currX_train))
            Y_train = np.vstack((Y_train, currY_train))
            X_valid = np.vstack((X_valid, currX_valid))
            Y_valid = np.vstack((Y_valid, currY_valid))
            X_test = np.vstack((X_test, currX_test))
            Y_test = np.vstack((Y_test, currY_test))

        X_train = torch.tensor(X_train).to(torch.float32)
        Y_train = torch.tensor(Y_train).to(torch.float32)
        X_valid = torch.tensor(X_valid).to(torch.float32)
        Y_valid = torch.tensor(Y_valid).to(torch.float32)
        X_test = torch.tensor(X_test).to(torch.float32)
        Y_test = torch.tensor(Y_test).to(torch.float32)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


    
