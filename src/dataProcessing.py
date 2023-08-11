import numpy as np
import torch
from sklearn.model_selection import train_test_split
from inputProcessing import *
import re
import sys

# Class to deal with data processing
# Should only be constructed when training the model
class DataProcessing:

    def __init__(self):
        self.IP = InputProcessing()
        self.validSubjects = self.IP.validSubjects
    # create Subject splits and write it to the txt file
    def createSubjectSplit(self):
        total_subjects = len(self.validSubjects)

        trainSize = int(0.7 * total_subjects)
        validSize = int(0.2 * total_subjects)

        shuffled_data = np.random.permutation(self.validSubjects)
        
        self.trainSubjects = shuffled_data[:trainSize]
        self.validationSubjects = shuffled_data[trainSize:validSize+trainSize]
        self.testSubjects = shuffled_data[trainSize + validSize:] 


        self.writeSplits(self.trainSubjects, self.validationSubjects, self.testSubjects)

    # Split data by subjects: 70% of subjects for train, 20% of subjects for validation, 10% for testing
    def dataSplitTypeOne(self, dataType):

        self.createSubjectSplit()

        X_train, Y_train = self.IP.extractData(self.trainSubjects, dataType, True)
        X_valid, Y_valid = self.IP.extractData(self.validationSubjects, dataType, True)
        X_test, Y_test = self.IP.extractData(self.testSubjects, dataType, True)

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


    # write the subject split data to a txt file
    def writeSplits(self, train, validation, test):
        with open("splitData.txt", "w") as file:
            file.write(f"{train}\n")
            file.write(f"{validation}\n")
            file.write(f"{test}\n")
    
    # read the subject split data to a txt file
    def readSplits(self):
        with open('splitData.txt', 'r') as file:
            listContent = file.read()

        pattern = r'\d+'

        lists = []

        for line in listContent.strip().split('\n'):
            values = re.findall(pattern, line)
            int_values = [int(val) for val in values]
            lists.append(int_values)

        train = lists[0] + lists[1]
        validation = lists[2]
        test = lists[3]
        return train, validation, test


# DP = DataProcessing("HRTF")
# splits = DP.dataSplitTypeOne("HRTF")