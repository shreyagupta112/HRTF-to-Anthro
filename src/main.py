from model import *
from train_and_validate import *
from inputProcessing import *
import torch
import numpy as np
import math
import os

class Main:

    def __init__(self, splitType, dataType):
        self.splitType = splitType
        self.dataType = dataType
        if dataType == "HRTF":
            self.model = Model(36)
        elif dataType == "raw":
            self.model = Model(203)
        else:
            self.model = Model(67)
        self.inputProcessing = InputProcessing()
        self.dataProcessing = DataProcessing()
        self.validSubjects = [3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
                         60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
                         148, 152, 153, 154, 155, 156, 162, 163, 165]

    # This method will train and test the model
    def trainAndTest(self):

        trainer = ModelTrainer(self.splitType, self.dataType)
        
        torch.manual_seed(41)

        trainer.trainModel(self.model)

        # mse = trainer.testModel(self.model)

        # print(mse)
    # This method will make a prediction for all valid subjects
    def predictAnthro(self):
        validSubjects = self.validSubjects
        trainSubjects, validationSubjects, testSubjects = self.dataProcessing.readSplits()
        anthro_prediction = []
        actual_anthro_pred = self.inputProcessing.extractAnthro(validSubjects, False)
        
        model = Model()
        if self.dataType == "trunc64":
            model = Model(67)
        if self.dataType == "raw":
            model = Model(203)

        torch.save(model.state_dict(), 'saved_model.pth')

        for subject in validSubjects:
            with torch.no_grad():
                # Make prediction
                input = torch.tensor(self.inputProcessing.extractHrirPos([subject], self.dataType)).to(torch.float32) 
                anthro_pred_left = model.forward(input[0:1250])
                anthro_pred_right = model.forward(input[1250:])
                anthro_pred_left = torch.mean(anthro_pred_left, dim=0)
                anthro_pred_right = torch.mean(anthro_pred_right, dim=0)
                anthro_prediction.append(anthro_pred_left)
                anthro_prediction.append(anthro_pred_right)
        
        # plot predicted vs actual for each subject
        for i in range(len(anthro_prediction)):
            prediction = plt.figure()
            plt.plot(range(len(anthro_prediction[i])), anthro_prediction[i], label = "prediction")
            plt.plot(range(len(actual_anthro_pred[i])), actual_anthro_pred[i], label = "actual")
            plt.legend(loc="upper right")
            plt.ylabel("Measurement")
            plt.xlabel("Anthro Point")
            index = math.floor(i/2)
            subjectNum = validSubjects[index]
            group = "N/A"
            if subjectNum in trainSubjects:
                group = "train"
            elif subjectNum in validationSubjects:
                group = "validation"
            elif subjectNum in testSubjects:
                group = "test"
            else:
                group = "N/A"
            if i % 2 == 0:
                plt.title(f"Anthro Prediction for Subject {subjectNum} Right Ear")
                prediction.savefig(f'../figures/{self.dataType}/{self.splitType}/predictions/{group}/{subjectNum}_right_pred.png')
            else:
                plt.title(f"Anthro Prediction for Subject {subjectNum} Left Ear") 
                prediction.savefig(f'../figures/{self.dataType}/{self.splitType}/predictions/{group}/{subjectNum}_left_pred.png')
            plt.close()
    

    '''
    def deletePredictionFiles(self):
        trainFilePath = "../figures/HRTF/split1/predictions/train"
        validFilePath = "../figures/HRTF/split1/predictions/validation"
        testFilePath = "../figures/HRTF/split1/predictions/test" 

        trainFiles = os.listdir(trainFilePath)
        validFiles = os.listdir(validFilePath)
        testFiles = os.listdir(testFilePath) 

        for file in trainFiles:
            filePath = os.path.join(trainFilePath, file)
            try:
                if os.path.isfile(filePath):
                    os.remove(filePath)
            except Exception as e:
                print(f"Could not delete file {filePath}")
        
        for file in validFiles:
            filePath = os.path.join(trainFilePath, file)
            try:
                if os.path.isfile(filePath):
                    os.remove(filePath)
            except Exception as e:
                print(f"Could not delete file {filePath}")
        
        for file in testFiles:
            filePath = os.path.join(trainFilePath, file)
            try:
                if os.path.isfile(filePath):
                    os.remove(filePath)
            except Exception as e:
                print(f"Could not delete file {filePath}")
    '''
        




main = Main("split1", "HRTF")
# main.deletePredictionFiles()
# main.trainAndTest()
main.predictAnthro()
