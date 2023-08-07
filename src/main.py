from model import *
from train_and_validate import *
from inputProcessing import *
from torchsummary import summary
import torch
import numpy as np
import math
import os

class Main:

    def __init__(self, splitType, dataType, activationFunction):
        self.splitType = splitType
        self.dataType = dataType
        self.activFunc = activationFunction
        if dataType == "HRTF":
            self.model = Model(activationFunction, 36)
        elif dataType == "raw":
            self.model = Model(activationFunction, 203)
        else:
            self.model = Model(activationFunction, 67)
        summary(self.model, input_size = (1, 1, 36))
        self.inputProcessing = InputProcessing()
        self.dataProcessing = DataProcessing()
        self.validSubjects = [3, 10, 18, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 
                         60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147,
                         148, 152, 153, 154, 155, 156, 162, 163, 165]
        self.trainer = ModelTrainer(self.splitType, self.dataType, self.activFunc)

    # This method will train the model
    def train(self):
        
        torch.manual_seed(41)

        self.trainer.trainModel(self.model)

    # This method will test the model
    def test(self, modelPath):
        self.deleteFiles(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/test')
        self.deleteFiles(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/train')
        self.deleteFiles(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/validation')
        mse = self.trainer.testModel(modelPath)

        print(mse)

    # This method will make a prediction for all valid subjects
    def predictAnthro(self, modelPath):
        self.deleteFiles(f"../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/train")
        self.deleteFiles(f"../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/validation")
        self.deleteFiles(f"../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/test")
        validSubjects = self.validSubjects
        trainSubjects, validationSubjects, testSubjects = self.dataProcessing.readSplits()
        anthro_prediction = []
        actual_anthro_pred = self.inputProcessing.extractAnthro(validSubjects, False)
        
        model = Model(self.activFunc)
        if self.dataType == "trunc64":
            model = Model(self.activFunc, 67)
        if self.dataType == "raw":
            model = Model(self.activFunc, 203)

        model.load_state_dict(torch.load(modelPath))

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
            prediction = plt.figure(figsize=(10, 6))
            plt.plot(range(len(anthro_prediction[i])), anthro_prediction[i], label = "prediction")
            plt.plot(range(len(actual_anthro_pred[i])), actual_anthro_pred[i], label = "actual")

            # Create a table for all prediction and actual anthro values
            anthro_prediction_truncated = [round(y, 3) for y in anthro_prediction[i].tolist()]
            anthro_actual_truncated = [round(y, 3) for y in actual_anthro_pred[i].tolist()]
            values_table = [[anthro, prediction, actual] for anthro, prediction, actual in zip(range(len(anthro_prediction_truncated)), anthro_prediction_truncated, anthro_actual_truncated)]
            column_labels = ['Anthro Point','Prediction', 'Actual']
            table = plt.table(cellText=values_table, colLabels=column_labels, loc='center right')

            # Adjust table properties
            table.auto_set_font_size(True)
            table.scale(0.3, 1)

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
                plt.title(f"Anthro Prediction for Subject {subjectNum} Left Ear")
                if not os.path.exists(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/{group}'):
                    os.makedirs(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/{group}')
                prediction.savefig(f'../figures/tanh/{self.dataType}/{self.splitType}/predictions/{group}/{subjectNum}_left_pred.png')
            else:
                plt.title(f"Anthro Prediction for Subject {subjectNum} Right Ear") 
                if not os.path.exists(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/{group}'):
                    os.makedirs(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/{group}')
                prediction.savefig(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/predictions/{group}/{subjectNum}_right_pred.png')
            plt.close()
    

    def deleteFiles(self, files):

        theFiles = os.listdir(files)

        for file in theFiles:
            filePath = os.path.join(files, file)
            try:
                if os.path.isfile(filePath):
                    os.remove(filePath)
            except Exception as e:
                print(f"Could not delete file {filePath}")
        




main = Main("split1", "HRTF", "tanh")
# main.train()
main.test('saved_model_tanh.pth')
# main.predictAnthro('saved_model_tanh.pth')
