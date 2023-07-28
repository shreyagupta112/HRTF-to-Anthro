from model import *
from train_and_validate import *
from inputProcessing import *
import torch
import numpy as np
import math

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
        self.trainer = ModelTrainer(splitType, dataType)
        self.inputProcessing = InputProcessing()
        self.dataProcessing = DataProcessing()

    # This method will train and test the model
    def trainAndTest(self):

        torch.manual_seed(41)

        self.trainer.trainModel(self.model)

        mse = self.trainer.testModel(self.model)

        print(mse)
    # This method will make a prediction for all valid subjects
    def predictAnthro(self):
        validSubjects = self.dataProcessing.validSubjects
        anthro_prediction = []
        actual_anthro_pred = self.inputProcessing.extractAnthro(validSubjects, False)
        

        for subject in validSubjects:
            with torch.no_grad():
                # Make prediction
                input = torch.tensor(self.inputProcessing.extractHrirPos([subject], self.dataType)).to(torch.float32) 
                anthro_pred_left = self.model.forward(input[0:1250])
                anthro_pred_right = self.model.forward(input[1250:])
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
            if i % 2 == 0:
                plt.title(f"Anthro Prediction for Subject {validSubjects[index]} Right Ear")
                prediction.savefig(f'../figures/{self.dataType}/{self.splitType}/predictions/{validSubjects[index]}_right_pred.png')
            else:
                plt.title(f"Anthro Prediction for Subject {validSubjects[index]} Left Ear") 
                prediction.savefig(f'../figures/{self.dataType}/{self.splitType}/predictions/{validSubjects[index]}_left_pred.png')
            plt.close()



main = Main("split1", "HRTF")
main.trainAndTest()
main.predictAnthro()
