from model import *
from train_and_validate import *
from inputProcessing import *
import torch
import numpy as np

class Main:

    def __init__(self):
        self.model = Model()
        self.trainer = ModelTrainer()

    # This method will train and test the model
    def trainAndTest(self):

        torch.manual_seed(41)

        self.trainer.trainModel(self.model)

        mape = self.trainer.basicValidation(self.model)

        print(mape)

    # This method will make a prediction given a subject
    def predictAnthro(self, subject):
        hrir = InputProcessing.extractHRIR(subject)
        pos = InputProcessing.extractPos(subject)
        hrir_pos = np.hstack((hrir, pos))
        input = torch.FloatTensor(hrir_pos)
        input_normal = (input - self.trainer.hrirPos_mean) / (self.trainer.hrirPos_std)
        
        with torch.no_grad():
            # Make prediction
            anthro_pred = self.model.forward(input_normal)

            # Reverse the normalization
            reversed_normal_anthro = []
            for tensor in anthro_pred:
                reversed_vector = (tensor * self.trainer.hrirPos_std) + self.trainer.hrirPos_mean
                reversed_normal_anthro.append(reversed_vector)
            
            #Find the average predictions across all hrirs
            anthroPrediction = torch.mean(torch.stack(reversed_normal_anthro), dim=0).tolist()

            return anthroPrediction
            
        

    
main = Main()
main.trainAndTest()
anthroPred = main.predictAnthro(3)
print(anthroPred)