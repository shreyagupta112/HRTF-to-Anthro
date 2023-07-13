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

            # reverse normalization
            anthro_pred = (anthro_pred * self.trainer.anthro_std) + self.trainer.anthro_mean

            anthro_pred = torch.mean(anthro_pred, dim=0)
            
            return anthro_pred
            
        


main = Main()
main.trainAndTest()
anthroPred = main.predictAnthro(3)
print(anthroPred)