from model import *
from train_and_validate import *
from inputProcessing import *
import torch
import numpy as np

class Main:

    def __init__(self):
        self.model = Model()
        self.trainer = ModelTrainer()
        self.inputProcessing = InputProcessing()

    # This method will train and test the model
    def trainAndTest(self):

        torch.manual_seed(41)

        self.trainer.trainModel(self.model)

        mse = self.trainer.testModel(self.model)

        print(mse)
    # This method will make a prediction given a subject
    def predictAnthro(self, subject, side):
        
        input = torch.tensor(self.inputProcessing.extractSingleHrirAndPos(subject)).to(torch.float32)
        
        with torch.no_grad():
            # Make prediction
            anthro_pred = 0
            if side == "LEFT":
                anthro_pred = self.model.forward(input[0:1250])
            else:
                anthro_pred = self.model.forward(input[1250:])

            anthro_pred = torch.mean(anthro_pred, dim=0)
            
            return anthro_pred



main = Main()
main.trainAndTest()
