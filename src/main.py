from model import *
from train import *
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
    def predictAnthroPos(self, subject):
        hrir = InputProcessing.extractHRIR(subject)
        hrir = torch.FloatTensor(hrir)
        hrir_normal = torch.nn.functional.normalize(hrir, p=2.0, dim=1)
        with torch.no_grad():
            # Make prediction
            anthro_pred, pos_pred = self.model.forward(hrir_normal)
            
            # Reverse the normalization
            reversed_normal_anthro = []
            for tensor in anthro_pred:
                lp_norm = np.linalg.norm(tensor, 2)
                reversed_vector = tensor * (1.0/lp_norm)
                reversed_normal_anthro.append(reversed_vector)

            reversed_normal_pos = []
            for tensor in pos_pred:
                lp_norm = np.linalg.norm(tensor, 2)
                reversed_vector = tensor * (1.0/lp_norm)
                reversed_normal_pos.append(reversed_vector)
            
            #Find the average predictions across all hrirs
            anthroPrediction = torch.mean(torch.stack(reversed_normal_anthro), dim=0).tolist()
            posPrediction = torch.mean(torch.stack(reversed_normal_pos), dim=0).tolist()

            return anthroPrediction, posPrediction
            

    
main = Main()
main.trainAndTest()
anthroPred, posPred = main.predictAnthroPos(3)
print(anthroPred)
print(posPred)