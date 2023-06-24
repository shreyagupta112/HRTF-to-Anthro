from model import *
from train import *
import torch

class Main:

    def main():

        torch.manual_seed(41)

        model = Model()

        trainer = ModelTrainer()

        trainer.trainModel(model)

        mape = trainer.basicValidation(model)

        print(mape)

    
Main.main()