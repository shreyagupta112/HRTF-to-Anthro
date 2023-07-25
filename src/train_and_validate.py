import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from inputProcessing import *
import numpy as np
from dataProcessing import *
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __init__(self, ):
        self.DP = DataProcessing()
        hrir_train, hrir_valid, hrir_test, anthro_train, anthro_valid, anthro_test = self.DP.dataSplitTypeOne()
        self.X_train = hrir_train
        self.X_valid = hrir_valid
        self.X_test = hrir_test
        self.Y_train = anthro_train
        self.Y_valid = anthro_valid
        self.Y_test = anthro_test
       
    # Method to train the model
    def trainModel(self, model):
        # Get training data
        X_train = self.X_train
        anthro_train = self.Y_train
        X_valid = self.X_valid
        anthro_valid = self.Y_valid
        X_valid = self.X_valid
        anthro_valid = self.Y_valid
        # Set loss function
        criterion = nn.MSELoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 80
        train_losses = []
        val_losses = []
        mse_train_data = []
        mse_validation_data = []
        min_valid_loss = np.inf
        for i in range(epochs):
            model.train()
            # propgate forward
            anthro_pred = model.forward(X_train)
            #calculate loss
            lossAnthroTrain = criterion(anthro_pred, anthro_train)
            #calculate individual losses
            indiv_mse = torch.mean((anthro_pred - anthro_train)**2, dim=0)
            #Keep track of losses
            train_losses.append(lossAnthroTrain.detach().numpy())
            mse_train_data.append(np.array(indiv_mse.detach().numpy()))
            
            #Do some backward propagation
            optimizer.zero_grad()
            lossAnthroTrain.backward()
            optimizer.step()
            train_loss = lossAnthroTrain.item()

            # Get the MSE of the validation data without chaning the weights (Cross Validation)
            model.eval()   
            with torch.no_grad():
                # Find loss for validation
                anthro_val_pred = model(X_valid)
                lossValAnthro = criterion(anthro_val_pred, anthro_valid)
            
                val_losses.append(lossValAnthro.detach().numpy())
                
                # get individual MSE's for validation
                val_output = torch.mean((anthro_val_pred - anthro_valid)**2, dim=0) 
                mse_validation_data.append(np.array(val_output.detach().numpy()))

                valid_loss = lossValAnthro.item()

                print(f'Epoch {i+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
            # Update validation loss
            if min_valid_loss > lossValAnthro:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}-->{lossValAnthro:.6f}) \t Saving The Model')
                min_valid_loss = lossValAnthro
                # save current state of model
                torch.save(model.state_dict(), 'saved_model.pth')

        # Plot total error per epoch
        trainLoss = plt.figure()
        plt.plot(range(epochs), train_losses, label = "training losses")
        plt.plot(range(epochs), val_losses, label = "validation losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        trainLoss.savefig('../figures/error.png')

        # Plot error for each anthro measurement
        for i in range(10):
            anthroMSE = plt.figure()
            mse_train_data = np.array(mse_train_data)
            mse_validation_data = np.array(mse_validation_data)
            mse_train = mse_train_data[:, i]
            mse_valid = mse_validation_data[:, i]

            plt.plot(range(epochs), mse_train, label = "training data")
            plt.plot(range(epochs), mse_valid, label = "validation data")

            ylabel = "MSE of Anthro Measure " + str(i)
            plotlabel = ylabel + " vs Epoch"
            figlabel = "../figures/" + str(i) + ".png"

            plt.ylabel(ylabel)
            plt.xlabel("Epoch")
            plt.title(plotlabel)
            anthroMSE.savefig(figlabel)

     # Method to test the model
    def testModel(self, model):
        X_test = self.X_test
        anthro_test = self.anthro_test
        criterion = nn.MSELoss()
        with torch.no_grad():
            # calculate MSE for whole predicition vector
            anthro_eval = model.forward(X_test) # X-test are features from test se, y_eval s predictions
            lossAnthro = criterion(anthro_eval, anthro_test) 

            # plot predicted vs actual for each anthropometric data point
            for i in range(len(anthro_eval[0])):
                prediction = plt.figure()
                anthro_eval_at_i = []
                anthro_test_at_i = []
                for j in range(len(anthro_eval)):
                    anthro_eval_at_i.append(anthro_eval[j][i])
                    anthro_test_at_i.append(anthro_test[j][i])
                plt.plot(range(len(anthro_eval)), anthro_eval_at_i, label = "prediction")
                plt.plot(range(len(anthro_test)), anthro_test_at_i, label = "actual")
                plt.ylabel("Measurement")
                plt.xlabel("HRIR")
                plt.title(f"Anthro Prediction for measurement{i}")
                prediction.savefig(f'../figures/{i}_pred.png')
        return lossAnthro
