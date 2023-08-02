import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from inputProcessing import *
import numpy as np
from model import *
from dataProcessing import *
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __init__(self, splitType, dataType):
        self.DP = DataProcessing()
        self.validSubjects = self.DP.validSubjects
        self.splitType = splitType
        self.dataType = dataType
        
    # Method to train the model
    def trainModel(self, model):
        hrir_train, hrir_valid, hrir_test, anthro_train, anthro_valid, anthro_test = self.DP.dataSplitTypeOne(self.dataType)
        self.trainSubjects, self.validationSubjects, self.testSubjects = self.DP.readSplits()
        
        self.X_train = hrir_train
        self.X_valid = hrir_valid
        self.X_test = hrir_test
        self.Y_train = anthro_train
        self.Y_valid = anthro_valid
        self.Y_test = anthro_test

        print(self.trainSubjects)
        print(self.validationSubjects)
        print(self.testSubjects)

        # Get training data
        X_train = self.X_train
        anthro_train = self.Y_train
        X_valid = self.X_valid
        anthro_valid = self.Y_valid
        X_valid = self.X_valid
        anthro_valid = self.Y_valid
        X_test = self.X_test
        anthro_test = self.Y_test
        # Set loss function
        criterion = nn.MSELoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 200
        train_losses = []
        val_losses = []
        test_losses = []
        mse_train_data = []
        mse_validation_data = []
        mse_test_data = []
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

                # Find loss on test set
                anthro_test_pred = model(X_test)
                lossTestAnthro = criterion(anthro_test_pred, anthro_test)

                test_losses.append(lossTestAnthro.detach().numpy())

                # get individual MSE's for testing
                test_output = torch.mean((anthro_test_pred - anthro_test)**2, dim=0)
                mse_test_data.append(np.array(test_output.detach().numpy()))

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
        plt.plot(range(epochs), test_losses, label = "test losses")
        plt.legend(loc="upper right")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        trainLoss.savefig(f'../figures/{self.dataType}/{self.splitType}/error.png')
        plt.close()

        # Plot error for each anthro measurement
        for i in range(10):
            anthroMSE = plt.figure()
            mse_train_data = np.array(mse_train_data)
            mse_validation_data = np.array(mse_validation_data)
            mse_test_data = np.array(mse_test_data)
            mse_train = mse_train_data[:, i]
            mse_valid = mse_validation_data[:, i]
            mse_test = mse_test_data[:, i]

            plt.plot(range(epochs), mse_train, label = "training data")
            plt.plot(range(epochs), mse_valid, label = "validation data")
            plt.plot(range(epochs), mse_test, label = "test data")
            plt.legend(loc="upper right")

            ylabel = "MSE of Anthro Measure " + str(i)
            plotlabel = ylabel + " vs Epoch"
            figlabel = f"../figures/{self.dataType}/{self.splitType}/indivAnthro/" + str(i) + ".png"

            plt.ylabel(ylabel)
            plt.xlabel("Epoch")
            plt.title(plotlabel)
            anthroMSE.savefig(figlabel)
            plt.close()

     # Method to test the model
    def testModel(self, modelPath):
        # Load model from saved Path

        train, validation, test = self.DP.readSplits()

        X_test, anthro_test = InputProcessing().extractData(test, self.dataType)
        X_test = torch.tensor(X_test).to(torch.float32)
        anthro_test = torch.tensor(anthro_test).to(torch.float32)
        
        model = Model()
        if self.dataType == "trunc64":
            model = Model(67)
        if self.dataType == "raw":
            model = Model(203)

        model.load_state_dict(torch.load(modelPath))

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
                plt.plot(range(len(anthro_eval_at_i)), anthro_eval_at_i, label = "prediction")
                plt.plot(range(len(anthro_test_at_i)), anthro_test_at_i, label = "actual")
                plt.legend(loc="upper right")
                plt.ylabel("Measurement")
                plt.xlabel("HRIR")
                plt.title(f"Anthro Prediction for measurement{i}")
                prediction.savefig(f'../figures/{self.dataType}/{self.splitType}/test/{i}_pred.png')
                plt.close()
        return lossAnthro
