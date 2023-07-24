import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from inputProcessing import *
import numpy as np
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __init__(self, ):
        hrir = InputProcessing.extractHRIR(3)
        pos = InputProcessing.extractPos(3)
        hrir_pos = np.hstack((hrir, pos))
        anthro = InputProcessing.extractAnthro(3, True)
        anthro = np.tile(anthro, (1250,1))
        # Split the data into training and test sets (70% training, 30% test)
        hrir_pos_train, hrir_pos_test, anthro_train, anthro_test = train_test_split(
            hrir_pos, anthro, test_size=0.3, random_state=41)

        # Split the test set again to get validation set (20% validation, 10% test)
        hrir_pos_valid, hrir_pos_test, anthro_valid, anthro_test = train_test_split(
            hrir_pos_test, anthro_test, test_size=0.33, random_state=41)

        # get mean and standard deviation
        self.anthro_mean = np.mean(anthro)
        self.anthro_std = np.std(anthro)
        self.hrirPos_mean = np.mean(hrir_pos)
        self.hrirPos_std = np.std(hrir_pos)

        # normalize inputs
        self.X_train = self.normalize(hrir_pos_train, self.hrirPos_mean, self.hrirPos_std)
        self.X_test = self.normalize(hrir_pos_test, self.hrirPos_mean, self.hrirPos_std)
        self.X_valid = self.normalize(hrir_pos_valid, self.hrirPos_mean, self.hrirPos_std)
        self.anthro_train = self.normalize(anthro_train, self.anthro_mean, self.anthro_std)
        self.anthro_test = self.normalize(anthro_test, self.anthro_mean, self.anthro_std)
        self.anthro_valid = self.normalize(anthro_valid, self.anthro_mean, self.anthro_std)

    # Method to normalize data
    def normalize(self, data, mean, std):
        torch_data = torch.FloatTensor(data)
        normalized_data = (torch_data - mean) / std
        return normalized_data
    
    # Method to train the model
    def trainModel(self, model):
        # Get training data
        X_train = self.X_train
        anthro_train = self.anthro_train
        X_valid = self.X_valid
        anthro_valid = self.anthro_valid
        X_valid = self.X_valid
        anthro_valid = self.anthro_valid
        # Set loss function
        criterion = nn.MSELoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 80
        train_losses = []
        val_losses = []
        indivLosses = []
        mse_train_data = []
        mse_validation_data = []
        min_valid_loss = np.inf
        for i in range(epochs):
            model.train()
            # propgate forward
            anthro_pred = model.forward(X_train)
            #calculate loss
            lossAnthro = criterion(anthro_pred, anthro_train)
            #calculate individual losses
            mse = nn.functional.mse_loss(anthro_pred, anthro_train, reduction='none')
            indivLosses.append(mse.detach().numpy())
            #Keep track of losses
            train_losses.append(lossAnthro.detach().numpy())
            train_output = torch.mean(mse, dim=0)
            mse_train_data.append(np.array(train_output.detach().numpy()))

            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {lossAnthro}')
            
            #Do some backward propagation
            model.train()
            optimizer.zero_grad()
            lossAnthro.backward()
            optimizer.step()

            # Get the MSE of the validation data without chaning the wieghts
            model.eval()   
            with torch.no_grad():
                anthro_val_pred = model(X_valid)
                anthro_val_pred = model(X_valid)

                loss_fn = nn.MSELoss()
                lossValAnthro = loss_fn(anthro_val_pred, anthro_valid)
                lossValAnthro = loss_fn(anthro_val_pred, anthro_valid)
                # plot validation error
                val_losses.append(lossValAnthro.detach().numpy())
                val_output = [0]*10
                for column_index in range(10):
                    val_output[column_index] = loss_fn(anthro_val_pred[:, column_index], anthro_valid[:, column_index])
                    val_output[column_index] = loss_fn(anthro_val_pred[:, column_index], anthro_valid[:, column_index])
                mse_validation_data.append(val_output)

                # do cross validation
                valid_loss = lossValAnthro.item() * X_valid.size(0)
                tot_loss_val = loss_fn(anthro_val_pred, anthro_valid)
                valid_loss = lossValAnthro.item() * X_valid.size(0)
                tot_loss_val = loss_fn(anthro_val_pred, anthro_valid)
                if min_valid_loss > tot_loss_val:
                    min_valid_loss = valid_loss
                    torch.save(model.state_dict(), 'saved_model.pth')

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