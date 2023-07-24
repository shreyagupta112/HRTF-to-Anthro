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
        hrir_pos_train,  hrir_pos_test, anthro_train, anthro_test = train_test_split(hrir_pos, anthro, test_size=0.1, random_state=41)

        # get mean and standard deviation
        self.anthro_mean = np.mean(anthro)
        self.anthro_std = np.std(anthro)
        self.hrirPos_mean = np.mean(hrir_pos)
        self.hrirPos_std = np.std(hrir_pos)

        # normalize inputs
        hrir_pos_train = torch.FloatTensor(hrir_pos_train)
        self.X_train = (hrir_pos_train - self.hrirPos_mean) / (self.hrirPos_std)
        hrir_pos_test = torch.FloatTensor(hrir_pos_test)
        self.X_test  = (hrir_pos_test - self.hrirPos_mean) / (self.hrirPos_std)
        hrir_pos_valid = torch.FloatTensor(hrir_pos_valid)
        self.X_valid  = (hrir_pos_valid - self.hrirPos_mean) / (self.hrirPos_std)
        anthro_train = torch.FloatTensor(anthro_train)
        self.anthro_train = (anthro_train - self.anthro_mean) / (self.anthro_std)
        anthro_test = torch.FloatTensor(anthro_test)
        self.anthro_test = (anthro_test - self.anthro_mean) / (self.anthro_std)
        anthro_valid = torch.FloatTensor(anthro_valid)
        self.anthro_valid = (anthro_valid - self.anthro_mean) / (self.anthro_std)

    # Method to train the model
    def trainModel(self, model):
        # Get training data
        X_train = self.X_train
        anthro_train = self.anthro_train
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

                loss_fn = nn.MSELoss()
                lossValAnthro = loss_fn(anthro_val_pred, anthro_valid)
                # plot validation error
                val_losses.append(lossValAnthro.detach().numpy())
                val_output = [0]*10
                for column_index in range(10):
                    val_output[column_index] = loss_fn(anthro_val_pred[:, column_index], anthro_valid[:, column_index])
                mse_validation_data.append(val_output)

                # do cross validation
                valid_loss = lossValAnthro.item() * X_valid.size(0)
                tot_loss_val = loss_fn(anthro_val_pred, anthro_valid)
                if min_valid_loss > tot_loss_val:
                    min_valid_loss = valid_loss
                    torch.save(model.state_dict(), 'saved_model.pth')

    def testModel(self, model):
        X_test = self.X_test
        anthro_test = self.anthro_test
        test_losses =[]
        mse_test_data = []

        anthro_test_pred = model(X_test)
        loss_fn = nn.MSELoss()
        lossValAnthro = loss_fn(anthro_test_pred, anthro_test)
        # plot validation error
        test_losses.append(lossValAnthro.detach().numpy())
        val_output = [0]*10
        for column_index in range(10):
            val_output[column_index] = loss_fn(anthro_test_pred[:, column_index], anthro_test[:, column_index])
        mse_test_data.append(val_output)

        '''

        # Plot losses
        trainLoss = plt.figure()
        plt.plot(range(epochs), train_losses, label = "training losses")
        plt.plot(range(epochs), val_losses, label = "validation losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        trainLoss.savefig('../figures/error.png')

        '''
     