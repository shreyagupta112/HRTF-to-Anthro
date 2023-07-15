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
        hrir_pos_train,  hrir_pos_test, anthro_train, anthro_test = train_test_split(hrir_pos, anthro, test_size=0.2, random_state=41)

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
        anthro_train = torch.FloatTensor(anthro_train)
        self.anthro_train = (anthro_train - self.anthro_mean) / (self.anthro_std)
        anthro_test = torch.FloatTensor(anthro_test)
        self.anthro_test = (anthro_test - self.anthro_mean) / (self.anthro_std)

    # Method to train the model
    def trainModel(self, model):
        # Get training data
        X_train = self.X_train
        anthro_train = self.anthro_train
        # Set loss function
        criterion = nn.MSELoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 80
        losses = []
        indivLosses = []

        mse_individual = [[0 for a in range(epochs)] for b in range(10)]
        for i in range(epochs):
            # propgate forward
            anthro_pred = model.forward(X_train)

            #calculate loss
            lossAnthro = criterion(anthro_pred, anthro_train)

            #calculate individual losses
            mse = nn.functional.mse_loss(anthro_pred, anthro_train, reduction='none')
            indivLosses.append(mse.detach().numpy())

            #Keep track of losses
            losses.append(lossAnthro.detach().numpy())

            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {lossAnthro}')

            # Calculate mean squared error
            for j, data in enumerate(X_train):
                y_anthro = model.forward(data)
                one_anthro = anthro_train[j]
                for k in range(10):
                    mse_individual[k][i // 10] += ((one_anthro[k].item() - y_anthro[k].item())**2)*(1/10)
                
            #Do some backward propagation
            optimizer.zero_grad()
            lossAnthro.backward()
            optimizer.step()

        # Plot losses
        '''
        trainLoss = plt.figure()
        plt.plot(range(epochs // 10), losses)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        trainLoss.savefig('../figures/error.png')
        '''

        # Plot mse
        for i in range(10):
            anthroMSE = plt.figure()
            mse_anthro = mse_individual[i]
            plt.plot(range(epochs), mse_anthro)
            ylabel = "Anthro Measure " + str(i)
            plt.ylabel(ylabel)
            plt.xlabel("Epoch")
            plotlabel = ylabel + " vs Epoch"
            plt.title(plotlabel)
            figlabel = "../figures/" + str(i) + ".png"
            anthroMSE.savefig(figlabel)
    
    def basicValidation(self, model):
        # Get data
        X_test = self.X_test
        anthro_test = self.anthro_test
        criterion = nn.MSELoss()
        with torch.no_grad():
            anthro_eval = model.forward(X_test) # X-test are features from test se, y_eval s predictions
            lossAnthro = criterion(anthro_eval, anthro_test) 
            totalLoss = lossAnthro #find loss or error
            mse = [0]* 10
            print(totalLoss)
            
            # Calculate mean squared error
            for i, data in enumerate(X_test):
                # find the percentage error in all anthropometric data outputs
                y_anthro = model.forward(data)
                one_anthro = anthro_test[i]
                for j in range(10):
                    mse[j] += (one_anthro[j] - y_anthro[j])**2
            for i in range(10):
                mse[j] *= (1/10)
            # plot the mse for each anthropometric data point
        return mse
    