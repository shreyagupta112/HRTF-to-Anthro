import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def getInput():
        Y = anthro_train 
        x_train, x_test, y_train, y_test = train_test_split(hrir_train, y, test_size=0.2, random_state=41)

    # Method to train the model
    def trainModel(self, epochs, model, optimizer, criterion, hrir_train, anthro_train, pos_train):
        losses = []
        for i in range(epochs):
            # propgate forward
            anthro_pred, pos_pred = model.forward(hrir_train)

            #calculate loss
            lossAnthro = criterion(anthro_pred, anthro_train) 
            lossPos = criterion(pos_pred, pos_train) 
            totalLoss = lossAnthro + lossPos

            #Keep track of losses
            losses.append(totalLoss.detach().numpy())

            #print every 10 epochs
            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {totalLoss}')
            
            #Do some backward propagation
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

    