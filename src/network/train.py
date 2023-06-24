import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:

    # Method to train the model
    def trainModel(self, epochs, model, optimizer, criterion, hrir_train, pos_train, anthro_train):
        
        X = self.deconstructAnthro(anthro_train)
        y =  hrir_train
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
        losses = []
        for i in range(epochs):
            # propgate forward
            x_pred, d_pred, theta_pred = model.forward(hrir_train, pos_train)

            #calculate loss
            lossX = criterion(x_pred, x_train) 
            lossD = criterion(d_pred, d_train) 
            lossTheta = criterion(theta_pred, theta_train)
            totalLoss = lossX + lossD + lossTheta

            #Keep track of losses
            losses.append(totalLoss.detach().numpy())

            #print every 10 epochs
            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {totalLoss}')
            
            #Do some backward propagation
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()
    
    #method to deconstruct anthro vector
    def deconstructAnthro(anthroVector):
        X = anthroVector[8:25]
        D = anthroVector[0:8]
        Theta = anthroVector[25:27]
        return X, D, Theta
    