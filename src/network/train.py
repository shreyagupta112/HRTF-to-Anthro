import torch
import torch.nn as nn

'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:

    # Method to train the model
    def trainModel(self, epochs, model, optimizer, criterion, hrir_train, pos_train, anthro_train):
        
        x_train, d_train, theta_train = self.deconstructAnthro(anthro_train)
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
    