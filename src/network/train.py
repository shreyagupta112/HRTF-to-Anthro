import torch
import torch.nn as nn

'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:

    # Method to train the model
    def trainModel(model, hrir_train, anthro_train, pos_train):
        # Set loss function
        criterion = nn.CrossEntropyLoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 100
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

    