import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from inputProcessing import *
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __innit__(self, ):
        hrir = InputProcessing.extractHRIR(3)
        anthro = InputProcessing.extractAnthro(3)
        np.tile(anthro, (1250,1))
        pos = InputProcessing.extractPos(3)
        self.hrir_train,  self.hrir_test,  self.anthro_train,  self.anthro_test,  self.pos_train,  self.pos_test = train_test_split(hrir, anthro, pos, test_size=0.2, random_state=41)


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

    
    def basicValidation(model, x_test, y_test):
        ape = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            y_eval = model.forward(x_test) # X-test are features from test se, y_eval s predictions
            loss = criterion(y_eval,y_test) #find losee or error
            print(loss)
            
            for i, data in enumerate(x_test):
                y_val = model.forward(data)
                prediction = y_val.argmax().item()
                per_err = (y_test[i] - prediction) /y_test[i]
                ape.append(abs(per_err))
            mape = sum(ape)/len(ape)
        return mape
    