import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from inputProcessing import *
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __init__(self, ):
        hrir = InputProcessing.extractHRIR(3)
        anthro = InputProcessing.extractAnthro(3)
        anthro = np.tile(anthro, (1250,1))
        pos = InputProcessing.extractPos(3)
        hrir_train,  hrir_test,  anthro_train,  anthro_test,  pos_train,  pos_test = train_test_split(hrir, anthro, pos, test_size=0.2, random_state=41)
        hrir_train = torch.FloatTensor(hrir_train)
        self.hrir_train = torch.nn.functional.normalize(hrir_train, p=2.0, dim = 1)

        hrir_test = torch.FloatTensor(hrir_test)
        self.hrir_test  = torch.nn.functional.normalize(hrir_test, p=2.0, dim = 1)

        anthro_train = torch.FloatTensor(anthro_train)
        self.anthro_train = torch.nn.functional.normalize(anthro_train, p=2.0, dim = 1)

        anthro_test = torch.FloatTensor(anthro_test)
        self.anthro_test = torch.nn.functional.normalize(anthro_test, p=2.0, dim = 1)

        pos_train = torch.FloatTensor(pos_train)
        self.pos_train = torch.nn.functional.normalize(pos_train, p=2.0, dim = 1)

        pos_test = torch.FloatTensor(pos_test)
        self.pos_test = torch.nn.functional.normalize(pos_test, p=2.0, dim = 1)

    # Method to train the model
    def trainModel(self, model):
        # Get training data
        hrir_train = self.hrir_train
        anthro_train = self.anthro_train
        pos_train = self.pos_train
        # Set loss function
        criterion = nn.CrossEntropyLoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 300
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

    
    def basicValidation(self, model):
        # Get data
        hrir_test = self.hrir_test
        anthro_test = self.anthro_test
        pos_test = self.pos_test
        ape = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            anthro_eval, pos_eval = model.forward(hrir_test) # X-test are features from test se, y_eval s predictions
            lossAnthro = criterion(anthro_eval, anthro_test) 
            lossPos = criterion(pos_eval, pos_test) 
            totalLoss = lossAnthro + lossPos #find loss or error
            print(totalLoss)
            
            for i, data in enumerate(hrir_test):
                y_anthro, y_pos = model.forward(data)
                #prediction_anthro = y_anthro.argmax().item()
                #per_err_anthro = (anthro_test[i] - prediction_anthro) /y_anthro[i]
                prediction_pos = y_pos.argmax().item()
                one_pos = pos_test[i]
                per_err_pos = 0
                for j in range(3):
                    per_err_pos += abs((one_pos[j] - prediction_pos) /y_pos[j])
                per_err_pos = per_err_pos/3
                ape.append(per_err_pos)
            mape = sum(ape)/len(ape)
        return float(mape)
    