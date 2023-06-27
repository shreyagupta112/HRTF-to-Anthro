import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from inputProcessing import *
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __init__(self, left_ear_only):
        hrir = InputProcessing.extractHRIR(3)
        anthro = InputProcessing.extractAnthro(3, left_ear_only)
        anthro = np.tile(anthro, (1250,1))
        pos = InputProcessing.extractPos(3)
        hrir_train,  hrir_test,  anthro_train,  anthro_test,  pos_train,  pos_test = train_test_split(hrir, anthro, pos, test_size=0.2, random_state=41)

        # normalize inputs
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

        # get mean and standard deviation of inputs
        self.anthro_mean = torch.mean(anthro_train)
        self.anthro_std = torch.std(anthro_train)
        self.pos_mean = torch.mean(pos_train)
        self.pos_std = torch.std(pos_train)

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

            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {totalLoss}')
            
            #Do some backward propagation
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()
        # Plot losses
        trainLoss = plt.figure()
        plt.plot(range(epochs), losses)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        plt.ylim(-30, 15)
        trainLoss.savefig('../../figures/error.png')

    
    def basicValidation(self, model, left_ear_only):
        # Get data
        hrir_test = self.hrir_test
        anthro_test = self.anthro_test
        pos_test = self.pos_test
        pos_ape = []
        anthro_ape = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            anthro_eval, pos_eval = model.forward(hrir_test) # X-test are features from test se, y_eval s predictions
            lossAnthro = criterion(anthro_eval, anthro_test) 
            lossPos = criterion(pos_eval, pos_test) 
            totalLoss = lossAnthro + lossPos #find loss or error
            print(totalLoss)
            
            for i, data in enumerate(hrir_test):
                # find the percentage error in all anthropometric data outputs
                y_anthro, y_pos = model.forward(data)
                prediction_anthro = y_anthro.argmax().item()
                one_anthro = anthro_test[i]
                per_err_anthro = 0
                anthro_length = 27
                if left_ear_only == False:
                    anthro_length = 37
                for j in range(anthro_length):
                    per_err_anthro += abs((one_anthro[j] - prediction_anthro) /y_anthro[j])
                per_err_anthro = per_err_anthro/27
                anthro_ape.append(per_err_anthro)
                

                # find percentage error in all position outputs 
                prediction_pos = y_pos.argmax().item()
                one_pos = pos_test[i]
                per_err_pos = 0
                for j in range(3):
                    per_err_pos += abs((one_pos[j] - prediction_pos) /y_pos[j])
                per_err_pos = per_err_pos/3
                pos_ape.append(per_err_pos)

            # plot the average anthro error across each hrir
            anthroError = plt.figure()
            plt.plot(range(len(anthro_ape)), anthro_ape)
            plt.ylabel("Error")
            plt.xlabel("HRIR")
            plt.title("Error In Anthro Measurement Predictions")
            anthroError.savefig("../../figures/anthro_error.png")

            #plot the average position error across each hrir
            posError = plt.figure()
            plt.plot(range(len(pos_ape)), pos_ape)
            plt.ylabel("Error")
            plt.xlabel("HRIR")
            plt.title("Error in Position Measurment Predictions")
            posError.savefig("../../figures/pos_error.png")

            # average error across all test datasets
            pos_mape = sum(pos_ape)/len(pos_ape)
            anthro_mape = sum(anthro_ape)/len(anthro_ape)
        return float(pos_mape), float(anthro_mape)
    