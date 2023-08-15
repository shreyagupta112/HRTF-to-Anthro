import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from inputProcessing import *
import numpy as np
from model import *
from dataProcessing import *
'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:
    def __init__(self, splitType, dataType, activationFunction):
        self.DP = DataProcessing()
        self.validSubjects = self.DP.validSubjects
        self.splitType = splitType
        self.dataType = dataType
        self.activFunc = activationFunction
        
    # Method to train the model
    def trainModel(self, model):
        hrir_train, hrir_valid, hrir_test, anthro_train, anthro_valid, anthro_test = self.DP.dataSplitTypeOne(self.dataType)
        self.trainSubjects, self.validationSubjects, self.testSubjects = self.DP.readSplits()
        
        self.X_train = hrir_train
        self.X_valid = hrir_valid
        self.X_test = hrir_test
        self.Y_train = anthro_train
        self.Y_valid = anthro_valid
        self.Y_test = anthro_test

        print(self.trainSubjects)
        print(self.validationSubjects)
        print(self.testSubjects)

        # Get training data
        X_train = self.X_train
        anthro_train = self.Y_train
        X_valid = self.X_valid
        anthro_valid = self.Y_valid
        X_valid = self.X_valid
        anthro_valid = self.Y_valid
        X_test = self.X_test
        anthro_test = self.Y_test
        # Set loss function
        criterion = nn.MSELoss()
        #Choose Adam Optimizer, learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #Set iterations
        epochs = 80
        train_losses = []
        val_losses = []
        test_losses = []
        mse_train_data = []
        mse_validation_data = []
        mse_test_data = []
        min_valid_loss = np.inf
        for i in range(epochs):
            model.train()
            # propgate forward
            anthro_pred = model.forward(X_train)
            #calculate loss
            print(len(anthro_pred), len(anthro_train))
            lossAnthroTrain = criterion(anthro_pred, anthro_train)
            #calculate individual losses
            indiv_mse = torch.mean((anthro_pred - anthro_train)**2, dim=0)
            #Keep track of losses
            train_losses.append(lossAnthroTrain.detach().numpy())
            mse_train_data.append(np.array(indiv_mse.detach().numpy()))
            
            #Do some backward propagation
            optimizer.zero_grad()
            lossAnthroTrain.backward()
            optimizer.step()
            train_loss = lossAnthroTrain.item()

            # Get the MSE of the validation data without chaning the weights (Cross Validation)
            model.eval()   
            with torch.no_grad():
                # Find loss for validation
                anthro_val_pred = model(X_valid)
                lossValAnthro = criterion(anthro_val_pred, anthro_valid)
            
                val_losses.append(lossValAnthro.detach().numpy())
                
                # get individual MSE's for validation
                val_output = torch.mean((anthro_val_pred - anthro_valid)**2, dim=0) 
                mse_validation_data.append(np.array(val_output.detach().numpy()))

                valid_loss = lossValAnthro.item()

                print(f'Epoch {i+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')

                # Find loss on test set
                anthro_test_pred = model(X_test)
                lossTestAnthro = criterion(anthro_test_pred, anthro_test)

                test_losses.append(lossTestAnthro.detach().numpy())

                # get individual MSE's for testing
                test_output = torch.mean((anthro_test_pred - anthro_test)**2, dim=0)
                mse_test_data.append(np.array(test_output.detach().numpy()))

            # Update validation loss
            if min_valid_loss > lossValAnthro:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}-->{lossValAnthro:.6f}) \t Saving The Model')
                min_valid_loss = lossValAnthro
                # save current state of model
                if self.activFunc == "tanh":
                    torch.save(model.state_dict(), 'saved_model_tanh.pth')
                else:
                    torch.save(model.state_dict(), 'saved_model.pth') 

        # Plot total error per epoch
        trainLoss = plt.figure()
        plt.plot(range(epochs), train_losses, label = "training losses")
        plt.plot(range(epochs), val_losses, label = "validation losses")
        plt.plot(range(epochs), test_losses, label = "test losses")
        plt.legend(loc="upper right")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        if not os.path.exists(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}'):
            os.makedirs(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}')
        trainLoss.savefig(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/error.png')
        plt.close()

        # Plot error for each anthro measurement
        for i in range(10):
            anthroMSE = plt.figure()
            mse_train_data = np.array(mse_train_data)
            mse_validation_data = np.array(mse_validation_data)
            mse_test_data = np.array(mse_test_data)
            mse_train = mse_train_data[:, i]
            mse_valid = mse_validation_data[:, i]
            mse_test = mse_test_data[:, i]

            plt.plot(range(epochs), mse_train, label = "training data")
            plt.plot(range(epochs), mse_valid, label = "validation data")
            plt.plot(range(epochs), mse_test, label = "test data")
            plt.legend(loc="upper right")

            ylabel = "MSE of Anthro Measure " + str(i)
            plotlabel = ylabel + " vs Epoch"
            if not os.path.exists(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/indivAnthro'):
                os.makedirs(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/indivAnthro')
            figlabel = f"../figures/{self.activFunc}/{self.dataType}/{self.splitType}/indivAnthro/" + str(i) + ".png"

            plt.ylabel(ylabel)
            plt.xlabel("Epoch")
            plt.title(plotlabel)
            anthroMSE.savefig(figlabel)
            plt.close()

     # Method to test the model
    def testModel(self, modelPath):
        # Load model from saved Path

        train, validation, test = self.DP.readSplits()

        X_test, anthro_test = InputProcessing().extractData(test, self.dataType)
        X_test = torch.tensor(X_test).to(torch.float32)
        anthro_test = torch.tensor(anthro_test).to(torch.float32)

        X_validation, anthro_validation = InputProcessing().extractData(validation, self.dataType)
        X_validation = torch.tensor(X_validation).to(torch.float32)
        anthro_validation = torch.tensor(anthro_validation).to(torch.float32) 

        X_train, anthro_train = InputProcessing().extractData(train, self.dataType)
        X_train = torch.tensor(X_train).to(torch.float32)
        anthro_train = torch.tensor(anthro_train).to(torch.float32)
        
        model = Model(self.activFunc)
        if self.dataType == "trunc64":
            model = Model(self.activFunc, 67)
        if self.dataType == "raw":
            model = Model(self.activFunc, 203)

        model.load_state_dict(torch.load(modelPath))

        criterion = nn.MSELoss()
        
        lossTest = self.createTestPlot(model, criterion, "test", test, X_test, anthro_test)

        validationTest = self.createTestPlot(model, criterion, "validation", validation, X_validation, anthro_validation) 

        trainTest = self.createTestPlot(model, criterion, "train", train, X_train, anthro_train)

        with torch.no_grad():
            # calculate MSE for whole predicition vector
            anthro_eval = model.forward(X_test) # X-test are features from test se, y_eval s predictions

            # plot predicted vs actual for each anthropometric data point
            for i in range(len(anthro_eval[0])):
                prediction = plt.figure()
                anthro_eval_at_i = []
                anthro_test_at_i = []
                for j in range(len(anthro_eval)):
                    anthro_eval_at_i.append(anthro_eval[j][i])
                    anthro_test_at_i.append(anthro_test[j][i])
                plt.plot(range(len(anthro_eval_at_i)), anthro_eval_at_i, label = "prediction")
                plt.plot(range(len(anthro_test_at_i)), anthro_test_at_i, label = "actual")
                plt.legend(loc="upper right")
                plt.ylabel("Measurement")
                plt.xlabel("HRIR")
                plt.title(f"Anthro Prediction for measurement{i}")
                prediction.savefig(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/{i}_pred.png')
                plt.close()
        
        with torch.no_grad():
            # calculate MSE for whole predicition vector
            anthro_eval = model.forward(X_train) # X-test are features from test se, y_eval s predictions

            # plot predicted vs actual for each anthropometric data point
            for i in range(len(anthro_eval[0])):
                prediction = plt.figure()
                anthro_eval_at_i = []
                anthro_train_at_i = []
                for j in range(len(anthro_eval)):
                    anthro_eval_at_i.append(anthro_eval[j][i])
                    anthro_train_at_i.append(anthro_train[j][i])
                plt.plot(range(len(anthro_eval_at_i)), anthro_eval_at_i, label = "prediction")
                plt.plot(range(len(anthro_train_at_i)), anthro_train_at_i, label = "actual")
                plt.legend(loc="upper right")
                plt.ylabel("Measurement")
                plt.xlabel("HRIR")
                plt.title(f"Anthro Prediction for measurement{i}")
                prediction.savefig(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/{i}_train_pred.png')
                plt.close()
        
        with torch.no_grad():
            # calculate MSE for whole predicition vector
            anthro_eval = model.forward(X_validation) # X-test are features from test se, y_eval s predictions

            # plot predicted vs actual for each anthropometric data point
            for i in range(len(anthro_eval[0])):
                prediction = plt.figure()
                anthro_eval_at_i = []
                anthro_validation_at_i = []
                for j in range(len(anthro_eval)):
                    anthro_eval_at_i.append(anthro_eval[j][i])
                    anthro_validation_at_i.append(anthro_validation[j][i])
                plt.plot(range(len(anthro_eval_at_i)), anthro_eval_at_i, label = "prediction")
                plt.plot(range(len(anthro_validation_at_i)), anthro_validation_at_i, label = "actual")
                plt.legend(loc="upper right")
                plt.ylabel("Measurement")
                plt.xlabel("HRIR")
                plt.title(f"Anthro Prediction for measurement{i}")
                prediction.savefig(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/{i}_valid_pred.png')
                plt.close()

        return lossTest, validationTest, trainTest
    
    def createTestPlot(self, model, lossFn, split, splitList, hrtf, anthro):
        with torch.no_grad():
            # calculate MSE for whole predicition vector
            anthro_eval = model.forward(hrtf) # X-test are features from test se, y_eval s predictions
            lossAnthro = lossFn(anthro_eval, anthro) 

            # plot predicted vs actual for each anthropometric data point
            for i in range(len(anthro_eval[0])):
                anthro_eval_at_i = []
                anthro_test_at_i = []
                for j in range(len(anthro_eval)):
                    anthro_eval_at_i.append(anthro_eval[j][i])
                    anthro_test_at_i.append(anthro[j][i])
                ind = 0
                for subject in splitList:
                    start = 2500*ind
                    end = 2500*ind + 2500
                    prediction = plt.figure()
                    plt.plot(range(len(anthro_eval_at_i[start:end])), anthro_eval_at_i[start:end], label = "prediction")
                    plt.plot(range(len(anthro_test_at_i[start:end])), anthro_test_at_i[start:end], label = "actual")
                    plt.legend(loc="upper right")
                    plt.ylabel("Measurement")
                    plt.xlabel("HRIR")
                    plt.title(f"Anthro Prediction for subject {subject} measurement{i}")

                    # Save each subject's graph with a unique filename
                    if not os.path.exists(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/{split}'):
                        os.makedirs(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/{split}')
                    prediction.savefig(f'../figures/{self.activFunc}/{self.dataType}/{self.splitType}/test/{split}/subject_{subject}_pos{i}_pred.png')

                    # Close the current figure to start a new one for the next subject
                    plt.close()
                    ind += 1

        return lossAnthro
     
    def plotHRIR(subject, position):
        hrir = InputProcessing().extractSingleHRIR(subject, "HRIR")
        hrir_plot = plt.figure()
        print(len(hrir), len(hrir[0]))
        plt.plot(range(len(hrir[:1250][position])), hrir[:1250][position], label = "left hrir")
        plt.plot(range(len(hrir[1250:][position])), hrir[1250:][position], label = "right hrir")
        plt.legend(loc="upper right")
        plt.ylabel("HRIR")
        plt.xlabel("Time")
        plt.title(f"Center HRIR Plot for subject {subject} at position {position}")
        plt.show()
        plt.close()
    def plotHRTF(self, subject, position):
        hrtf = InputProcessing().extractSingleHRIR(subject, "HRTF", True)
        hrtf_plot = plt.figure()
        print(len(hrtf), len(hrtf[0]))
        plt.plot(range(len(hrtf[position])), hrtf[position], label = "norm left hrtf")
        # plt.plot(range(len(hrtf[1250:][position])), hrtf[1250:][position], label = "right hrir")
        plt.legend(loc="upper right")
        plt.ylabel("HRTF")
        plt.xlabel("Frequency")
        plt.title(f"Normalized HRTF Plot for subject {subject} at position {position}")
        plt.show()
        plt.close()

MT = ModelTrainer("split1", "HRTF", "tanh")
MT.plotHRTF(3, 0)
