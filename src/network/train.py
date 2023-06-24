import torch
import torch.nn as nn

'''
This class contains methods relevant towards training
a model
'''
class ModelTrainer:

    # Method to train the model
    def trainModel(epochs, model, optimizer, criterion, hrir_train, pos_train, anthro_train, anthro_pred):
        
        for i in range(epochs):
            # propgate forward
            print(i)
    
    #Method to deconstruct anthro vector
    def deconstructAnthro(anthroVector):
        print(anthroVector)