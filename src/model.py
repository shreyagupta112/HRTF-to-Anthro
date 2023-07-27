import torch 
import torch.nn as nn
import torch.nn.functional as F


'''
This class represents the model architecture used for 
this simple model

One input vectors:
    HRIR of left ear + Src Position: 1 x 64
    
Output Vectors:
    Ear Anthro Measurements: 1 x 10

'''

class Model(nn.Module):

    def __init__(self, 
                 hrir_pos=203, 
                 h1=30, h2=45, h3=30, 
                 ear_anthro=10):
        super().__init__() # instantiate our nn.Module
        #Connect model
        self.fc1 = nn.Linear(hrir_pos, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_output = nn.Linear(h3, ear_anthro)
    
    # Function to propagate forward
    def forward(self, hrir_l):

        layer1 = F.relu(self.fc1(hrir_l))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))

        Ear_Anthro = self.fc_output(layer3)

        return Ear_Anthro



