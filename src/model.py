import torch 
import torch.nn as nn
import torch.nn.functional as F


'''
This class represents the model architecture used for 
this simple model

One input vectors:
    HRIR of left ear: 1 X 200

Output Vectors:
    Left-Ear-Only Anthro Measurements: 27 x 1 
    Position vector: 1 X 3

'''

class Model(nn.Module):

    def __init__(self, 
                 hrir=64, 
                 h1=10, h2=15, h3=10, 
                 anthro=27, pos=3):
        super().__init__() # instantiate our nn.Module
        #Connect model
        self.fc1 = nn.Linear(hrir, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_output1 = nn.Linear(h3, anthro)
        self.fc_output2 = nn.Linear(h3, pos)
    
    # Function to propagate forward
    def forward(self, hrir_l):

        layer1 = F.relu(self.fc1(hrir_l))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))

        Anthro = self.fc_output1(layer3)
        Pos = self.fc_output2(layer3)

        return Anthro, Pos



