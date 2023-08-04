import torch 
import torch.nn as nn
import torch.nn.functional as F


'''
This class represents the model architecture used for 
this simple model

One input vectors:
    HRTF of left ear + Src Position: 1 x 36
    
Output Vectors:
    Ear Anthro Measurements: 1 x 10

'''

class Model(nn.Module):

    def __init__(self, 
                 hrir_pos=36, 
                 h1=40, h2=50, h3=60, h4=70, h5=80, h6=65, h7=45, h8=25, h9=15,
                 ear_anthro=10):
        super().__init__() # instantiate our nn.Module
        #Connect model
        self.fc1 = nn.Linear(hrir_pos, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4) 
        self.fc5 = nn.Linear(h4, h5)
        self.fc6 = nn.Linear(h5, h6)
        self.fc7 = nn.Linear(h6, h7)
        self.fc8 = nn.Linear(h7, h8)
        self.fc9 = nn.Linear(h8, h9)
        self.fc_output = nn.Linear(h9, ear_anthro)
    
    # Function to propagate forward
    def forward(self, hrir_l):
        # Make activation functions tanh
        layer1 = F.tanh(self.fc1(hrir_l))
        layer2 = F.tanh(self.fc2(layer1))
        layer3 = F.tanh(self.fc3(layer2))
        layer4 = F.tanh(self.fc4(layer3))
        layer5 = F.tanh(self.fc5(layer4))
        layer6 = F.tanh(self.fc6(layer5))
        layer7 = F.tanh(self.fc7(layer6))
        layer8 = F.tanh(self.fc8(layer7))
        layer9 = F.tanh(self.fc9(layer8))

        Ear_Anthro = self.fc_output(layer9)

        return Ear_Anthro



