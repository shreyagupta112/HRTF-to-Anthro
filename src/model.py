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

    def __init__(self, activationFunction="relu",
                 hrir_pos=36, 
                 h1=40, h2=50, h3=60, h4=70, h5=80, h6=65, h7=45, h8=25, h9=15,
                 ear_anthro=1):
        super().__init__() # instantiate our nn.Module
        self.actfunc = activationFunction
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
        self.TanH = nn.Tanh()
        self.fc_output = nn.Linear(h9, ear_anthro)
    
    # Function to propagate forward
    def forward(self, hrir_l):

        if self.actfunc == "tanh":
            # Make activation functions tanh
            layer1 = self.fc1(hrir_l)
            layer1 = self.TanH(layer1)
            layer2 = self.fc2(layer1)
            layer2 = self.TanH(layer2)
            layer3 = self.fc3(layer2)
            layer3 = self.TanH(layer3)
            layer4 = self.fc4(layer3)
            layer4 = self.TanH(layer4)
            layer5 = self.fc5(layer4)
            layer5 = self.TanH(layer5)
            layer6 = self.fc6(layer5)
            layer6 = self.TanH(layer6)
            layer7 = self.fc7(layer6)
            layer7 = self.TanH(layer7)
            layer8 = self.fc8(layer7)
            layer8 = self.TanH(layer8)
            layer9 = self.fc9(layer8)
            layer9 = self.TanH(layer9)

            Ear_Anthro = self.fc_output(layer9)
            Ear_Anthro = self.TanH(Ear_Anthro)

            return Ear_Anthro

        else:
            # Make activation functions tanh
            layer1 = F.relu(self.fc1(hrir_l))
            layer2 = F.relu(self.fc2(layer1))
            layer3 = F.relu(self.fc3(layer2))
            layer4 = F.relu(self.fc4(layer3))
            layer5 = F.relu(self.fc5(layer4))
            layer6 = F.relu(self.fc6(layer5))
            layer7 = F.relu(self.fc7(layer6))
            layer8 = F.relu(self.fc8(layer7))
            layer9 = F.relu(self.fc9(layer8))

            Ear_Anthro = self.fc_output(layer9)

            return Ear_Anthro  


