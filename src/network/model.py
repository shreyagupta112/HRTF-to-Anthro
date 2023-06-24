import torch 
import torch.nn as nn
import torch.nn.functional as F


'''
This class represents the model architecture used for 
this simple model

Two input vectors:
    HRIR of left ear: 1 X 200
    Position vector: 1 X 3

Output Vectors:
    X (head and torso): 17 x 1
    D (ear measurements): 8 x 1
    Theta (pinna angles): 2 x 1

'''

class Model(nn.Module):

    def __init__(self, 
                 hrir=200, pos=3, 
                 h1=10, h2=15, h3=10, 
                 x=17, d=8, theta=2):
        super().__init__() # instantiate our nn.Module
        #Connect model
        self.fc1 = nn.Linear(hrir+pos, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_output1 = nn.Linear(h3, x)
        self.fc_output2 = nn.Linear(h3, d)
        self.fc_output3 = nn.Linear(h3, theta)
    
    # Function to propagate forward
    def forward(self, hrir_l, pos):
        combinedInputs = torch.cat((hrir_l, pos), dim=1)

        layer1 = F.relu(self.fc1(combinedInputs))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))

        x = self.fc_output1(layer3)
        d = self.fc_output2(layer3)
        theta = self.fc_output3(layer3)

        return x, d, theta



