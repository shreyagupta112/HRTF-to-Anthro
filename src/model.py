import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
                 h1=50, h2=70, h3=90, h4=110, h5=130, h6=120, h7=100, h8=10, h9=20, h10=30, h11=20, h12=10, 
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
        self.fc8 = nn.Linear(h8, h8)
        self.fc9 = nn.Linear(h8, h9)
        self.fc10 = nn.Linear(h9, h10)
        self.fc11 = nn.Linear(h10, h11)
        self.fc12 = nn.Linear(h11, h12)
        self.fc_output = nn.Linear(h12, ear_anthro)
        self.TanH = nn.Tanh()
        # self.fc1 = nn.Linear(hrir_pos, h1)
        # self.fc2 = nn.Linear(h1, h2)
        # self.fc3 = nn.Linear(h2, h3)
        # self.fc4 = nn.Linear(h3, h4) 
        # self.fc5 = nn.Linear(h4, h5)
        # self.fc6 = nn.Linear(h5, h6)
        # self.fc7 = nn.Linear(h6, h7)
        # self.fc8 = nn.Linear(h7, h8)
        # self.fc9 = nn.Linear(h8, h9)
        # self.TanH = nn.Tanh()
        # self.fc_output = nn.Linear(h9, ear_anthro)
    
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
            initial_output = F.relu(self.fc7(layer6))

            # Subnetwork's
            output_tensors = []
            numpy_values = []
            for i in range(10, 101, 10):
                sub_layer1 = F.relu(self.fc8(initial_output[:, i-10:i]))
                sub_layer2 = F.relu(self.fc9(sub_layer1))
                sub_layer3 = F.relu(self.fc10(sub_layer2))
                sub_layer4 = F.relu(self.fc11(sub_layer3))
                sub_layer5 = F.relu(self.fc12(sub_layer4))
                i_output = F.relu(self.fc_output(sub_layer5))
                output_tensors.append(i_output)
                numpy_values.append(i_output.detach().numpy())
            
            numpy_values = np.array(numpy_values)
            print(np.shape(numpy_values))
            
            output = torch.cat(output_tensors, dim=0)

            # layer1 = F.relu(self.fc1(hrir_l))
            # layer2 = F.relu(self.fc2(layer1))
            # layer3 = F.relu(self.fc3(layer2))
            # layer4 = F.relu(self.fc4(layer3))
            # layer5 = F.relu(self.fc5(layer4))
            # layer6 = F.relu(self.fc6(layer5))
            # layer7 = F.relu(self.fc7(layer6))
            # layer8 = F.relu(self.fc8(layer7))
            # layer9 = F.relu(self.fc9(layer8))

            # Ear_Anthro = self.fc_output(layer9)

            # return Ear_Anthro  

            return output


