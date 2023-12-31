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
class SubNetwork(nn.Module):
    
    def __init__(self, input=10, h2=20, h3=30, h4=20, h5=10, output=1):
        super(SubNetwork, self).__init__() # instantiate our nn.Module

        self.fc1 = nn.Linear(input, h2)
        self.fc2 = nn.Linear(h2, h3)
        self.fc3 = nn.Linear(h3, h4)
        self.fc4 = nn.Linear(h4, h5)
        self.output = nn.Linear(h5, output)

    def forward(self, input):

        layer1 = F.relu(self.fc1(input))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))
        layer4 = F.relu(self.fc4(layer3))

        output = F.relu(self.output(layer4))

        return output
class Model(nn.Module):

    def __init__(self, activationFunction="relu",
                 hrir_pos=36, 
                 h1=50, h2=70, h3=90, h4=110, h5=130, h6=120, h7=100, h8=10, h9=20, h10=30, h11=20, h12=10, 
                 ear_anthro=1):
        super(Model, self).__init__() # instantiate our nn.Module
        self.actfunc = activationFunction
        self.subnetwork1 = SubNetwork()
        self.subnetwork2 = SubNetwork()
        self.subnetwork3 = SubNetwork()
        self.subnetwork4 = SubNetwork()
        self.subnetwork5 = SubNetwork()
        self.subnetwork6 = SubNetwork()
        self.subnetwork7 = SubNetwork()
        self.subnetwork8 = SubNetwork()
        self.subnetwork9 = SubNetwork()
        self.subnetwork10 = SubNetwork()
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
            print(np.shape(initial_output))
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

            # anthro1 = self.subnetwork1.forward(initial_output[:, 0:10])
            # output_tensors.append(anthro1)
            # anthro2 = self.subnetwork1.forward(initial_output[:, 10:20])
            # output_tensors.append(anthro2)
            # anthro3 = self.subnetwork1.forward(initial_output[:, 20:30])
            # output_tensors.append(anthro3)
            # anthro4 = self.subnetwork4.forward(initial_output[:, 30:40])
            # output_tensors.append(anthro4)
            # anthro5 = self.subnetwork1.forward(initial_output[:, 40:50])
            # output_tensors.append(anthro5)
            # anthro6 = self.subnetwork1.forward(initial_output[:, 50:60])
            # output_tensors.append(anthro6)
            # anthro7 = self.subnetwork1.forward(initial_output[:, 60:70])
            # output_tensors.append(anthro7)
            # anthro8 = self.subnetwork1.forward(initial_output[:, 70:80])
            # output_tensors.append(anthro8)
            # anthro9 = self.subnetwork1.forward(initial_output[:, 80:90])
            # output_tensors.append(anthro9)
            # anthro10 = self.subnetwork1.forward(initial_output[:, 90:100])
            # output_tensors.append(anthro10)
            
            # numpy_values = np.array(numpy_values)
            # print(np.shape(numpy_values))
            
            output = torch.cat(output_tensors, dim=1)

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


