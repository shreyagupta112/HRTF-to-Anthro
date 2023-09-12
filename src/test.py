from model import *
import numpy as np
from inputProcessing import *

the_model = Model()
IP = InputProcessing()

input = IP.extractSingleHrirAndPos(3, "HRTF")
tensor_input = torch.tensor(input).to(torch.float32)
forward_pass = the_model.forward(tensor_input)
print(forward_pass)
print(np.shape(forward_pass))