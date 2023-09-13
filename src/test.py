from model import *
import numpy as np
from inputProcessing import *

the_model = Model()
IP = InputProcessing()

input, anthro = IP.extractData([3, 10, 18, 20, 21], "HRTF")
tensor_input = torch.tensor(input).to(torch.float32)
forward_pass = the_model.forward(tensor_input)
print(forward_pass)
print(np.shape(forward_pass))