import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

class Validation:
    def basicValidation(x_test, y_test):
        ape = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            y_eval = Model.forward(x_test) # X-test are features from test se, y_eval s predictions
            loss = criterion(y_eval,y_test) #find losee or error
            print(loss)
            
            for i, data in enumerate(x_test):
                y_val = Model.forward(data)
                prediction = y_val.argmax().item()
                per_err = (y_test[i] - prediction) /y_test[i]
                ape.append(abs(per_err))
            mape = sum(ape)/len(ape)
        return mape