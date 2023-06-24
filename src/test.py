import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Model Class that inherits nn.Module
class Model(nn.Module):

    #Input layer(4 features (for now)) --> Hidden Layer 1 (Number of neurons) --> H2 (# of neurons) --> output
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # instantiate our nn.Module
        # fully connected
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # Function which moves data through network
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

#Pick a manual seed for randomization
torch.manual_seed(41)

model = Model()

url = ''

my_df = pd.read_csv(url)

# Some data processing stuff happens...

X = my_df.drop('variety', axis=1)
y = my_df['variety']

X = X.values
y = y.values

#Create test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=41)

#Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Convert y labels to tensor long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set the criterion of model to measure the error, how far off predictions are from the data
criterion = nn.CrossEntropyLoss()
#Choose Adam Optimizer, learning rate, 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train our model!
#Epochs? (One run thru all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train) # Get predicted results

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train) # predicted values vs the y_train

    #Keep Track of our losses
    losses.append(loss.detach().numpy())

    #print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')
    
    #Do some back propagation: take the error rate of forward propagation and feed it back
    # thru the network to find tune the weights

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()





