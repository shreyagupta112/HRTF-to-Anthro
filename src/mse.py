import torch
import torch.nn as nn

# Sample data
y_true = torch.tensor([3.0, 5.0, 7.0])
y_pred = torch.tensor([2.5, 4.8, 7.2])
true = torch.tensor([3.0])
pred = torch.tensor([5.0])
# Define the Mean Squared Error Loss function
loss_fn = nn.MSELoss()

# Compute the loss output
loss_output = loss_fn(pred, true)


print(loss_output)
