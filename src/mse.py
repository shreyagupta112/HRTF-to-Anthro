import torch

# Example data
predicted_values = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])  # 2x3 tensor
actual_values = torch.tensor([[1.5, 3.5, 4.5], [4.5, 5.5, 7.5]])    # 2x3 tensor

# Calculate the Mean Squared Error (MSE) across each dimension
mse_per_dimension = torch.mean((predicted_values - actual_values) ** 2, dim=0)

print(mse_per_dimension)

