import numpy as np

array1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
mean = np.array([[3, 3, 3, 3, 3]])
std = np.array([[1, 1, 1, 1, 1]])
norm = (array1 - mean) / std
print(norm)