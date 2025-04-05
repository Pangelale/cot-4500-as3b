from main.assignment_3 import gaussian_elimination
from main.assignment_3 import LU
from main.assignment_3 import diagonally_dominant
from main.assignment_3 import positive_definite

import numpy as np

# Testing gaussian elimination
A = np.array([[2, -1, 1],
              [1, 3, 1],
              [-1, 5, 4]])

b = np.array([6, 0, -3])
gaussian_elimination(A, b)

# Testing LU
A = np.array([[1, 1, 0, 3],
             [2, 1, -1, 1],
             [3, -1, -1, 2],
             [-1, 2, 3, -1]])

LU(A)

# Testing diagonally dominant
A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])

ans = diagonally_dominant(A)
print(f"\nIs the following matrix diagonally dominate? {ans}")

# Testing positive definite
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

res = positive_definite(A)
print(f"Is the following matrix positive definite? {res}")