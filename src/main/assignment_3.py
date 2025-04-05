import numpy as np

# Gaussian elimination & backward substitution
def gaussian_elimination(A, b):
    n = len(b) # Num of eq.
    
    # Forward Elimination for upper triangular matrix
    for k in range(n): # Loop over diagonals
        for i in range(k+1, n): # Starting below pivot, looping over rows
            ratio = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= ratio * A[k, j]
            b[i] -= ratio * b[k]
    
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1][n-1]
    
    # Back Substitution
    for i in range(n-1, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += A[i, j] * x[j]
        x[i] = (b[i] - sum) / A[i, i]
    
    x = x.astype(int)
    print(x)        

# LU Factorization
def LU(A):
    n = len(A)
    L = np.eye(n) # Identity matrix
    U = np.copy(A)
    
    # Perform LU decomposition
    for k in range(n):
        for i in range(k+1, n):
            ratio = U[i, k] / U[k, k]
            L[i, k] = ratio  

            for j in range(k, n):
                U[i, j] -= ratio * U[k, j]  # U's upper triangular
    
    determinant = np.linalg.det(A)
    print(f"\n{determinant}")
    print(f"\n{L}")
    print(f"\n{U}")

# Determine diagonally dominant
def diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        sum = 0
        for j in range(n):
            if i != j:
                sum += A[i,j]
        if A[i, i] < sum:
            return False
    return True 

# Determine positive definite
def positive_definite(A):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):  
            if A[i, j] != A[j, i]:
                return False

    eigenvalues = np.linalg.eigvals(A)
    
    if np.all(eigenvalues) > 0:
        return True
