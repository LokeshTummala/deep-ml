import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    
    Args:
        A: Square matrix as a list of lists
    
    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
        
    # Initialize L with 1s on the diagonal and U with 0s
    L = np.eye(n)
    U = np.zeros((n, n))
        
    for i in range(n):
        # 1. Evaluate elements for the Upper Triangular Matrix (U)
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            
        # 2. Evaluate elements for the Lower Triangular Matrix (L)
        for j in range(i + 1, n):
            if np.isclose(U[i, i], 0.0):
                raise ValueError("Zero pivot encountered. Pure Doolittle's method without pivoting fails.")
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
                
    return L, U
