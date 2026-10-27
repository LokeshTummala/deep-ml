import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
        
    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    # Compute the Singular Value Decomposition (SVD) of A
    # A = U * diag(S) * Vh
    _, s, vh = np.linalg.svd(A)
    
    # Identify which singular values are close to zero based on the tolerance
    # Since s only contains non-zero elements up to min(m, n), we account for 
    # the remaining columns of Vh up to n.
    num_singular_values = len(s)
    null_mask = s < tol
    
    # The columns of V (rows of vh) corresponding to small singular values,
    # as well as any columns beyond the length of s, form the null space.
    # We transpose vh to get the vectors as columns.
    v = vh.T
    
    # Extract the vectors corresponding to the null space
    null_space_from_s = v[:, :num_singular_values][:, null_mask]
    null_space_beyond_s = v[:, num_singular_values:]
    
    # Concatenate both sets of vectors to get the full null space
    null_space = np.hstack((null_space_from_s, null_space_beyond_s))
    
    return null_space
