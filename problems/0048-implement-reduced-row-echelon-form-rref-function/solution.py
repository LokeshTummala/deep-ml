import numpy as np

def rref(matrix):
    """
    Computes the Reduced Row Echelon Form (RREF) of a matrix.
    
    Args:
        matrix: A 2D numpy array or a nested list representing the matrix.
        
    Returns:
        A new 2D numpy array representing the RREF of the input matrix.
    """
    # Convert input to float matrix to handle division properly
    A = np.array(matrix, dtype=np.float64)
    rows, cols = A.shape
    
    # Track the current pivot position
    pivot_row = 0
    
    # Establish a tolerance threshold for floating point zeros
    tol = 1e-9
    
    for pivot_col in range(cols):
        # 1. Find the best row to swap with (Partial Pivoting)
        # We look for the absolute largest element in the current column to maximize numerical stability
        swap_row = pivot_row + np.argmax(np.abs(A[pivot_row:rows, pivot_col]))
        
        # If the largest value in this column is close to zero, skip the column
        if np.abs(A[swap_row, pivot_col]) < tol:
            continue
            
        # 2. Swap the current row with the row containing the maximum pivot element
        if swap_row != pivot_row:
            A[[pivot_row, swap_row]] = A[[swap_row, pivot_row]]
            
        # 3. Scale the pivot row to make the leading coefficient equal 1
        pivot_val = A[pivot_row, pivot_col]
        A[pivot_row] = A[pivot_row] / pivot_val
        
        # 4. Eliminate elements above and below the current pivot entry
        for r in range(rows):
            if r != pivot_row:
                factor = A[r, pivot_col]
                A[r] = A[r] - factor * A[pivot_row]
                
        # Move to the next row for the next pivot
        pivot_row += 1
        if pivot_row >= rows:
            break
            
    # Clean up micro-values close to zero caused by floating-point arithmetic
    A[np.abs(A) < tol] = 0.0
            
    return A
