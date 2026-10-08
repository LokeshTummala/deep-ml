import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float] | int:
    # Handle the edge case where 'a' is an empty list
    if not a or not a[0]:
        return -1
        
    # Get dimensions
    num_columns_a = len(a[0])
    len_b = len(b)

    # Multiplication condition check
    if num_columns_a != len_b:
        return -1
    
    # Calculate the dot product and convert the NumPy array back to a standard Python list
    return np.dot(a, b).tolist()
