import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    if norm_type == 'l1':
        # Flattening ensures the norm treats 2D arrays as entrywise (vector) norms
        return float(np.linalg.norm(arr.ravel(), ord=1))
    elif norm_type == 'l2':
        return float(np.linalg.norm(arr.ravel(), ord=2))
    elif norm_type == 'linf':
        return float(np.linalg.norm(arr.ravel(), ord=np.inf))
    elif norm_type == "frobenius":
        # Successfully fixed dimension check
        if np.ndim(arr) != 2:
            raise ValueError("ValueError")
        else:
            return float(np.linalg.norm(arr, ord='fro'))
    else : 
        raise ValueError("ValueError")
