import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    try:
        # 1. Safely convert inputs to PyTorch tensors first
        a_t = torch.as_tensor(a, dtype=torch.float)
        b_t = torch.as_tensor(b, dtype=torch.float)
    except (ValueError, TypeError):
        return torch.tensor(-1)

    # 2. Check for empty inputs using .numel() (safely handles lists, numpy, and torch)
    if a_t.numel() == 0 or b_t.numel() == 0:
        return torch.tensor(-1)
        
    # 3. Ensure 'a' is 2D and 'b' is 1D before checking matching inner dimensions
    if a_t.ndim != 2 or b_t.ndim != 1 or a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
        
    # 4. Compute and return the 1-D tensor product
    return torch.matmul(a_t, b_t)

