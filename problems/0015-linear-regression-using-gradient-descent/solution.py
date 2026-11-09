import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
  """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for
          intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
  m, n = X.shape
  y = y.reshape(-1, 1)  # Ensure y is a column vector of shape (m, 1)
  theta = np.zeros((n, 1))  # Initialize weights to zeros of shape (n, 1)

  for _ in range(iterations):
    # 1. Calculate predictions: (m, n) @ (n, 1) -> (m, 1)
    predictions = X @ theta

    # 2. Calculate prediction errors: (m, 1)
    errors = predictions - y

    # 3. Compute the gradient of MSE loss: (n, m) @ (m, 1) -> (n, 1)
    gradient = (1 / m) * (X.T @ errors)

    # 4. Update weights in the direction of steepest descent
    theta -= alpha * gradient

  return theta.flatten()
