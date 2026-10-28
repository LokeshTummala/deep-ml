import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # 1. Create polynomial objects from the coefficients
    g = np.poly1d(g_coeffs)
    h = np.poly1d(h_coeffs)
    
    # 2. Compute their derivatives
    g_prime = np.polyder(g)
    h_prime = np.polyder(h)
    
    # 3. Evaluate polynomials and their derivatives at the point x
    g_x = g(x)
    h_x = h(x)
    g_prime_x = g_prime(x)
    h_prime_x = h_prime(x)
    
    # 4. Check for division by zero
    if h_x == 0:
        raise ZeroDivisionError("The denominator polynomial evaluates to zero at the given x.")
        
    # 5. Apply the quotient rule formula
    derivative_value = (g_prime_x * h_x - g_x * h_prime_x) / (h_x ** 2)
    
    return float(derivative_value)
