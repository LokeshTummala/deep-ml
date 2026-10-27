import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# Convert inputs to float arrays for numerical stability
	A = np.array(A, dtype=float)
	b = np.array(b, dtype=float)
	
	# Set up initial guess
	if x0 is None:
		x = np.zeros_like(b)
	else:
		x = np.array(x0, dtype=float)
		
	# 1. Calculate initial residual vector (r = b - Ax)
	r = b - A @ x
	
	# 2. Initialize search direction vector (p = r)
	p = r.copy()
	
	# Pre-compute residual dot product for the first iteration
	r_dot_old = np.dot(r, r)
	
	# Main iteration loop
	for _ in range(n):
		# If the initial residual is already below tolerance, exit early
		if np.sqrt(r_dot_old) < tol:
			break
			
		# Compute matrix-vector product once per loop for efficiency
		Ap = A @ p
		
		# Step size alpha
		alpha = r_dot_old / np.dot(p, Ap)
		
		# Update solution estimate
		x = x + alpha * p
		
		# Update residual vector
		r = r - alpha * Ap
		
		# Convergence check using the Euclidean norm of the residual
		r_dot_new = np.dot(r, r)
		if np.sqrt(r_dot_new) < tol:
			break
			
		# Improvement factor beta
		beta = r_dot_new / r_dot_old
		
		# Update search direction
		p = r + beta * p
		
		# Save current residual dot product for the next iteration
		r_dot_old = r_dot_new
		
	return x
