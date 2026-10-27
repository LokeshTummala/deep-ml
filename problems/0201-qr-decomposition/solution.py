import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	# Convert input to a floating-point NumPy array
	A_arr = np.array(A, dtype=float)
	m, n = A_arr.shape
	
	Q = np.zeros((m, n))
	R = np.zeros((n, n))
	
	for j in range(n):
		# Start with the original column vector from matrix A
		v = A_arr[:, j]
		
		for i in range(j):
			# Calculate projection coefficients
			R[i, j] = np.dot(Q[:, i], A_arr[:, j])
			# Subtract projection to enforce orthogonality
			v = v - R[i, j] * Q[:, i]
			
		# Compute the magnitude norm of the vector
		R[j, j] = np.linalg.norm(v)
		
		if np.isclose(R[j, j], 0.0):
			raise ValueError("Columns of matrix A are linearly dependent. Classical Gram-Schmidt cannot continue.")
			
		# Normalize vector to enforce orthonormality
		Q[:, j] = v / R[j, j]
		
	# Convert back to native Python lists of lists to match return hints
	return Q.tolist(), R.tolist()
