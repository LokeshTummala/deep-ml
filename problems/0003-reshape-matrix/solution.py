import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	a = np.array(a)
	prod_curr = a.shape[0]*a.shape[1]
	prod_new = new_shape[0]*new_shape[1]

	if prod_curr != prod_new:
		return []
	reshaped_matrix = a.reshape(new_shape)
	return reshaped_matrix