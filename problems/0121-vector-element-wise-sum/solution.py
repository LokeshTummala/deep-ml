import numpy as np
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	a = np.array(a)
	b = np.array(b)

	if a.shape != b.shape :
		return -1
	result = []
	for i in range(len(a)):
		result.append(a[i]+b[i])
	return result