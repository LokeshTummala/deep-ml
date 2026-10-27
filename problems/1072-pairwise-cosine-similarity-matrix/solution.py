import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    X = np.array(X)
    dot_product = np.dot(X,X.T)
    row_norms = np.linalg.norm(X,axis=1)
    norm_outer = np.outer(row_norms,row_norms)
    similarity_matrix = dot_product/(norm_outer + 1e-9)

    return similarity_matrix