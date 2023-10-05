# Set of helper methods for applying different operations on vectors or metrices

import numpy as np


def normalize_row_vectors(X):
    """
        returns the normalized version of the row vectors in X(each row vector of X divided by its norm)
    """
    row_square_sum = np.sum(X * X, axis=1)
    row_sqrt = np.sqrt(row_square_sum)
    X_normalized = np.divide(X, row_sqrt[:, None], out=X, where=X != 0) 
    return X_normalized


def cos_similarity(X1, X2):
    """
        return the cosine similarity between each row of X1 and each row of X2 matrices
    """
    X1_norm = normalize_row_vectors(X1)
    X2_norm = normalize_row_vectors(X2)
    cos_matrix = np.matmul(X1_norm, X2_norm.T)
    return cos_matrix


def minkowski_distance(v1, v2, p):
    """
        return the minkowski distance between two vectors v1, and v2
        When p = 1, the distance becomes manhattan_distance
        When p = 2, the distance becomes euclidean_distance
    """
    return sum(pow(abs(e1 - e2), p) for e1, e2 in zip(v1, v2)) ** (1 / p)


def euclidean_distance(X1, X2):
    """
        returns the euclidean distance between each row of X1 with each row of X2
    """
    euclidean_matrix = np.zeros((X1.shape[0], X2.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(0, X2.shape[1]):
            # euclidean_matrix[i, j] = minkowski_distance(v1=X1[i], v2=X2[:, j], p=2)
            euclidean_matrix[i, j] = np.linalg.norm(X1[i] - X2[:, j])
    return euclidean_matrix
