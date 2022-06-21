import numpy as np


def is_matrix_quadratic(matrix: list[list[float]]) -> bool:
    first_dimension = len(matrix)
    for line in matrix:
        if len(line) != first_dimension:
            return False
    return True


def is_upper_triangular(matrix: np.ndarray):
    return np.alltrue(np.triu(matrix) == matrix)


def is_diagonal(matrix: np.ndarray):
    return np.alltrue(np.diag(matrix) == matrix)


def swap_down(matrix: np.ndarray, i) -> int:
    k = i + 1
    while matrix[i][i] == 0 and k < matrix.shape[0]:
        if matrix[k][i] != 0:
            matrix[[i, k]] = matrix[[k, i]]
            return k
        k += 1
    return -1
