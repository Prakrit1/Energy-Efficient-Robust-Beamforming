import numpy as np

def neumann_matrix_inversion_approximation(
    matrix: np.ndarray,
    order: int,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    identity_matrix = np.eye(n, dtype=matrix.dtype)

    fro_norm = np.linalg.norm(matrix, ord='fro')
    if fro_norm == 0 or not np.isfinite(fro_norm):
        return np.zeros_like(matrix)

    normalized_matrix = matrix / fro_norm

    if not np.allclose(normalized_matrix, normalized_matrix.conj().T, atol=1e-12):
        return np.zeros_like(matrix)

    try:
        eigvals = np.linalg.eigvalsh(normalized_matrix)
    except Exception:
        return np.zeros_like(matrix)

    if not np.all(np.isfinite(eigvals)):
        return np.zeros_like(matrix)

    eigenvalue_min = np.min(eigvals)
    eigenvalue_max = np.max(eigvals)

    if eigenvalue_min <= 1e-12:
        return np.zeros_like(matrix)

    denominator = eigenvalue_min + eigenvalue_max
    if denominator <= 0 or not np.isfinite(denominator):
        return np.zeros_like(matrix)

    scaling_factor = 2.0 / denominator
    if not np.isfinite(scaling_factor):
        return np.zeros_like(matrix)

    operator = identity_matrix - scaling_factor * normalized_matrix
    if not np.all(np.isfinite(operator)):
        return np.zeros_like(matrix)

    approx = identity_matrix.copy()
    power = identity_matrix.copy()

    for _ in range(order):
        power = power @ operator
        if not np.all(np.isfinite(power)):
            return np.zeros_like(matrix)

        approx += power
        if not np.all(np.isfinite(approx)):
            return np.zeros_like(matrix)

    normalized_inverse_approximation = scaling_factor * approx
    if not np.all(np.isfinite(normalized_inverse_approximation)):
        return np.zeros_like(matrix)

    matrix_inverse_approximation = normalized_inverse_approximation / fro_norm
    if not np.all(np.isfinite(matrix_inverse_approximation)):
        return np.zeros_like(matrix)

    return matrix_inverse_approximation
