import numpy as np
import numdifftools as nd


def is_saddle(func, x_saddle):
    hessian = nd.Hessian(func)(x_saddle)
    eigenvalues = np.linalg.eigvals(hessian)
    return np.any(eigenvalues > 0) and np.any(eigenvalues < 0)


def is_mininmum(func, x_min):
    hessian = nd.Hessian(func)(x_min)
    eigenvalues = np.linalg.eigvals(hessian)
    return np.all(eigenvalues > 0 + 1e-10)


def is_maximum(func, x_min):
    hessian = nd.Hessian(func)(x_min)
    eigenvalues = np.linalg.eigvals(hessian)
    return np.all(eigenvalues < 0)
