# -*- encoding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_triangular


def ols(X, y):
    """Frequentist Oridinary Least Square solution."""
    # X @ B = y
    # X.T @ X @ B = X.T @ y
    # Cholesky factorize covariance matrix into upper and lower triangles
    # L.T @ L @ B = X.T @ y
    # let L @ B = r and X.T @ y = d
    # solve linear system L.T @ r = d using back substitution
    # then solve linear sysetm L @ B = r using forward substitution
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    # reduce along rows and substract means of variables.
    X_offset, y_offset = np.mean(X, axis=0), np.mean(y, axis=0)
    X -= X_offset
    y -= y_offset

    L = np.linalg.cholesky(X.T @ X)
    d = X.T @ y
    # back substituion
    r = solve_triangular(L.T, d, lower=False)
    # forward substituion
    theta = solve_triangular(L, r, lower=True)
    intercept = y_offset - np.dot(X_offset, theta)
    return intercept, theta