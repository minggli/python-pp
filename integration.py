#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad, trapz
from scipy.stats import norm


def quadratic(z: float) -> float:
    return - np.square(z) + 2 * z - 1


x: np.ndarray = np.linspace(-1, 2, 1000)
y = np.fromiter((quadratic(i) for i in x), dtype=np.float32)

trapezoidal = trapz(y, x)
quadrature, _ = quad(quadratic, -1, 2)
assert np.isclose(trapezoidal, quadrature)


def gaussian_pdf(t: float) -> float:
    return norm.pdf(t)


quadrature, _ = quad(gaussian_pdf, -np.inf, np.inf)
x = np.linspace(-1e2, 1e2, 1e4)
y = np.fromiter((gaussian_pdf(i) for i in x), dtype=np.float32)
trapezoidal = trapz(y, x)
assert np.isclose(trapezoidal, quadrature)
