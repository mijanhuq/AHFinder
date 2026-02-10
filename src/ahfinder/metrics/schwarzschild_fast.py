"""
Numba-optimized Schwarzschild metric functions.

Provides JIT-compiled versions of the Schwarzschild metric computations
for significant speedup.
"""

import numpy as np
from numba import jit
from .base import Metric


@jit(nopython=True, cache=True)
def _r_and_l_fast(x: float, y: float, z: float):
    """Compute r and the null vector l^i."""
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-14:
        return 1e-14, np.array([1.0, 0.0, 0.0])
    l = np.array([x/r, y/r, z/r])
    return r, l


@jit(nopython=True, cache=True)
def gamma_fast(x: float, y: float, z: float, M: float) -> np.ndarray:
    """Compute 3-metric γ_ij = δ_ij + 2H l_i l_j."""
    r, l = _r_and_l_fast(x, y, z)
    H = M / r

    result = np.eye(3)
    for i in range(3):
        for j in range(3):
            result[i, j] += 2 * H * l[i] * l[j]
    return result


@jit(nopython=True, cache=True)
def gamma_inv_fast(x: float, y: float, z: float, M: float) -> np.ndarray:
    """Compute inverse 3-metric γ^ij."""
    r, l = _r_and_l_fast(x, y, z)
    H = M / r
    factor = 2 * H / (1 + 2 * H)

    result = np.eye(3)
    for i in range(3):
        for j in range(3):
            result[i, j] -= factor * l[i] * l[j]
    return result


@jit(nopython=True, cache=True)
def dgamma_fast(x: float, y: float, z: float, M: float) -> np.ndarray:
    """
    Compute ∂_k γ_ij analytically.

    γ_ij = δ_ij + 2H l_i l_j
    ∂_k γ_ij = 2 (∂_k H) l_i l_j + 2H (∂_k l_i) l_j + 2H l_i (∂_k l_j)
    """
    r, l = _r_and_l_fast(x, y, z)
    H = M / r

    dgamma = np.zeros((3, 3, 3))

    for k in range(3):
        # ∂_k H = -H l_k / r
        dH_dk = -H * l[k] / r

        for i in range(3):
            # ∂_k l_i = (δ_ki - l_k l_i) / r
            delta_ki = 1.0 if k == i else 0.0
            dl_i_dk = (delta_ki - l[k] * l[i]) / r

            for j in range(3):
                delta_kj = 1.0 if k == j else 0.0
                dl_j_dk = (delta_kj - l[k] * l[j]) / r

                dgamma[k, i, j] = (
                    2 * dH_dk * l[i] * l[j]
                    + 2 * H * dl_i_dk * l[j]
                    + 2 * H * l[i] * dl_j_dk
                )

    return dgamma


@jit(nopython=True, cache=True)
def extrinsic_curvature_fast(x: float, y: float, z: float, M: float) -> np.ndarray:
    """Compute extrinsic curvature K_ij."""
    r, l = _r_and_l_fast(x, y, z)
    H = M / r
    alpha = 1.0 / np.sqrt(1 + 2 * H)

    factor = 2 * H * alpha / r

    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            delta_ij = 1.0 if i == j else 0.0
            K[i, j] = factor * (delta_ij - (2 + H) * l[i] * l[j])

    return K


@jit(nopython=True, cache=True)
def K_trace_fast(x: float, y: float, z: float, M: float) -> float:
    """Compute trace K = γ^ij K_ij."""
    gamma_inv = gamma_inv_fast(x, y, z, M)
    K = extrinsic_curvature_fast(x, y, z, M)

    trace = 0.0
    for i in range(3):
        for j in range(3):
            trace += gamma_inv[i, j] * K[i, j]
    return trace


class SchwarzschildMetricFast(Metric):
    """
    Fast Schwarzschild spacetime using Numba JIT compilation.

    Drop-in replacement for SchwarzschildMetric with significant speedup.
    """

    def __init__(self, M: float = 1.0):
        if M <= 0:
            raise ValueError("Mass must be positive")
        self.M = M

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        return gamma_fast(x, y, z, self.M)

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        return gamma_inv_fast(x, y, z, self.M)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        return dgamma_fast(x, y, z, self.M)

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        return extrinsic_curvature_fast(x, y, z, self.M)

    def K_trace(self, x: float, y: float, z: float) -> float:
        return K_trace_fast(x, y, z, self.M)

    def lapse(self, x: float, y: float, z: float) -> float:
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-14:
            r = 1e-14
        H = self.M / r
        return 1.0 / np.sqrt(1 + 2 * H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        r, l = _r_and_l_fast(x, y, z)
        H = self.M / r
        return 2 * H * l / (1 + 2 * H)

    def horizon_radius(self) -> float:
        return 2 * self.M


# Warm up JIT compilation
def _warmup():
    gamma_fast(1.0, 0.0, 0.0, 1.0)
    gamma_inv_fast(1.0, 0.0, 0.0, 1.0)
    dgamma_fast(1.0, 0.0, 0.0, 1.0)
    extrinsic_curvature_fast(1.0, 0.0, 0.0, 1.0)
    K_trace_fast(1.0, 0.0, 0.0, 1.0)

try:
    _warmup()
except Exception:
    pass
