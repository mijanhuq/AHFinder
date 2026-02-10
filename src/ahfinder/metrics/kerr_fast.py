"""
Numba-optimized Kerr metric functions.

Provides JIT-compiled versions of the Kerr metric computations
for significant speedup.
"""

import numpy as np
from numba import jit
from .base import Metric


@jit(nopython=True, cache=True)
def _compute_r_fast(x: float, y: float, z: float, a: float) -> float:
    """
    Compute the Kerr radial coordinate r from Cartesian coordinates.

    r is defined implicitly by: x² + y² + z² = r² + a²(1 - z²/r²)
    which can be rewritten as: r⁴ - (x² + y² + z² - a²)r² - a²z² = 0
    """
    R_sq = x**2 + y**2 + z**2

    if abs(a) < 1e-14:
        return np.sqrt(R_sq)

    # Solve r⁴ - (R² - a²)r² - a²z² = 0
    # Let u = r², then: u² - (R² - a²)u - a²z² = 0
    b = -(R_sq - a**2)
    c = -(a**2) * (z**2)

    discriminant = b**2 - 4 * c
    if discriminant < 0:
        discriminant = 0.0

    u = (-b + np.sqrt(discriminant)) / 2

    if u < 0:
        u = 0.0

    return np.sqrt(u)


@jit(nopython=True, cache=True)
def _r_and_l_fast(x: float, y: float, z: float, a: float):
    """
    Compute Kerr r and the null vector l^i.

    l^i = ((rx + ay)/(r² + a²), (ry - ax)/(r² + a²), z/r)
    """
    r = _compute_r_fast(x, y, z, a)

    if r < 1e-14:
        return 1e-14, np.array([1.0, 0.0, 0.0])

    r2_a2 = r**2 + a**2

    l = np.array([
        (r * x + a * y) / r2_a2,
        (r * y - a * x) / r2_a2,
        z / r
    ])

    return r, l


@jit(nopython=True, cache=True)
def _H_fast(r: float, z: float, M: float, a: float) -> float:
    """Compute H = Mr³ / (r⁴ + a²z²)."""
    denom = r**4 + a**2 * z**2
    if denom < 1e-28:
        return M * r**3 / 1e-28
    return M * r**3 / denom


@jit(nopython=True, cache=True)
def gamma_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute 3-metric γ_ij = δ_ij + 2H l_i l_j."""
    r, l = _r_and_l_fast(x, y, z, a)
    H = _H_fast(r, z, M, a)

    result = np.eye(3)
    for i in range(3):
        for j in range(3):
            result[i, j] += 2 * H * l[i] * l[j]
    return result


@jit(nopython=True, cache=True)
def gamma_inv_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute inverse 3-metric γ^ij."""
    r, l = _r_and_l_fast(x, y, z, a)
    H = _H_fast(r, z, M, a)
    factor = 2 * H / (1 + 2 * H)

    result = np.eye(3)
    for i in range(3):
        for j in range(3):
            result[i, j] -= factor * l[i] * l[j]
    return result


@jit(nopython=True, cache=True)
def dgamma_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute ∂_k γ_ij numerically with JIT."""
    h = 1e-6
    dgamma = np.zeros((3, 3, 3))

    # Compute gamma at offset points
    g_xp = gamma_kerr_fast(x + h, y, z, M, a)
    g_xm = gamma_kerr_fast(x - h, y, z, M, a)
    g_yp = gamma_kerr_fast(x, y + h, z, M, a)
    g_ym = gamma_kerr_fast(x, y - h, z, M, a)
    g_zp = gamma_kerr_fast(x, y, z + h, M, a)
    g_zm = gamma_kerr_fast(x, y, z - h, M, a)

    for i in range(3):
        for j in range(3):
            dgamma[0, i, j] = (g_xp[i, j] - g_xm[i, j]) / (2 * h)
            dgamma[1, i, j] = (g_yp[i, j] - g_ym[i, j]) / (2 * h)
            dgamma[2, i, j] = (g_zp[i, j] - g_zm[i, j]) / (2 * h)

    return dgamma


@jit(nopython=True, cache=True)
def shift_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute shift β^i = 2H l^i / (1 + 2H)."""
    r, l = _r_and_l_fast(x, y, z, a)
    H = _H_fast(r, z, M, a)
    factor = 2 * H / (1 + 2 * H)
    return factor * l


@jit(nopython=True, cache=True)
def lapse_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> float:
    """Compute lapse α = 1/√(1 + 2H)."""
    r = _compute_r_fast(x, y, z, a)
    H = _H_fast(r, z, M, a)
    return 1.0 / np.sqrt(1 + 2 * H)


@jit(nopython=True, cache=True)
def christoffel_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute Christoffel symbols Γ^i_jk."""
    gamma_inv = gamma_inv_kerr_fast(x, y, z, M, a)
    dg = dgamma_kerr_fast(x, y, z, M, a)

    chris = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0.0
                for l in range(3):
                    val += gamma_inv[i, l] * (
                        dg[j, l, k] + dg[k, j, l] - dg[l, j, k]
                    )
                chris[i, j, k] = 0.5 * val
    return chris


@jit(nopython=True, cache=True)
def extrinsic_curvature_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """
    Compute extrinsic curvature K_ij.

    K_ij = (1/2α)(D_i β_j + D_j β_i) for stationary spacetimes
    """
    h = 1e-6

    r = _compute_r_fast(x, y, z, a)
    H = _H_fast(r, z, M, a)

    if r < 1e-10:
        return np.zeros((3, 3))

    alpha = 1.0 / np.sqrt(1 + 2 * H)
    gamma_down = gamma_kerr_fast(x, y, z, M, a)
    chris = christoffel_kerr_fast(x, y, z, M, a)
    beta = shift_kerr_fast(x, y, z, M, a)

    # Lower index on beta: β_j = γ_jk β^k
    beta_down = np.zeros(3)
    for j in range(3):
        for k in range(3):
            beta_down[j] += gamma_down[j, k] * beta[k]

    # Compute ∂_i β_j (lowered) numerically
    d_beta_down = np.zeros((3, 3))

    for i_dir in range(3):
        # Positive offset
        if i_dir == 0:
            xp, yp, zp = x + h, y, z
            xm, ym, zm = x - h, y, z
        elif i_dir == 1:
            xp, yp, zp = x, y + h, z
            xm, ym, zm = x, y - h, z
        else:
            xp, yp, zp = x, y, z + h
            xm, ym, zm = x, y, z - h

        gamma_p = gamma_kerr_fast(xp, yp, zp, M, a)
        beta_p = shift_kerr_fast(xp, yp, zp, M, a)
        gamma_m = gamma_kerr_fast(xm, ym, zm, M, a)
        beta_m = shift_kerr_fast(xm, ym, zm, M, a)

        # β_j = γ_jk β^k
        beta_down_p = np.zeros(3)
        beta_down_m = np.zeros(3)
        for j in range(3):
            for k in range(3):
                beta_down_p[j] += gamma_p[j, k] * beta_p[k]
                beta_down_m[j] += gamma_m[j, k] * beta_m[k]

        for j in range(3):
            d_beta_down[i_dir, j] = (beta_down_p[j] - beta_down_m[j]) / (2 * h)

    # D_i β_j = ∂_i β_j - Γ^k_ij β_k
    D_beta = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            D_beta[i, j] = d_beta_down[i, j]
            for k in range(3):
                D_beta[i, j] -= chris[k, i, j] * beta_down[k]

    # K_ij = (1/2α)(D_i β_j + D_j β_i)
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = 0.5 / alpha * (D_beta[i, j] + D_beta[j, i])

    return K


@jit(nopython=True, cache=True)
def K_trace_kerr_fast(x: float, y: float, z: float, M: float, a: float) -> float:
    """Compute trace K = γ^ij K_ij."""
    gamma_inv = gamma_inv_kerr_fast(x, y, z, M, a)
    K = extrinsic_curvature_kerr_fast(x, y, z, M, a)

    trace = 0.0
    for i in range(3):
        for j in range(3):
            trace += gamma_inv[i, j] * K[i, j]
    return trace


class KerrMetricFast(Metric):
    """
    Fast Kerr spacetime using Numba JIT compilation.

    Drop-in replacement for KerrMetric with significant speedup.
    """

    def __init__(self, M: float = 1.0, a: float = 0.0):
        if M <= 0:
            raise ValueError("Mass must be positive")
        if abs(a) > M:
            raise ValueError("|a| must be ≤ M for Kerr black hole")
        self.M = M
        self.a = a

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        return gamma_kerr_fast(x, y, z, self.M, self.a)

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        return gamma_inv_kerr_fast(x, y, z, self.M, self.a)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        return dgamma_kerr_fast(x, y, z, self.M, self.a)

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        return extrinsic_curvature_kerr_fast(x, y, z, self.M, self.a)

    def K_trace(self, x: float, y: float, z: float) -> float:
        return K_trace_kerr_fast(x, y, z, self.M, self.a)

    def lapse(self, x: float, y: float, z: float) -> float:
        return lapse_kerr_fast(x, y, z, self.M, self.a)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        return shift_kerr_fast(x, y, z, self.M, self.a)

    def horizon_radius_equatorial(self) -> float:
        return self.M + np.sqrt(self.M**2 - self.a**2)

    def horizon_radius_polar(self) -> float:
        return self.horizon_radius_equatorial()

    def horizon_area(self) -> float:
        r_plus = self.horizon_radius_equatorial()
        return 4 * np.pi * (r_plus**2 + self.a**2)

    def irreducible_mass(self) -> float:
        A = self.horizon_area()
        return np.sqrt(A / (16 * np.pi))


# Warm up JIT compilation
def _warmup():
    gamma_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)
    gamma_inv_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)
    dgamma_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)
    shift_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)
    christoffel_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)
    extrinsic_curvature_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)
    K_trace_kerr_fast(3.0, 0.0, 0.0, 1.0, 0.5)

try:
    _warmup()
except Exception:
    pass
