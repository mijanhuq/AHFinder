"""
Kerr metric with analytical derivatives.

Computes dH/dx and dl/dx analytically using chain rule,
which is faster than numerical differentiation and works correctly
with boosted metrics.
"""

import numpy as np
from numba import jit
from .base import Metric


@jit(nopython=True, cache=True)
def _compute_r_kerr(x: float, y: float, z: float, a: float) -> float:
    """Compute Kerr radial coordinate r."""
    R_sq = x**2 + y**2 + z**2

    if abs(a) < 1e-14:
        return np.sqrt(R_sq)

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
def _compute_dr_dx(x: float, y: float, z: float, r: float, a: float) -> np.ndarray:
    """
    Compute dr/dx_i analytically.

    From r^4 - (R^2 - a^2)r^2 - a^2*z^2 = 0, implicit differentiation gives:
    dr/dx = r*x / (r^2 + a^2*z^2/r^2) for x, y
    dr/dz = (r*z + a^2*z/r) / (r^2 + a^2*z^2/r^2)
    """
    if r < 1e-14:
        return np.array([1.0, 0.0, 0.0])

    # Denominator: 2r^3 - (R^2 - a^2)*r = r(2r^2 - R^2 + a^2)
    # But easier form: r^4 + a^2*z^2 gives d(r^4)/dr = 4r^3
    # From implicit diff: 4r^3 dr/dx_i = 2r x_i (for i=x,y) or 2r z + 2a^2 z (for i=z)

    denom = 2 * r * (2 * r**2 - (x**2 + y**2 + z**2) + a**2)
    if abs(denom) < 1e-14:
        denom = 1e-14 if denom >= 0 else -1e-14

    dr = np.zeros(3)
    dr[0] = 2 * r * x / denom
    dr[1] = 2 * r * y / denom
    dr[2] = (2 * r * z + 2 * a**2 * z / r) / denom if r > 1e-10 else z / r

    return dr


@jit(nopython=True, cache=True)
def _compute_H_and_dH(x: float, y: float, z: float, r: float, M: float, a: float):
    """
    Compute H and dH/dx_i analytically.

    H = M*r^3 / (r^4 + a^2*z^2)
    dH/dx_i = M * d/dx_i[r^3 / (r^4 + a^2*z^2)]
    """
    r4 = r**4
    a2z2 = a**2 * z**2
    denom = r4 + a2z2

    if denom < 1e-28:
        denom = 1e-28

    H = M * r**3 / denom

    # Get dr/dx
    dr = _compute_dr_dx(x, y, z, r, a)

    # dH/dx_i = M * [3r^2 * dr/dx_i * denom - r^3 * d(denom)/dx_i] / denom^2
    # d(denom)/dx_i = 4r^3 * dr/dx_i + 2*a^2*z * (1 if i==z else 0)
    dH = np.zeros(3)
    for i in range(3):
        d_denom = 4 * r**3 * dr[i]
        if i == 2:  # z component
            d_denom += 2 * a**2 * z

        dH[i] = M * (3 * r**2 * dr[i] * denom - r**3 * d_denom) / (denom**2)

    return H, dH


@jit(nopython=True, cache=True)
def _compute_l_and_dl(x: float, y: float, z: float, r: float, a: float):
    """
    Compute null vector l and dl/dx_j analytically.

    l = ((rx + ay)/(r^2+a^2), (ry - ax)/(r^2+a^2), z/r)
    """
    if r < 1e-14:
        l = np.array([1.0, 0.0, 0.0])
        dl = np.zeros((3, 3))
        return l, dl

    r2_a2 = r**2 + a**2
    dr = _compute_dr_dx(x, y, z, r, a)

    # l components
    l = np.array([
        (r * x + a * y) / r2_a2,
        (r * y - a * x) / r2_a2,
        z / r
    ])

    # dl_i/dx_j
    # l_0 = (rx + ay) / (r^2 + a^2)
    # l_1 = (ry - ax) / (r^2 + a^2)
    # l_2 = z / r

    dl = np.zeros((3, 3))

    # Derivatives of 1/(r^2 + a^2)
    # d/dx_j[1/(r^2+a^2)] = -2r*dr/dx_j / (r^2+a^2)^2
    d_inv_r2a2 = np.zeros(3)
    for j in range(3):
        d_inv_r2a2[j] = -2 * r * dr[j] / (r2_a2**2)

    # dl_0/dx_j = d/dx_j[(rx + ay) / (r^2+a^2)]
    # = [d(rx+ay)/dx_j * (r^2+a^2) + (rx+ay) * d(r^2+a^2)/dx_j] / (r^2+a^2)^2... wait
    # Actually: d/dx_j[(rx + ay) / (r^2+a^2)] = d(rx+ay)/dx_j / (r^2+a^2) + (rx+ay) * d_inv_r2a2[j]
    for j in range(3):
        # d(rx + ay)/dx_j
        d_num0 = dr[j] * x + (1.0 if j == 0 else 0.0) * r + (1.0 if j == 1 else 0.0) * a
        dl[0, j] = d_num0 / r2_a2 + (r * x + a * y) * d_inv_r2a2[j]

        # d(ry - ax)/dx_j
        d_num1 = dr[j] * y + (1.0 if j == 1 else 0.0) * r - (1.0 if j == 0 else 0.0) * a
        dl[1, j] = d_num1 / r2_a2 + (r * y - a * x) * d_inv_r2a2[j]

        # d(z/r)/dx_j = (delta_j2 * r - z * dr[j]) / r^2
        d_num2 = (1.0 if j == 2 else 0.0) * r - z * dr[j]
        dl[2, j] = d_num2 / (r**2)

    return l, dl


@jit(nopython=True, cache=True)
def gamma_kerr_analytical(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute γ_ij = δ_ij + 2H l_i l_j."""
    r = _compute_r_kerr(x, y, z, a)
    H, _ = _compute_H_and_dH(x, y, z, r, M, a)
    l, _ = _compute_l_and_dl(x, y, z, r, a)

    result = np.eye(3)
    for i in range(3):
        for j in range(3):
            result[i, j] += 2 * H * l[i] * l[j]
    return result


@jit(nopython=True, cache=True)
def gamma_inv_kerr_analytical(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """Compute γ^ij = δ^ij - 2H/(1+2H) l^i l^j."""
    r = _compute_r_kerr(x, y, z, a)
    H, _ = _compute_H_and_dH(x, y, z, r, M, a)
    l, _ = _compute_l_and_dl(x, y, z, r, a)

    factor = 2 * H / (1 + 2 * H)
    result = np.eye(3)
    for i in range(3):
        for j in range(3):
            result[i, j] -= factor * l[i] * l[j]
    return result


@jit(nopython=True, cache=True)
def dgamma_kerr_analytical(x: float, y: float, z: float, M: float, a: float) -> np.ndarray:
    """
    Compute ∂_k γ_ij analytically.

    γ_ij = δ_ij + 2H l_i l_j
    ∂_k γ_ij = 2 dH_k l_i l_j + 2H dl_ik l_j + 2H l_i dl_jk
    """
    r = _compute_r_kerr(x, y, z, a)
    H, dH = _compute_H_and_dH(x, y, z, r, M, a)
    l, dl = _compute_l_and_dl(x, y, z, r, a)

    dgamma = np.zeros((3, 3, 3))

    for k in range(3):
        for i in range(3):
            for j in range(3):
                dgamma[k, i, j] = (
                    2 * dH[k] * l[i] * l[j]
                    + 2 * H * dl[i, k] * l[j]
                    + 2 * H * l[i] * dl[j, k]
                )

    return dgamma


class KerrMetricAnalytical(Metric):
    """
    Kerr metric with analytical derivatives.

    This version computes dgamma analytically using chain rule,
    which works correctly with boosted metrics unlike numerical differentiation.
    """

    def __init__(self, M: float = 1.0, a: float = 0.0):
        if M <= 0:
            raise ValueError("Mass must be positive")
        if abs(a) > M:
            raise ValueError("|a| must be ≤ M")
        self.M = M
        self.a = a

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        return gamma_kerr_analytical(x, y, z, self.M, self.a)

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        return gamma_inv_kerr_analytical(x, y, z, self.M, self.a)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        return dgamma_kerr_analytical(x, y, z, self.M, self.a)

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute K_ij using analytical derivatives."""
        # Import from kerr_fast to avoid code duplication
        from .kerr_fast import extrinsic_curvature_kerr_fast
        return extrinsic_curvature_kerr_fast(x, y, z, self.M, self.a)

    def K_trace(self, x: float, y: float, z: float) -> float:
        gamma_inv = self.gamma_inv(x, y, z)
        K = self.extrinsic_curvature(x, y, z)
        return np.sum(gamma_inv * K)

    def lapse(self, x: float, y: float, z: float) -> float:
        r = _compute_r_kerr(x, y, z, self.a)
        H, _ = _compute_H_and_dH(x, y, z, r, self.M, self.a)
        return 1.0 / np.sqrt(1 + 2 * H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        r = _compute_r_kerr(x, y, z, self.a)
        H, _ = _compute_H_and_dH(x, y, z, r, self.M, self.a)
        l, _ = _compute_l_and_dl(x, y, z, r, self.a)
        return 2 * H * l / (1 + 2 * H)

    def get_H_l_and_derivatives(self, x: float, y: float, z: float):
        """
        Return H, l, dH, dl for use in boosted metrics.

        Returns:
            H: scalar
            l: (3,) array
            dH: (3,) array of dH/dx_i
            dl: (3,3) array where dl[i,j] = dl_i/dx_j
        """
        r = _compute_r_kerr(x, y, z, self.a)
        H, dH = _compute_H_and_dH(x, y, z, r, self.M, self.a)
        l, dl = _compute_l_and_dl(x, y, z, r, self.a)
        return H, l, dH, dl

    def horizon_radius_equatorial(self) -> float:
        return self.M + np.sqrt(self.M**2 - self.a**2)

    def horizon_radius_polar(self) -> float:
        return self.horizon_radius_equatorial()


# Warmup
def _warmup():
    gamma_kerr_analytical(3.0, 0.0, 0.0, 1.0, 0.5)
    dgamma_kerr_analytical(3.0, 0.0, 0.0, 1.0, 0.5)

try:
    _warmup()
except Exception:
    pass
