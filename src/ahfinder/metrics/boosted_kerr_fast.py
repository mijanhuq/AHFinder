"""
Fast boosted Kerr metric using semi-analytical approach.

Computes H and l numerically in rest frame, then combines them
analytically to get gamma, dgamma, and K_ij. This is faster than
full numerical differentiation because:
1. We only need 6 metric evaluations (for dH/dx and dl/dx) instead of 36
2. The combination formulas are computed analytically

Reference: Kerr-Schild form g_μν = η_μν + 2H l_μ l_ν
"""

import numpy as np
from numba import jit
from .base import Metric


@jit(nopython=True, cache=True)
def _compute_H_l_rest_frame(x_rest: np.ndarray, M: float, a: float):
    """
    Compute H and l in the rest frame for Kerr metric.

    Returns H, l (3-vector)
    """
    xr, yr, zr = x_rest

    # Compute Kerr r
    R_sq = xr**2 + yr**2 + zr**2
    if abs(a) < 1e-14:
        r = np.sqrt(R_sq)
    else:
        b = -(R_sq - a**2)
        c = -(a**2) * (zr**2)
        disc = b**2 - 4*c
        if disc < 0:
            disc = 0.0
        u = (-b + np.sqrt(disc)) / 2
        if u < 0:
            u = 0.0
        r = np.sqrt(u)

    if r < 1e-14:
        r = 1e-14

    # H = M*r^3 / (r^4 + a^2*z^2)
    denom = r**4 + a**2 * zr**2
    if denom < 1e-28:
        denom = 1e-28
    H = M * r**3 / denom

    # l = ((rx + ay)/(r^2+a^2), (ry - ax)/(r^2+a^2), z/r)
    r2_a2 = r**2 + a**2
    l = np.array([
        (r * xr + a * yr) / r2_a2,
        (r * yr - a * xr) / r2_a2,
        zr / r
    ])

    return H, l, r


@jit(nopython=True, cache=True)
def _compute_dH_dl_numerical(x_rest: np.ndarray, M: float, a: float, h: float = 1e-6):
    """
    Compute dH/dx_i and dl_j/dx_i numerically in rest frame.

    Returns:
        dH: (3,) array of dH/dx_i
        dl: (3,3) array where dl[j,i] = dl_j/dx_i
    """
    dH = np.zeros(3)
    dl = np.zeros((3, 3))

    for i in range(3):
        # Offset vectors
        x_plus = x_rest.copy()
        x_minus = x_rest.copy()
        x_plus[i] += h
        x_minus[i] -= h

        H_plus, l_plus, _ = _compute_H_l_rest_frame(x_plus, M, a)
        H_minus, l_minus, _ = _compute_H_l_rest_frame(x_minus, M, a)

        dH[i] = (H_plus - H_minus) / (2 * h)
        for j in range(3):
            dl[j, i] = (l_plus[j] - l_minus[j]) / (2 * h)

    return dH, dl


@jit(nopython=True, cache=True)
def _compute_dgamma_from_H_l(H: float, l: np.ndarray, dH: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """
    Compute ∂_k γ_ij from H, l and their derivatives.

    γ_ij = δ_ij + 2H l_i l_j
    ∂_k γ_ij = 2 dH_k l_i l_j + 2H dl_ik l_j + 2H l_i dl_jk

    Note: dl[j,k] = dl_j/dx_k
    """
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


@jit(nopython=True, cache=True)
def _compute_christoffel_from_dgamma(gamma_inv: np.ndarray, dgamma: np.ndarray) -> np.ndarray:
    """Compute Christoffel symbols from dgamma."""
    chris = np.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                val = 0.0
                for l in range(3):
                    val += gamma_inv[k, l] * (
                        dgamma[i, l, j] + dgamma[j, i, l] - dgamma[l, i, j]
                    )
                chris[k, i, j] = 0.5 * val
    return chris


@jit(nopython=True, cache=True)
def _compute_K_from_H_l_chris(
    H: float, l: np.ndarray, dH: np.ndarray, dl: np.ndarray,
    chris: np.ndarray
) -> np.ndarray:
    """
    Compute K_ij from H, l, their derivatives, and Christoffel symbols.

    α = 1/√(1+2H)
    β_j = 2H l_j
    ∂_i β_j = 2 dH_i l_j + 2H dl_ji
    D_i β_j = ∂_i β_j - Γ^k_ij β_k = ∂_i β_j - 2H Γ^k_ij l_k
    K_ij = (1/2α)(D_i β_j + D_j β_i)
    """
    alpha = 1.0 / np.sqrt(1 + 2 * H)

    # Compute D_i β_j
    D_beta = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # ∂_i β_j = 2 dH_i l_j + 2H dl_ji
            d_beta_ij = 2 * dH[i] * l[j] + 2 * H * dl[j, i]

            # Γ^k_ij l_k
            chris_term = 0.0
            for k in range(3):
                chris_term += chris[k, i, j] * l[k]

            # D_i β_j = ∂_i β_j - 2H Γ^k_ij l_k
            D_beta[i, j] = d_beta_ij - 2 * H * chris_term

    # K_ij = (1/2α)(D_i β_j + D_j β_i)
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = 0.5 / alpha * (D_beta[i, j] + D_beta[j, i])

    return K


@jit(nopython=True, cache=True)
def _transform_derivatives_to_lab(
    dH_rest: np.ndarray, dl_rest: np.ndarray,
    Lambda: np.ndarray
) -> tuple:
    """
    Transform derivatives from rest frame to lab frame.

    ∂/∂x_lab_i = Σ_j Λ_ji × ∂/∂x_rest_j

    For a boost, Λ is the coordinate transformation matrix (Lorentz contraction).
    """
    # dH_lab[i] = Σ_j Λ[j,i] * dH_rest[j]
    dH_lab = np.zeros(3)
    for i in range(3):
        for j in range(3):
            dH_lab[i] += Lambda[j, i] * dH_rest[j]

    # dl_lab[k,i] = Σ_j Λ[j,i] * dl_rest[k,j]
    dl_lab = np.zeros((3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                dl_lab[k, i] += Lambda[j, i] * dl_rest[k, j]

    return dH_lab, dl_lab


class FastBoostedKerrMetric(Metric):
    """
    Fast boosted Kerr metric using semi-analytical approach.

    Computes H, l and their derivatives numerically in rest frame,
    then combines them analytically using Kerr-Schild formulas.
    """

    def __init__(self, M: float = 1.0, a: float = 0.0, velocity: np.ndarray = None):
        if M <= 0:
            raise ValueError("Mass must be positive")
        if abs(a) > M:
            raise ValueError("|a| must be ≤ M")

        self.M = M
        self.a = a

        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0])
        velocity = np.asarray(velocity, dtype=float)

        self.velocity = velocity.copy()
        self.v_mag = np.linalg.norm(velocity)

        if self.v_mag >= 1.0:
            raise ValueError("Boost velocity must have |v| < 1")

        if self.v_mag > 1e-14:
            self.lorentz_gamma = 1.0 / np.sqrt(1 - self.v_mag**2)
            self.n_hat = velocity / self.v_mag
        else:
            self.lorentz_gamma = 1.0
            self.n_hat = np.array([1.0, 0.0, 0.0])

        # Coordinate transformation: x_rest = Lambda @ x_lab
        self.Lambda = np.eye(3) + (self.lorentz_gamma - 1) * np.outer(self.n_hat, self.n_hat)

    def _get_rest_coords(self, x: float, y: float, z: float) -> np.ndarray:
        """Transform lab coordinates to rest frame."""
        x_lab = np.array([x, y, z])
        return self.Lambda @ x_lab

    def _compute_all(self, x: float, y: float, z: float):
        """Compute H, l, dH, dl in lab frame."""
        x_rest = self._get_rest_coords(x, y, z)

        # Get H, l in rest frame
        H, l, r = _compute_H_l_rest_frame(x_rest, self.M, self.a)

        # Get derivatives in rest frame
        dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, self.M, self.a)

        # Transform derivatives to lab frame
        dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, self.Lambda)

        return H, l, dH, dl

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        H, l, _, _ = self._compute_all(x, y, z)
        result = np.eye(3)
        for i in range(3):
            for j in range(3):
                result[i, j] += 2 * H * l[i] * l[j]
        return result

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        H, l, _, _ = self._compute_all(x, y, z)
        f = 2 * H / (1 + 2 * H)
        result = np.eye(3)
        for i in range(3):
            for j in range(3):
                result[i, j] -= f * l[i] * l[j]
        return result

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        H, l, dH, dl = self._compute_all(x, y, z)
        return _compute_dgamma_from_H_l(H, l, dH, dl)

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        H, l, dH, dl = self._compute_all(x, y, z)
        gamma_inv = self.gamma_inv(x, y, z)
        dgamma = _compute_dgamma_from_H_l(H, l, dH, dl)
        chris = _compute_christoffel_from_dgamma(gamma_inv, dgamma)
        return _compute_K_from_H_l_chris(H, l, dH, dl, chris)

    def K_trace(self, x: float, y: float, z: float) -> float:
        gamma_inv = self.gamma_inv(x, y, z)
        K = self.extrinsic_curvature(x, y, z)
        return np.sum(gamma_inv * K)

    def lapse(self, x: float, y: float, z: float) -> float:
        H, _, _, _ = self._compute_all(x, y, z)
        return 1.0 / np.sqrt(1 + 2 * H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        H, l, _, _ = self._compute_all(x, y, z)
        return 2 * H * l / (1 + 2 * H)


def fast_boost_kerr(M: float = 1.0, a: float = 0.0, velocity=None) -> FastBoostedKerrMetric:
    """
    Create a fast boosted Kerr metric.

    Args:
        M: Black hole mass
        a: Spin parameter
        velocity: 3-vector boost velocity

    Returns:
        FastBoostedKerrMetric instance
    """
    if velocity is None:
        velocity = [0.0, 0.0, 0.0]
    return FastBoostedKerrMetric(M=M, a=a, velocity=np.array(velocity))


# Warmup
def _warmup():
    m = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.3, 0.0, 0.0]))
    m.gamma(3.0, 0.0, 0.0)
    m.dgamma(3.0, 0.0, 0.0)
    m.extrinsic_curvature(3.0, 0.0, 0.0)

try:
    _warmup()
except Exception:
    pass
