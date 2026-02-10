"""
Numba-optimized residual evaluation functions.

These JIT-compiled versions provide significant speedup for the
compute_expansion function which is called many times during
Newton iteration.
"""

import numpy as np
from numba import jit, float64
from numba.types import Tuple


@jit(nopython=True, cache=True)
def compute_christoffel_fast(gamma_inv: np.ndarray, dgamma: np.ndarray) -> np.ndarray:
    """
    Compute Christoffel symbols Γ^k_{ij} from metric derivatives.

    Γ^k_{ij} = (1/2) γ^{kl} (∂_i γ_{lj} + ∂_j γ_{il} - ∂_l γ_{ij})

    Args:
        gamma_inv: Inverse metric γ^{ij}, shape (3, 3)
        dgamma: Metric derivatives ∂_k γ_{ij}, shape (3, 3, 3)

    Returns:
        Christoffel symbols Γ^k_{ij}, shape (3, 3, 3)
    """
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
def compute_expansion_fast(
    grad_phi: np.ndarray,
    hess_phi: np.ndarray,
    gamma_inv: np.ndarray,
    dgamma: np.ndarray,
    K_tensor: np.ndarray,
    K_trace: float
) -> float:
    """
    Compute the expansion Θ of outgoing null normals (JIT-compiled).

    The expansion is:
        Θ = D_i s^i + K_{ij} s^i s^j - K

    Args:
        grad_phi: First derivatives ∂_i φ, shape (3,)
        hess_phi: Second derivatives ∂_i ∂_j φ, shape (3, 3)
        gamma_inv: Inverse metric γ^{ij}, shape (3, 3)
        dgamma: Metric derivatives ∂_k γ_{ij}, shape (3, 3, 3)
        K_tensor: Extrinsic curvature K_{ij}, shape (3, 3)
        K_trace: Trace K = γ^{ij} K_{ij}

    Returns:
        Expansion Θ (should be zero on apparent horizon)
    """
    # ω = γ^{ij} ∂_i φ ∂_j φ = |∇φ|²
    omega = 0.0
    for i in range(3):
        for j in range(3):
            omega += gamma_inv[i, j] * grad_phi[i] * grad_phi[j]

    if omega < 1e-20:
        return 0.0

    sqrt_omega = np.sqrt(omega)

    # n^i = γ^{ij} ∂_j φ (unnormalized normal)
    n_up = np.zeros(3)
    for i in range(3):
        for j in range(3):
            n_up[i] += gamma_inv[i, j] * grad_phi[j]

    # s^i = n^i / √ω (unit outward normal)
    s_up = n_up / sqrt_omega

    # Compute Christoffel symbols
    chris = compute_christoffel_fast(gamma_inv, dgamma)

    # Contracted Christoffel: Γ^k = γ^{ij} Γ^k_{ij}
    Gamma_up = np.zeros(3)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                Gamma_up[k] += gamma_inv[i, j] * chris[k, i, j]

    # Coordinate Laplacian: γ^{ij} ∂_i ∂_j φ
    coord_laplacian = 0.0
    for i in range(3):
        for j in range(3):
            coord_laplacian += gamma_inv[i, j] * hess_phi[i, j]

    # Covariant Laplacian: Δφ = γ^{ij} ∇_i ∇_j φ = γ^{ij} ∂_i ∂_j φ - Γ^k ∂_k φ
    gamma_dot = 0.0
    for k in range(3):
        gamma_dot += Gamma_up[k] * grad_phi[k]
    laplacian = coord_laplacian - gamma_dot

    # Coordinate projection term: (n^i n^j / ω) ∂_i ∂_j φ
    coord_proj = 0.0
    for i in range(3):
        for j in range(3):
            coord_proj += n_up[i] * n_up[j] * hess_phi[i, j]
    coord_proj /= omega

    # Christoffel correction: (n^i n^j Γ^k_{ij} / ω) ∂_k φ
    n_n_chris = np.zeros(3)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                n_n_chris[k] += n_up[i] * n_up[j] * chris[k, i, j]

    chris_proj = 0.0
    for k in range(3):
        chris_proj += n_n_chris[k] * grad_phi[k]
    chris_proj /= omega

    # Covariant projection term
    proj_term = coord_proj - chris_proj

    # Divergence of unit normal: D_i s^i = (Δφ - proj_term) / √ω
    div_s = (laplacian - proj_term) / sqrt_omega

    # Extrinsic curvature term: K_{ij} s^i s^j
    K_ss = 0.0
    for i in range(3):
        for j in range(3):
            K_ss += K_tensor[i, j] * s_up[i] * s_up[j]

    # Expansion: Θ = D_i s^i + K_{ij} s^i s^j - K
    return div_s + K_ss - K_trace


# Warm up JIT compilation
def _warmup():
    """Pre-compile the JIT functions."""
    grad_phi = np.array([1.0, 0.0, 0.0])
    hess_phi = np.zeros((3, 3))
    gamma_inv = np.eye(3)
    dgamma = np.zeros((3, 3, 3))
    K_tensor = np.zeros((3, 3))
    K_trace = 0.0
    compute_expansion_fast(grad_phi, hess_phi, gamma_inv, dgamma, K_tensor, K_trace)


# Warm up on module import
try:
    _warmup()
except Exception:
    pass
