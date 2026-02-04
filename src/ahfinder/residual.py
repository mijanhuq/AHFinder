"""
Residual evaluation for the apparent horizon equation.

The apparent horizon is the outermost marginally outer trapped surface (MOTS),
where the expansion of outgoing null normals vanishes:

    Θ = D_i s^i + K_ij s^i s^j - K = 0

where s^i is the outward unit normal to the surface.

Reference: Huq, Choptuik & Matzner (2000), Section II.D
"""

import numpy as np
from typing import Tuple, Optional
from .surface import SurfaceMesh
from .stencil import CartesianStencil
from .interpolation import BiquarticInterpolator
from .metrics.base import Metric


def compute_expansion(
    grad_phi: np.ndarray,
    hess_phi: np.ndarray,
    gamma_inv: np.ndarray,
    dgamma: np.ndarray,
    K_tensor: np.ndarray,
    K_trace: float
) -> float:
    """
    Compute the expansion Θ of outgoing null normals.

    The level set function is φ = r - ρ(θ, φ), so the surface is φ = 0.
    The outward unit normal is s^i = γ^{ij} ∂_j φ / |∇φ|

    The expansion is:
        Θ = D_i s^i + K_{ij} s^i s^j - K

    Using the projection formula:
        D_i s^i = (1/√ω) [Δφ - (n^i n^j / ω) ∇_i ∇_j φ]

    where:
        - ω = γ^{ij} ∂_i φ ∂_j φ
        - n^i = γ^{ij} ∂_j φ
        - Δφ = γ^{ij} ∇_i ∇_j φ (covariant Laplacian)
        - ∇_i ∇_j φ = ∂_i ∂_j φ - Γ^k_{ij} ∂_k φ

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
    omega = np.einsum('ij,i,j->', gamma_inv, grad_phi, grad_phi)

    if omega < 1e-20:
        return 0.0

    sqrt_omega = np.sqrt(omega)

    # n^i = γ^{ij} ∂_j φ (unnormalized normal)
    n_up = np.einsum('ij,j->i', gamma_inv, grad_phi)

    # s^i = n^i / √ω (unit outward normal)
    s_up = n_up / sqrt_omega

    # Compute Christoffel symbols Γ^k_{ij} from metric derivatives
    # Γ^k_{ij} = (1/2) γ^{kl} (∂_i γ_{lj} + ∂_j γ_{il} - ∂_l γ_{ij})
    chris = np.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    chris[k, i, j] += 0.5 * gamma_inv[k, l] * (
                        dgamma[i, l, j] + dgamma[j, i, l] - dgamma[l, i, j]
                    )

    # Contracted Christoffel: Γ^k = γ^{ij} Γ^k_{ij}
    Gamma_up = np.einsum('ij,kij->k', gamma_inv, chris)

    # Coordinate Laplacian: γ^{ij} ∂_i ∂_j φ
    coord_laplacian = np.einsum('ij,ij->', gamma_inv, hess_phi)

    # Covariant Laplacian: Δφ = γ^{ij} ∇_i ∇_j φ = γ^{ij} ∂_i ∂_j φ - Γ^k ∂_k φ
    laplacian = coord_laplacian - np.dot(Gamma_up, grad_phi)

    # Coordinate projection term: (n^i n^j / ω) ∂_i ∂_j φ
    coord_proj = np.einsum('i,j,ij->', n_up, n_up, hess_phi) / omega

    # Christoffel correction to projection: (n^i n^j / ω) Γ^k_{ij} ∂_k φ
    # = (n^i n^j Γ^k_{ij} / ω) ∂_k φ
    n_n_chris = np.einsum('i,j,kij->k', n_up, n_up, chris)
    chris_proj = np.dot(n_n_chris, grad_phi) / omega

    # Covariant projection term: (n^i n^j / ω) ∇_i ∇_j φ
    proj_term = coord_proj - chris_proj

    # Divergence of unit normal: D_i s^i = (Δφ - proj_term) / √ω
    div_s = (laplacian - proj_term) / sqrt_omega

    # Extrinsic curvature term: K_{ij} s^i s^j
    K_ss = np.einsum('ij,i,j->', K_tensor, s_up, s_up)

    # Expansion: Θ = D_i s^i + K_{ij} s^i s^j - K
    Theta = div_s + K_ss - K_trace

    return Theta


def compute_dgamma_inv(
    gamma_inv: np.ndarray,
    dgamma: np.ndarray
) -> np.ndarray:
    """
    Compute derivatives of inverse metric from metric derivatives.

    ∂_k γ^{ab} = -γ^{ac} γ^{bd} ∂_k γ_{cd}

    Args:
        gamma_inv: Inverse metric γ^ab, shape (3, 3)
        dgamma: Metric derivatives ∂_k γ_ab, shape (3, 3, 3)

    Returns:
        Array of shape (3, 3, 3) with ∂_k γ^ab
    """
    dgamma_inv = np.zeros((3, 3, 3))

    for k in range(3):
        dgamma_inv[k] = -gamma_inv @ dgamma[k] @ gamma_inv

    return dgamma_inv


class ResidualEvaluator:
    """
    Evaluates the residual F[ρ] = Θ for the apparent horizon equation.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        stencil: CartesianStencil,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        """
        Initialize residual evaluator.

        Args:
            mesh: SurfaceMesh instance
            stencil: CartesianStencil for derivative computation
            metric: Metric providing geometric data
            center: Center of coordinate system
        """
        self.mesh = mesh
        self.stencil = stencil
        self.metric = metric
        self.center = center

    def evaluate_at_point(
        self,
        rho: np.ndarray,
        i_theta: int,
        i_phi: int
    ) -> float:
        """
        Evaluate F[ρ] at a single grid point.

        Args:
            rho: Full (N_s, N_s) grid of radial values
            i_theta, i_phi: Grid point indices

        Returns:
            Value of F[ρ] at this point
        """
        mesh = self.mesh
        cx, cy, cz = self.center

        # Get surface point coordinates
        theta = mesh.theta[i_theta]
        phi = mesh.phi[i_phi]
        r = rho[i_theta, i_phi]

        x0 = cx + r * np.sin(theta) * np.cos(phi)
        y0 = cy + r * np.sin(theta) * np.sin(phi)
        z0 = cz + r * np.cos(theta)

        # Compute φ derivatives using Cartesian stencil
        grad_phi, hess_phi = self.stencil.compute_all_derivatives(
            rho, x0, y0, z0, self.center
        )

        # Get metric quantities at this point
        gamma_inv = self.metric.gamma_inv(x0, y0, z0)
        dgamma = self.metric.dgamma(x0, y0, z0)
        K_tensor = self.metric.extrinsic_curvature(x0, y0, z0)
        K_trace = self.metric.K_trace(x0, y0, z0)

        return compute_expansion(
            grad_phi, hess_phi,
            gamma_inv, dgamma,
            K_tensor, K_trace
        )

    def evaluate(self, rho: np.ndarray) -> np.ndarray:
        """
        Evaluate F[ρ] at all independent grid points.

        Args:
            rho: Full (N_s, N_s) grid of radial values

        Returns:
            Array of length n_independent with residual values
        """
        indices = self.mesh.independent_indices()
        residual = np.zeros(len(indices))

        for k, (i_theta, i_phi) in enumerate(indices):
            residual[k] = self.evaluate_at_point(rho, i_theta, i_phi)

        return residual

    def residual_norm(self, rho: np.ndarray) -> float:
        """
        Compute L2 norm of the residual.

        Args:
            rho: Full (N_s, N_s) grid of radial values

        Returns:
            L2 norm of F[ρ]
        """
        F = self.evaluate(rho)
        return np.linalg.norm(F)


def create_residual_evaluator(
    mesh: SurfaceMesh,
    interpolator: BiquarticInterpolator,
    metric: Metric,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_factor: float = 0.5
) -> ResidualEvaluator:
    """
    Create a residual evaluator with all necessary components.

    Args:
        mesh: SurfaceMesh instance
        interpolator: BiquarticInterpolator instance
        metric: Metric providing geometric data
        center: Center of coordinate system
        spacing_factor: Stencil spacing factor

    Returns:
        ResidualEvaluator instance
    """
    stencil = CartesianStencil(mesh, interpolator, spacing_factor)
    return ResidualEvaluator(mesh, stencil, metric, center)
