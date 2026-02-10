"""
Vectorized residual evaluation for the apparent horizon equation.

Processes all grid points in batches instead of point-by-point,
enabling efficient use of vectorized interpolation and metric calls.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple
from .surface import SurfaceMesh
from .interpolation_lagrange import LagrangeInterpolator, interpolate_batch_lagrange
from .metrics.base import Metric


@jit(nopython=True, cache=True)
def compute_all_coords(
    theta: np.ndarray,
    phi: np.ndarray,
    rho_flat: np.ndarray,
    center: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Cartesian coordinates for all grid points.

    Args:
        theta: 1D array of theta values for each point
        phi: 1D array of phi values for each point
        rho_flat: 1D array of radial values for each point
        center: (cx, cy, cz) center of coordinates

    Returns:
        x, y, z: 1D arrays of Cartesian coordinates
    """
    cx, cy, cz = center
    n = len(theta)

    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)

    for i in range(n):
        sin_th = np.sin(theta[i])
        cos_th = np.cos(theta[i])
        sin_ph = np.sin(phi[i])
        cos_ph = np.cos(phi[i])
        r = rho_flat[i]

        x[i] = cx + r * sin_th * cos_ph
        y[i] = cy + r * sin_th * sin_ph
        z[i] = cz + r * cos_th

    return x, y, z


@jit(nopython=True, cache=True)
def compute_all_stencil_points(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    h: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 27 stencil points for each grid point.

    Args:
        x0, y0, z0: 1D arrays of center points (n_pts,)
        h: Stencil spacing

    Returns:
        x_stencil, y_stencil, z_stencil: 2D arrays of shape (n_pts, 27)
    """
    n_pts = len(x0)

    x_stencil = np.empty((n_pts, 27))
    y_stencil = np.empty((n_pts, 27))
    z_stencil = np.empty((n_pts, 27))

    offsets = np.array([-1.0, 0.0, 1.0])

    idx = 0
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                ox = h * offsets[di]
                oy = h * offsets[dj]
                oz = h * offsets[dk]

                for p in range(n_pts):
                    x_stencil[p, idx] = x0[p] + ox
                    y_stencil[p, idx] = y0[p] + oy
                    z_stencil[p, idx] = z0[p] + oz

                idx += 1

    return x_stencil, y_stencil, z_stencil


@jit(nopython=True, cache=True)
def cartesian_to_spherical_batch(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cx: float,
    cy: float,
    cz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to spherical coordinates.

    Args:
        x, y, z: 1D arrays of Cartesian coordinates
        cx, cy, cz: Center point

    Returns:
        r, theta, phi: 1D arrays of spherical coordinates
    """
    n = len(x)
    r = np.empty(n)
    theta = np.empty(n)
    phi = np.empty(n)

    for i in range(n):
        dx = x[i] - cx
        dy = y[i] - cy
        dz = z[i] - cz

        r_val = np.sqrt(dx*dx + dy*dy + dz*dz)
        r[i] = r_val

        if r_val < 1e-14:
            theta[i] = 0.0
            phi[i] = 0.0
        else:
            cos_th = dz / r_val
            if cos_th > 1.0:
                cos_th = 1.0
            elif cos_th < -1.0:
                cos_th = -1.0
            theta[i] = np.arccos(cos_th)

            phi_val = np.arctan2(dy, dx)
            if phi_val < 0:
                phi_val += 2 * np.pi
            phi[i] = phi_val

    return r, theta, phi


@jit(nopython=True, cache=True)
def compute_derivatives_batch(
    phi_values: np.ndarray,
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute first and second derivatives from stencil values for all points.

    Args:
        phi_values: Array of shape (n_pts, 27) with φ values at stencil points
        h: Stencil spacing

    Returns:
        grad_phi: Array of shape (n_pts, 3) with first derivatives
        hess_phi: Array of shape (n_pts, 3, 3) with second derivatives
    """
    n_pts = phi_values.shape[0]
    h2 = h * h

    grad_phi = np.empty((n_pts, 3))
    hess_phi = np.empty((n_pts, 3, 3))

    # Stencil index mapping: idx = di*9 + dj*3 + dk where di,dj,dk in {0,1,2}
    # di=0 means offset=-1, di=1 means offset=0, di=2 means offset=+1

    for p in range(n_pts):
        # Central point: (1,1,1) -> idx = 1*9 + 1*3 + 1 = 13
        phi_0 = phi_values[p, 13]

        # First derivatives: (f(x+h) - f(x-h)) / (2h)
        # ∂φ/∂x: (2,1,1) - (0,1,1) -> idx 22 - idx 4
        grad_phi[p, 0] = (phi_values[p, 22] - phi_values[p, 4]) / (2 * h)
        # ∂φ/∂y: (1,2,1) - (1,0,1) -> idx 16 - idx 10
        grad_phi[p, 1] = (phi_values[p, 16] - phi_values[p, 10]) / (2 * h)
        # ∂φ/∂z: (1,1,2) - (1,1,0) -> idx 14 - idx 12
        grad_phi[p, 2] = (phi_values[p, 14] - phi_values[p, 12]) / (2 * h)

        # Second derivatives (diagonal): (f(x+h) - 2f(x) + f(x-h)) / h²
        hess_phi[p, 0, 0] = (phi_values[p, 22] - 2*phi_0 + phi_values[p, 4]) / h2
        hess_phi[p, 1, 1] = (phi_values[p, 16] - 2*phi_0 + phi_values[p, 10]) / h2
        hess_phi[p, 2, 2] = (phi_values[p, 14] - 2*phi_0 + phi_values[p, 12]) / h2

        # Off-diagonal: (f(++)-f(+-)-f(-+)+f(--)) / (4h²)
        # ∂²φ/∂x∂y: (2,2,1)-(2,0,1)-(0,2,1)+(0,0,1) -> 25-19-7+1
        hess_phi[p, 0, 1] = (phi_values[p, 25] - phi_values[p, 19]
                           - phi_values[p, 7] + phi_values[p, 1]) / (4 * h2)
        hess_phi[p, 1, 0] = hess_phi[p, 0, 1]

        # ∂²φ/∂x∂z: (2,1,2)-(2,1,0)-(0,1,2)+(0,1,0) -> 23-21-5+3
        hess_phi[p, 0, 2] = (phi_values[p, 23] - phi_values[p, 21]
                           - phi_values[p, 5] + phi_values[p, 3]) / (4 * h2)
        hess_phi[p, 2, 0] = hess_phi[p, 0, 2]

        # ∂²φ/∂y∂z: (1,2,2)-(1,2,0)-(1,0,2)+(1,0,0) -> 17-15-11+9
        hess_phi[p, 1, 2] = (phi_values[p, 17] - phi_values[p, 15]
                           - phi_values[p, 11] + phi_values[p, 9]) / (4 * h2)
        hess_phi[p, 2, 1] = hess_phi[p, 1, 2]

    return grad_phi, hess_phi


@jit(nopython=True, cache=True, parallel=True)
def compute_expansion_batch(
    grad_phi: np.ndarray,
    hess_phi: np.ndarray,
    gamma_inv: np.ndarray,
    dgamma: np.ndarray,
    K_tensor: np.ndarray,
    K_trace: np.ndarray
) -> np.ndarray:
    """
    Compute expansion Θ for all points in batch.

    Args:
        grad_phi: Shape (n_pts, 3)
        hess_phi: Shape (n_pts, 3, 3)
        gamma_inv: Shape (n_pts, 3, 3)
        dgamma: Shape (n_pts, 3, 3, 3)
        K_tensor: Shape (n_pts, 3, 3)
        K_trace: Shape (n_pts,)

    Returns:
        theta: Shape (n_pts,) - expansion at each point
    """
    n_pts = grad_phi.shape[0]
    result = np.empty(n_pts)

    for p in prange(n_pts):
        # ω = γ^{ij} ∂_i φ ∂_j φ
        omega = 0.0
        for i in range(3):
            for j in range(3):
                omega += gamma_inv[p, i, j] * grad_phi[p, i] * grad_phi[p, j]

        if omega < 1e-20:
            result[p] = 0.0
            continue

        sqrt_omega = np.sqrt(omega)

        # n^i = γ^{ij} ∂_j φ
        n_up = np.zeros(3)
        for i in range(3):
            for j in range(3):
                n_up[i] += gamma_inv[p, i, j] * grad_phi[p, j]

        # s^i = n^i / √ω
        s_up = np.zeros(3)
        for i in range(3):
            s_up[i] = n_up[i] / sqrt_omega

        # Christoffel symbols: Γ^k_{ij} = (1/2) γ^{kl} (∂_i γ_{lj} + ∂_j γ_{il} - ∂_l γ_{ij})
        chris = np.zeros((3, 3, 3))
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        # ∂_i γ_{lj} + ∂_j γ_{il} - ∂_l γ_{ij}
                        bracket_l = dgamma[p, i, l, j] + dgamma[p, j, i, l] - dgamma[p, l, i, j]
                        chris[k, i, j] += 0.5 * gamma_inv[p, k, l] * bracket_l

        # Contracted Christoffel: Γ^k = γ^{ij} Γ^k_{ij}
        Gamma_up = np.zeros(3)
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    Gamma_up[k] += gamma_inv[p, i, j] * chris[k, i, j]

        # Coordinate Laplacian: γ^{ij} ∂_i ∂_j φ
        coord_laplacian = 0.0
        for i in range(3):
            for j in range(3):
                coord_laplacian += gamma_inv[p, i, j] * hess_phi[p, i, j]

        # Covariant Laplacian: Δφ = coord_laplacian - Γ^k ∂_k φ
        laplacian = coord_laplacian
        for k in range(3):
            laplacian -= Gamma_up[k] * grad_phi[p, k]

        # Projection term: (n^i n^j / ω) ∂_i ∂_j φ
        coord_proj = 0.0
        for i in range(3):
            for j in range(3):
                coord_proj += n_up[i] * n_up[j] * hess_phi[p, i, j]
        coord_proj /= omega

        # Christoffel correction: (n^i n^j / ω) Γ^k_{ij} ∂_k φ
        chris_proj = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    chris_proj += n_up[i] * n_up[j] * chris[k, i, j] * grad_phi[p, k]
        chris_proj /= omega

        proj_term = coord_proj - chris_proj

        # Divergence: D_i s^i = (Δφ - proj_term) / √ω
        div_s = (laplacian - proj_term) / sqrt_omega

        # Extrinsic curvature term: K_{ij} s^i s^j
        K_ss = 0.0
        for i in range(3):
            for j in range(3):
                K_ss += K_tensor[p, i, j] * s_up[i] * s_up[j]

        # Expansion: Θ = D_i s^i + K_{ij} s^i s^j - K
        result[p] = div_s + K_ss - K_trace[p]

    return result


class VectorizedResidualEvaluator:
    """
    Vectorized residual evaluator that processes all grid points in batches.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        interpolator: LagrangeInterpolator,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing_factor: float = 0.5
    ):
        self.mesh = mesh
        self.interpolator = interpolator
        self.metric = metric
        self.center = center
        self.h = spacing_factor * mesh.d_theta

        # Precompute grid point info
        indices = mesh.independent_indices()
        self.n_pts = len(indices)
        self._indices = indices

        self._theta_grid = np.array([mesh.theta[i] for i, _ in indices])
        self._phi_grid = np.array([mesh.phi[j] for _, j in indices])

    def evaluate(self, rho: np.ndarray) -> np.ndarray:
        """
        Evaluate F[ρ] at all independent grid points using vectorized operations.
        """
        cx, cy, cz = self.center
        h = self.h

        # Extract rho values at independent points
        rho_flat = np.array([rho[i, j] for i, j in self._indices])

        # 1. Compute all surface point coordinates
        x0, y0, z0 = compute_all_coords(
            self._theta_grid, self._phi_grid, rho_flat, self.center
        )

        # 2. Compute all stencil points (n_pts, 27)
        x_stencil, y_stencil, z_stencil = compute_all_stencil_points(x0, y0, z0, h)

        # Flatten for batch processing
        x_flat = x_stencil.ravel()
        y_flat = y_stencil.ravel()
        z_flat = z_stencil.ravel()

        # 3. Convert all stencil points to spherical coordinates
        r_sph, theta_sph, phi_sph = cartesian_to_spherical_batch(
            x_flat, y_flat, z_flat, cx, cy, cz
        )

        # 4. Batch interpolation - single call for all stencil points
        rho_interp = self.interpolator.interpolate_batch(rho, theta_sph, phi_sph)

        # 5. Compute φ = r - ρ
        phi_flat = r_sph - rho_interp
        phi_values = phi_flat.reshape(self.n_pts, 27)

        # 6. Compute derivatives for all points
        grad_phi, hess_phi = compute_derivatives_batch(phi_values, h)

        # 7. Get metric quantities at all surface points
        gamma_inv = np.empty((self.n_pts, 3, 3))
        dgamma = np.empty((self.n_pts, 3, 3, 3))
        K_tensor = np.empty((self.n_pts, 3, 3))
        K_trace = np.empty(self.n_pts)

        for p in range(self.n_pts):
            gamma_inv[p] = self.metric.gamma_inv(x0[p], y0[p], z0[p])
            dgamma[p] = self.metric.dgamma(x0[p], y0[p], z0[p])
            K_tensor[p] = self.metric.extrinsic_curvature(x0[p], y0[p], z0[p])
            K_trace[p] = self.metric.K_trace(x0[p], y0[p], z0[p])

        # 8. Compute expansion for all points
        theta_expansion = compute_expansion_batch(
            grad_phi, hess_phi, gamma_inv, dgamma, K_tensor, K_trace
        )

        return theta_expansion


def create_vectorized_residual_evaluator(
    mesh: SurfaceMesh,
    metric: Metric,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_factor: float = 0.5
) -> VectorizedResidualEvaluator:
    """Create a vectorized residual evaluator."""
    interpolator = LagrangeInterpolator(mesh)
    return VectorizedResidualEvaluator(
        mesh, interpolator, metric, center, spacing_factor
    )
