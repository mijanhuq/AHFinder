"""
Surface regularization for Level Flow evolution.

Provides smoothing utilities to prevent high-frequency oscillations
during Level Flow evolution. Uses simple neighbor-averaging which
is stable and effective.

Note: The expansion Θ is computed by the existing residual evaluator
infrastructure - this module only provides optional smoothing.
"""

import numpy as np
from numba import jit
from typing import Tuple

from ..surface import SurfaceMesh


@jit(nopython=True, cache=True)
def smooth_surface_average(rho: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply smoothing via averaging with neighbors.

    This is a simple, stable smoothing method that blends each
    point with its 4 neighbors (north, south, east, west).

    Args:
        rho: Surface shape (N_s, N_s)
        alpha: Smoothing strength (0 = no smoothing, 1 = full averaging)

    Returns:
        Smoothed surface (N_s, N_s)
    """
    N_s = rho.shape[0]
    rho_smooth = rho.copy()

    # Interior points
    for i in range(1, N_s - 1):
        for j in range(N_s):
            jp = (j + 1) % N_s
            jm = (j - 1) % N_s

            # 4-neighbor average
            avg = 0.25 * (rho[i - 1, j] + rho[i + 1, j] + rho[i, jp] + rho[i, jm])
            rho_smooth[i, j] = (1 - alpha) * rho[i, j] + alpha * avg

    # Poles: average over the entire ring
    north_avg = 0.0
    south_avg = 0.0
    for j in range(N_s):
        north_avg += rho[1, j]
        south_avg += rho[-2, j]
    north_avg /= N_s
    south_avg /= N_s

    # Blend pole values with neighboring ring
    rho_smooth[0, :] = (1 - alpha) * rho[0, 0] + alpha * north_avg
    rho_smooth[-1, :] = (1 - alpha) * rho[-1, 0] + alpha * south_avg

    return rho_smooth


@jit(nopython=True, cache=True)
def smooth_surface_gaussian(rho: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian-weighted smoothing.

    Uses a 5x5 stencil with Gaussian weights for smoother results
    than simple averaging.

    Args:
        rho: Surface shape (N_s, N_s)
        sigma: Gaussian width in grid units

    Returns:
        Smoothed surface (N_s, N_s)
    """
    N_s = rho.shape[0]
    rho_smooth = np.zeros((N_s, N_s))

    # Precompute Gaussian weights for 5x5 stencil
    weights = np.zeros((5, 5))
    total = 0.0
    for di in range(-2, 3):
        for dj in range(-2, 3):
            w = np.exp(-(di*di + dj*dj) / (2 * sigma * sigma))
            weights[di + 2, dj + 2] = w
            total += w

    # Normalize
    for i in range(5):
        for j in range(5):
            weights[i, j] /= total

    # Apply to interior points
    for i in range(2, N_s - 2):
        for j in range(N_s):
            val = 0.0
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    jj = (j + dj) % N_s
                    val += weights[di + 2, dj + 2] * rho[i + di, jj]
            rho_smooth[i, j] = val

    # Near-pole regions: use simple averaging
    for i in [1, N_s - 2]:
        for j in range(N_s):
            jp = (j + 1) % N_s
            jm = (j - 1) % N_s
            rho_smooth[i, j] = 0.5 * rho[i, j] + 0.125 * (
                rho[i - 1, j] + rho[i + 1, j] + rho[i, jp] + rho[i, jm]
            )

    # Poles
    north_avg = 0.0
    south_avg = 0.0
    for j in range(N_s):
        north_avg += rho[1, j]
        south_avg += rho[-2, j]

    rho_smooth[0, :] = 0.5 * rho[0, 0] + 0.5 * north_avg / N_s
    rho_smooth[-1, :] = 0.5 * rho[-1, 0] + 0.5 * south_avg / N_s

    return rho_smooth


def regularized_velocity(
    theta: np.ndarray,
    max_velocity: float = 1.0
) -> np.ndarray:
    """
    Compute regularized flow velocity to prevent instability.

    Uses: v = -Θ / (1 + |Θ|/v_max) to bound velocity magnitude.

    Args:
        theta: Expansion field Θ (N_s, N_s)
        max_velocity: Maximum allowed velocity magnitude

    Returns:
        Flow velocity field (N_s, N_s)
    """
    return -theta / (1.0 + np.abs(theta) / max_velocity)


class SurfaceSmoother:
    """
    Applies smoothing to Level Flow surfaces.

    Provides methods to smooth surfaces during evolution to prevent
    high-frequency oscillations without affecting convergence.

    Args:
        mesh: SurfaceMesh instance
        method: Smoothing method ('average' or 'gaussian')
        strength: Smoothing strength (0 to 1)
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        method: str = 'average',
        strength: float = 0.1
    ):
        self.mesh = mesh
        self.method = method
        self.strength = strength
        self.N_s = mesh.N_s

    def smooth(self, rho: np.ndarray, strength: float = None) -> np.ndarray:
        """
        Apply smoothing to surface.

        Args:
            rho: Surface shape (N_s, N_s)
            strength: Override default strength

        Returns:
            Smoothed surface
        """
        alpha = strength if strength is not None else self.strength

        if self.method == 'gaussian':
            # For Gaussian, strength controls sigma
            return smooth_surface_gaussian(rho, sigma=alpha * 2)
        else:
            return smooth_surface_average(rho, alpha)

    def apply_iterative(
        self,
        rho: np.ndarray,
        n_iterations: int = 3,
        strength: float = None
    ) -> np.ndarray:
        """
        Apply multiple smoothing iterations.

        Multiple weak smoothing passes can be more effective than
        one strong pass while preserving features better.

        Args:
            rho: Surface shape
            n_iterations: Number of smoothing passes
            strength: Strength per iteration

        Returns:
            Smoothed surface
        """
        alpha = strength if strength is not None else self.strength / n_iterations
        result = rho.copy()

        for _ in range(n_iterations):
            result = self.smooth(result, alpha)

        return result


# Keep these names for backwards compatibility with __init__.py
MeanCurvatureRegularizer = SurfaceSmoother


def compute_laplacian_spherical(mesh: SurfaceMesh, rho: np.ndarray) -> np.ndarray:
    """
    Placeholder for backwards compatibility.

    Note: Direct Laplacian computation on the sphere has numerical issues
    near the poles. Use SurfaceSmoother.smooth() instead for regularization.
    """
    import warnings
    warnings.warn(
        "compute_laplacian_spherical has numerical issues near poles. "
        "Use SurfaceSmoother.smooth() for regularization instead.",
        DeprecationWarning
    )
    # Return zeros - the Laplacian approach is deprecated
    return np.zeros_like(rho)


def estimate_optimal_epsilon(mesh: SurfaceMesh, rho: np.ndarray) -> float:
    """
    Estimate optimal smoothing strength based on surface properties.

    Args:
        mesh: SurfaceMesh instance
        rho: Current surface shape

    Returns:
        Suggested smoothing strength (0 to 1)
    """
    # Scale by surface variation
    rho_range = np.max(rho) - np.min(rho)
    rho_mean = np.mean(rho)

    # Relative variation
    rel_variation = rho_range / (rho_mean + 1e-10)

    # More smoothing for rough surfaces, less for smooth
    # Typical range: 0.05 to 0.2
    strength = 0.05 + 0.15 * min(rel_variation, 1.0)

    return strength
