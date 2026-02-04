"""
27-point Cartesian stencil for computing derivatives.

At each surface point (x₀, y₀, z₀), constructs a 3×3×3 cube of points
and uses finite differences to compute ∂φ/∂xⁱ and ∂²φ/∂xⁱ∂xʲ where
φ = r - ρ(θ, φ).

This approach avoids coordinate singularities at the poles by working
in Cartesian coordinates.

Reference: Huq, Choptuik & Matzner (2000), Section II.C
"""

import numpy as np
from typing import Tuple
from .surface import SurfaceMesh
from .interpolation import BiquarticInterpolator


class CartesianStencil:
    """
    Computes derivatives of φ = r - ρ(θ, φ) using a 27-point Cartesian stencil.

    At each surface point, creates a 3×3×3 cube of sample points with spacing h,
    evaluates φ at each point via interpolation, and computes derivatives using
    standard finite difference operators.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        interpolator: BiquarticInterpolator,
        spacing_factor: float = 0.5
    ):
        """
        Initialize Cartesian stencil.

        Args:
            mesh: SurfaceMesh instance
            interpolator: BiquarticInterpolator for evaluating ρ off-grid
            spacing_factor: Factor c such that h = c × d_theta
        """
        self.mesh = mesh
        self.interpolator = interpolator
        self.spacing_factor = spacing_factor
        self.h = spacing_factor * mesh.d_theta

        # Stencil offsets: -1, 0, 1 in each direction
        self._offsets = np.array([-1, 0, 1])

        # Precompute offset combinations for vectorized computation
        # Shape: (27, 3) - each row is [di, dj, dk]
        di, dj, dk = np.meshgrid(self._offsets, self._offsets, self._offsets, indexing='ij')
        self._offset_vectors = np.stack([di.ravel(), dj.ravel(), dk.ravel()], axis=1)

    def compute_phi_stencil(
        self,
        rho: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Evaluate φ = r - ρ on the 27-point stencil around (x0, y0, z0).

        Vectorized implementation for better performance.

        Args:
            rho: Grid values of ρ(θ, φ)
            x0, y0, z0: Center point of stencil
            center: Origin for spherical coordinates

        Returns:
            Array of shape (3, 3, 3) with φ values
        """
        h = self.h
        cx, cy, cz = center

        # Compute all 27 stencil points at once
        # offsets shape: (27, 3), each row is [di, dj, dk]
        stencil_points = np.array([x0, y0, z0]) + h * self._offset_vectors

        # Extract x, y, z coordinates
        x_pts = stencil_points[:, 0]
        y_pts = stencil_points[:, 1]
        z_pts = stencil_points[:, 2]

        # Convert to spherical coordinates (vectorized)
        dx = x_pts - cx
        dy = y_pts - cy
        dz = z_pts - cz

        r_pts = np.sqrt(dx**2 + dy**2 + dz**2)

        # Handle points at origin
        r_safe = np.maximum(r_pts, 1e-14)

        theta_pts = np.arccos(np.clip(dz / r_safe, -1, 1))
        phi_pts = np.arctan2(dy, dx)
        phi_pts = np.where(phi_pts < 0, phi_pts + 2*np.pi, phi_pts)

        # Interpolate ρ at all points using batch method
        rho_interp = self.interpolator.interpolate_batch(rho, theta_pts, phi_pts)

        # φ = r - ρ
        phi_values = r_pts - rho_interp

        return phi_values.reshape(3, 3, 3)

    def compute_derivatives(
        self,
        phi_stencil: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute first and second derivatives of φ from stencil values.

        Uses second-order accurate central differences.

        Args:
            phi_stencil: Array of shape (3, 3, 3) with φ values

        Returns:
            Tuple of:
                - grad_phi: Array of shape (3,) with [∂φ/∂x, ∂φ/∂y, ∂φ/∂z]
                - hess_phi: Array of shape (3, 3) with ∂²φ/∂xⁱ∂xʲ
        """
        h = self.h
        h2 = h * h

        # Central point value
        phi_0 = phi_stencil[1, 1, 1]

        # First derivatives: (f(x+h) - f(x-h)) / (2h)
        grad_phi = np.array([
            (phi_stencil[2, 1, 1] - phi_stencil[0, 1, 1]) / (2 * h),  # ∂φ/∂x
            (phi_stencil[1, 2, 1] - phi_stencil[1, 0, 1]) / (2 * h),  # ∂φ/∂y
            (phi_stencil[1, 1, 2] - phi_stencil[1, 1, 0]) / (2 * h),  # ∂φ/∂z
        ])

        # Second derivatives
        hess_phi = np.zeros((3, 3))

        # Diagonal: (f(x+h) - 2f(x) + f(x-h)) / h²
        hess_phi[0, 0] = (phi_stencil[2, 1, 1] - 2 * phi_0 + phi_stencil[0, 1, 1]) / h2
        hess_phi[1, 1] = (phi_stencil[1, 2, 1] - 2 * phi_0 + phi_stencil[1, 0, 1]) / h2
        hess_phi[2, 2] = (phi_stencil[1, 1, 2] - 2 * phi_0 + phi_stencil[1, 1, 0]) / h2

        # Off-diagonal: (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4h²)
        hess_phi[0, 1] = (
            phi_stencil[2, 2, 1] - phi_stencil[2, 0, 1]
            - phi_stencil[0, 2, 1] + phi_stencil[0, 0, 1]
        ) / (4 * h2)
        hess_phi[1, 0] = hess_phi[0, 1]

        hess_phi[0, 2] = (
            phi_stencil[2, 1, 2] - phi_stencil[2, 1, 0]
            - phi_stencil[0, 1, 2] + phi_stencil[0, 1, 0]
        ) / (4 * h2)
        hess_phi[2, 0] = hess_phi[0, 2]

        hess_phi[1, 2] = (
            phi_stencil[1, 2, 2] - phi_stencil[1, 2, 0]
            - phi_stencil[1, 0, 2] + phi_stencil[1, 0, 0]
        ) / (4 * h2)
        hess_phi[2, 1] = hess_phi[1, 2]

        return grad_phi, hess_phi

    def compute_all_derivatives(
        self,
        rho: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute first and second derivatives of φ at a surface point.

        Args:
            rho: Grid values of ρ(θ, φ)
            x0, y0, z0: Surface point coordinates
            center: Origin for spherical coordinates

        Returns:
            Tuple of (grad_phi, hess_phi)
        """
        phi_stencil = self.compute_phi_stencil(rho, x0, y0, z0, center)
        return self.compute_derivatives(phi_stencil)


class DerivativeCache:
    """
    Caches derivative computations for efficiency during Jacobian evaluation.
    """

    def __init__(self, mesh: SurfaceMesh, stencil: CartesianStencil):
        """
        Initialize cache.

        Args:
            mesh: SurfaceMesh instance
            stencil: CartesianStencil instance
        """
        self.mesh = mesh
        self.stencil = stencil
        self._cache = {}

    def clear(self):
        """Clear all cached values."""
        self._cache = {}

    def get_derivatives(
        self,
        rho: np.ndarray,
        i_theta: int,
        i_phi: int,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get derivatives at a grid point, using cache if available.

        Args:
            rho: Grid values (used to check cache validity)
            i_theta, i_phi: Grid point indices
            center: Coordinate origin

        Returns:
            Tuple of (grad_phi, hess_phi)
        """
        key = (i_theta, i_phi)

        if key not in self._cache:
            # Compute surface point coordinates
            theta = self.mesh.theta[i_theta]
            phi_coord = self.mesh.phi[i_phi]
            r = rho[i_theta, i_phi]

            x0 = center[0] + r * np.sin(theta) * np.cos(phi_coord)
            y0 = center[1] + r * np.sin(theta) * np.sin(phi_coord)
            z0 = center[2] + r * np.cos(theta)

            grad_phi, hess_phi = self.stencil.compute_all_derivatives(
                rho, x0, y0, z0, center
            )
            self._cache[key] = (grad_phi.copy(), hess_phi.copy())

        return self._cache[key]


def create_stencil(
    mesh: SurfaceMesh,
    interpolator: BiquarticInterpolator,
    spacing_factor: float = 0.5
) -> CartesianStencil:
    """
    Create a Cartesian stencil for derivative computations.

    Args:
        mesh: SurfaceMesh instance
        interpolator: BiquarticInterpolator instance
        spacing_factor: Factor c such that h = c × d_theta

    Returns:
        CartesianStencil instance
    """
    return CartesianStencil(mesh, interpolator, spacing_factor)
