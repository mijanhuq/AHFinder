"""
Surface mesh management for apparent horizon parameterization.

The surface is parameterized as r = ρ(θ, φ) where (θ, φ) are spherical
coordinates on an N_s × N_s grid covering [0, π] × [0, 2π).

Pole handling:
- θ = 0 (north pole): All φ points collapse to single point
- θ = π (south pole): All φ points collapse to single point
- Total independent points: N_s² - 2N_s + 2

Reference: Huq, Choptuik & Matzner (2000), Section II.A
"""

import numpy as np
from typing import Tuple, Optional


class SurfaceMesh:
    """
    Manages the (θ, φ) mesh for horizon surface parameterization.

    The mesh is structured as an N_s × N_s grid where:
    - θ runs from 0 to π (N_s points, including poles)
    - φ runs from 0 to 2π (N_s points, periodic)

    Attributes:
        N_s: Number of grid points in each direction
        theta: 1D array of θ values
        phi: 1D array of φ values
        d_theta: Grid spacing in θ
        d_phi: Grid spacing in φ
    """

    def __init__(self, N_s: int = 33):
        """
        Initialize surface mesh.

        Args:
            N_s: Number of grid points in each direction (should be odd for symmetry)
        """
        if N_s < 5:
            raise ValueError("N_s must be at least 5 for meaningful resolution")

        self.N_s = N_s

        # θ grid: [0, π] with N_s points including endpoints
        self.theta = np.linspace(0, np.pi, N_s)
        self.d_theta = np.pi / (N_s - 1)

        # φ grid: [0, 2π) with N_s points (periodic, exclude 2π)
        self.phi = np.linspace(0, 2 * np.pi, N_s, endpoint=False)
        self.d_phi = 2 * np.pi / N_s

        # Create 2D meshgrids
        self._theta_grid, self._phi_grid = np.meshgrid(
            self.theta, self.phi, indexing='ij'
        )

        # Compute independent point count
        # Total N_s × N_s, minus (N_s - 1) redundant at each pole
        self._n_independent = N_s * N_s - 2 * (N_s - 1)

    @property
    def n_independent(self) -> int:
        """Number of independent grid points (excluding pole redundancies)."""
        return self._n_independent

    def theta_phi_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return 2D meshgrids of θ and φ values.

        Returns:
            Tuple of (theta_grid, phi_grid), each with shape (N_s, N_s)
            Index convention: [i_theta, i_phi]
        """
        return self._theta_grid.copy(), self._phi_grid.copy()

    def xyz_from_rho(
        self,
        rho: np.ndarray,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert ρ(θ, φ) to Cartesian coordinates.

        Args:
            rho: Array of shape (N_s, N_s) with radial values
            center: Center point (x0, y0, z0) of the coordinate system

        Returns:
            Tuple of (x, y, z) arrays, each with shape (N_s, N_s)
        """
        if rho.shape != (self.N_s, self.N_s):
            raise ValueError(f"rho must have shape ({self.N_s}, {self.N_s})")

        x0, y0, z0 = center

        sin_theta = np.sin(self._theta_grid)
        cos_theta = np.cos(self._theta_grid)
        sin_phi = np.sin(self._phi_grid)
        cos_phi = np.cos(self._phi_grid)

        x = x0 + rho * sin_theta * cos_phi
        y = y0 + rho * sin_theta * sin_phi
        z = z0 + rho * cos_theta

        return x, y, z

    def rho_from_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Compute radial distance from Cartesian coordinates.

        Args:
            x, y, z: Cartesian coordinates (any shape)
            center: Center point

        Returns:
            Array of radial distances with same shape as input
        """
        x0, y0, z0 = center
        return np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    def independent_indices(self) -> np.ndarray:
        """
        Return indices of independent grid points.

        At the poles (θ = 0 and θ = π), all φ values correspond to the same
        physical point, so we only include φ = 0 for these rows.

        Returns:
            Array of shape (n_independent, 2) with (i_theta, i_phi) indices
        """
        indices = []

        for i_theta in range(self.N_s):
            if i_theta == 0 or i_theta == self.N_s - 1:
                # Pole: only include φ = 0
                indices.append((i_theta, 0))
            else:
                # Non-pole: include all φ values
                for i_phi in range(self.N_s):
                    indices.append((i_theta, i_phi))

        return np.array(indices)

    def flat_to_grid(self, flat_values: np.ndarray) -> np.ndarray:
        """
        Convert flat array of independent values to full (N_s, N_s) grid.

        Args:
            flat_values: Array of length n_independent

        Returns:
            Array of shape (N_s, N_s) with pole values replicated
        """
        if len(flat_values) != self._n_independent:
            raise ValueError(
                f"Expected {self._n_independent} values, got {len(flat_values)}"
            )

        grid = np.zeros((self.N_s, self.N_s))
        indices = self.independent_indices()

        for k, (i_theta, i_phi) in enumerate(indices):
            grid[i_theta, i_phi] = flat_values[k]

        # Replicate pole values to all φ
        grid[0, :] = grid[0, 0]  # North pole
        grid[-1, :] = grid[-1, 0]  # South pole

        return grid

    def grid_to_flat(self, grid: np.ndarray) -> np.ndarray:
        """
        Extract independent values from full grid.

        Args:
            grid: Array of shape (N_s, N_s)

        Returns:
            Array of length n_independent
        """
        if grid.shape != (self.N_s, self.N_s):
            raise ValueError(f"Grid must have shape ({self.N_s}, {self.N_s})")

        indices = self.independent_indices()
        return np.array([grid[i, j] for i, j in indices])

    def neighbor_indices(
        self,
        i_theta: int,
        i_phi: int,
        half_width: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices of neighboring points for stencil operations.

        Handles periodicity in φ and boundary conditions at poles.

        Args:
            i_theta, i_phi: Center point indices
            half_width: Half-width of stencil (total width = 2*half_width + 1)

        Returns:
            Tuple of (theta_indices, phi_indices) arrays
        """
        # θ indices with boundary clamping
        i_th = np.arange(
            max(0, i_theta - half_width),
            min(self.N_s, i_theta + half_width + 1)
        )

        # φ indices with periodic wrapping
        i_ph = np.arange(i_phi - half_width, i_phi + half_width + 1) % self.N_s

        return i_th, i_ph

    def stencil_for_point(
        self,
        i_theta: int,
        i_phi: int,
        size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices for biquartic interpolation stencil.

        Returns a size × size stencil of indices centered on the given point,
        with special handling near poles.

        Args:
            i_theta, i_phi: Center point indices
            size: Stencil size (4 for biquartic)

        Returns:
            Tuple of (theta_indices, phi_indices) arrays, each of length 'size'
        """
        half = size // 2

        # θ stencil: try to center, but shift if near boundary
        i_th_start = i_theta - half + 1
        if i_th_start < 0:
            i_th_start = 0
        elif i_th_start + size > self.N_s:
            i_th_start = self.N_s - size
        i_th = np.arange(i_th_start, i_th_start + size)

        # φ stencil: periodic
        i_ph_start = i_phi - half + 1
        i_ph = np.arange(i_ph_start, i_ph_start + size) % self.N_s

        return i_th, i_ph

    def surface_area_element(
        self,
        rho: np.ndarray,
        i_theta: int,
        i_phi: int
    ) -> float:
        """
        Compute surface area element at a grid point.

        For a surface r = ρ(θ, φ), the area element is:
        dA = √(ρ² + (∂ρ/∂θ)²) × √(ρ² sin²θ + (∂ρ/∂φ)²) × dθ dφ

        (Simplified for nearly spherical surfaces)

        Args:
            rho: Full (N_s, N_s) grid of radial values
            i_theta, i_phi: Grid point indices

        Returns:
            Area element value
        """
        theta = self.theta[i_theta]
        r = rho[i_theta, i_phi]

        # Compute derivatives using central differences
        if 0 < i_theta < self.N_s - 1:
            drho_dtheta = (rho[i_theta + 1, i_phi] - rho[i_theta - 1, i_phi]) / (2 * self.d_theta)
        else:
            drho_dtheta = 0.0

        i_phi_p = (i_phi + 1) % self.N_s
        i_phi_m = (i_phi - 1) % self.N_s
        drho_dphi = (rho[i_theta, i_phi_p] - rho[i_theta, i_phi_m]) / (2 * self.d_phi)

        sin_theta = np.sin(theta)

        # Metric on the surface
        g_theta_theta = r**2 + drho_dtheta**2
        g_phi_phi = r**2 * sin_theta**2 + drho_dphi**2

        # Area element (ignoring off-diagonal terms for simplicity)
        return np.sqrt(g_theta_theta * g_phi_phi) * self.d_theta * self.d_phi


def create_sphere(mesh: SurfaceMesh, radius: float) -> np.ndarray:
    """
    Create a spherical surface ρ(θ, φ) = constant.

    Args:
        mesh: SurfaceMesh instance
        radius: Radius of the sphere

    Returns:
        Array of shape (N_s, N_s) with constant radial value
    """
    return np.full((mesh.N_s, mesh.N_s), radius)


def create_ellipsoid(
    mesh: SurfaceMesh,
    a: float,
    b: float,
    c: float
) -> np.ndarray:
    """
    Create an ellipsoid surface with semi-axes (a, b, c).

    The ellipsoid equation is (x/a)² + (y/b)² + (z/c)² = 1.
    In spherical coordinates: ρ(θ, φ) = abc / √((bc sin θ cos φ)² + (ac sin θ sin φ)² + (ab cos θ)²)

    Args:
        mesh: SurfaceMesh instance
        a, b, c: Semi-axes in x, y, z directions

    Returns:
        Array of shape (N_s, N_s) with radial values
    """
    theta, phi = mesh.theta_phi_grid()

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    denominator = np.sqrt(
        (b * c * sin_theta * cos_phi)**2 +
        (a * c * sin_theta * sin_phi)**2 +
        (a * b * cos_theta)**2
    )

    return a * b * c / denominator
