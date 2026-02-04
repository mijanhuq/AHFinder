"""
Numerical metric from grid data.

Provides metric quantities by interpolating from numerical data
stored on a 3D Cartesian grid.

This allows the apparent horizon finder to work with metric data
from numerical relativity simulations.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple, Optional
from .base import Metric


class NumericalMetric(Metric):
    """
    Metric interpolated from numerical grid data.

    Expects metric data on a uniform Cartesian grid, with interpolation
    used to evaluate metric quantities at arbitrary points.

    The grid data dictionary should contain:
        - 'gamma_xx', 'gamma_xy', 'gamma_xz', 'gamma_yy', 'gamma_yz', 'gamma_zz'
        - 'K_xx', 'K_xy', 'K_xz', 'K_yy', 'K_yz', 'K_zz'
        - Optionally: 'alpha' (lapse), 'beta_x', 'beta_y', 'beta_z' (shift)

    Each array should have shape (nx, ny, nz).

    Attributes:
        bounds: Grid bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        interpolation_method: Interpolation method ('linear' or 'cubic')
    """

    def __init__(
        self,
        grid_data: Dict[str, np.ndarray],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        interpolation_method: str = 'linear'
    ):
        """
        Initialize numerical metric from grid data.

        Args:
            grid_data: Dictionary mapping component names to 3D arrays
            bounds: Grid bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            interpolation_method: 'linear' or 'cubic'
        """
        self.bounds = bounds
        self.interpolation_method = interpolation_method

        # Validate required components
        gamma_components = ['gamma_xx', 'gamma_xy', 'gamma_xz',
                           'gamma_yy', 'gamma_yz', 'gamma_zz']
        K_components = ['K_xx', 'K_xy', 'K_xz', 'K_yy', 'K_yz', 'K_zz']

        for comp in gamma_components + K_components:
            if comp not in grid_data:
                raise ValueError(f"Missing required component: {comp}")

        # Get grid shape from first component
        shape = grid_data['gamma_xx'].shape
        if len(shape) != 3:
            raise ValueError("Grid data must be 3D arrays")

        nx, ny, nz = shape

        # Create coordinate arrays
        x = np.linspace(bounds[0][0], bounds[0][1], nx)
        y = np.linspace(bounds[1][0], bounds[1][1], ny)
        z = np.linspace(bounds[2][0], bounds[2][1], nz)

        self._coords = (x, y, z)

        # Build interpolators
        self._interpolators = {}

        for name, data in grid_data.items():
            if data.shape != shape:
                raise ValueError(f"Component {name} has wrong shape")
            self._interpolators[name] = RegularGridInterpolator(
                (x, y, z), data,
                method=interpolation_method,
                bounds_error=False,
                fill_value=None
            )

        # Check for optional components
        self._has_lapse = 'alpha' in grid_data
        self._has_shift = all(f'beta_{c}' in grid_data for c in ['x', 'y', 'z'])

    def _interp(self, name: str, x: float, y: float, z: float) -> float:
        """Interpolate a single component at a point."""
        return float(self._interpolators[name]((x, y, z)))

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute 3-metric γ_ij by interpolation.
        """
        g = np.zeros((3, 3))

        g[0, 0] = self._interp('gamma_xx', x, y, z)
        g[0, 1] = g[1, 0] = self._interp('gamma_xy', x, y, z)
        g[0, 2] = g[2, 0] = self._interp('gamma_xz', x, y, z)
        g[1, 1] = self._interp('gamma_yy', x, y, z)
        g[1, 2] = g[2, 1] = self._interp('gamma_yz', x, y, z)
        g[2, 2] = self._interp('gamma_zz', x, y, z)

        return g

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute inverse 3-metric by matrix inversion.
        """
        g = self.gamma(x, y, z)
        return np.linalg.inv(g)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute ∂_k γ_ij by numerical differentiation of interpolated values.
        """
        h = 1e-6

        dgamma = np.zeros((3, 3, 3))

        coords = [
            (x + h, y, z), (x - h, y, z),
            (x, y + h, z), (x, y - h, z),
            (x, y, z + h), (x, y, z - h)
        ]

        gammas = [self.gamma(*c) for c in coords]

        dgamma[0] = (gammas[0] - gammas[1]) / (2 * h)
        dgamma[1] = (gammas[2] - gammas[3]) / (2 * h)
        dgamma[2] = (gammas[4] - gammas[5]) / (2 * h)

        return dgamma

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute extrinsic curvature K_ij by interpolation.
        """
        K = np.zeros((3, 3))

        K[0, 0] = self._interp('K_xx', x, y, z)
        K[0, 1] = K[1, 0] = self._interp('K_xy', x, y, z)
        K[0, 2] = K[2, 0] = self._interp('K_xz', x, y, z)
        K[1, 1] = self._interp('K_yy', x, y, z)
        K[1, 2] = K[2, 1] = self._interp('K_yz', x, y, z)
        K[2, 2] = self._interp('K_zz', x, y, z)

        return K

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute lapse α by interpolation (if available).
        """
        if self._has_lapse:
            return self._interp('alpha', x, y, z)
        return 1.0

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute shift β^i by interpolation (if available).
        """
        if self._has_shift:
            return np.array([
                self._interp('beta_x', x, y, z),
                self._interp('beta_y', x, y, z),
                self._interp('beta_z', x, y, z)
            ])
        return np.zeros(3)


def create_numerical_metric(
    grid_data: Dict[str, np.ndarray],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    interpolation_method: str = 'linear'
) -> NumericalMetric:
    """
    Create a numerical metric from grid data.

    Args:
        grid_data: Dictionary of metric components
        bounds: Grid bounds
        interpolation_method: Interpolation method

    Returns:
        NumericalMetric instance
    """
    return NumericalMetric(grid_data, bounds, interpolation_method)


def metric_from_schwarzschild_grid(
    M: float = 1.0,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-10, 10), (-10, 10), (-10, 10)),
    resolution: int = 64
) -> NumericalMetric:
    """
    Create a numerical metric grid from analytical Schwarzschild data.

    Useful for testing the numerical metric infrastructure.

    Args:
        M: Black hole mass
        bounds: Grid bounds
        resolution: Number of points in each direction

    Returns:
        NumericalMetric instance
    """
    from .schwarzschild import SchwarzschildMetric

    schw = SchwarzschildMetric(M)

    nx = ny = nz = resolution
    x = np.linspace(bounds[0][0], bounds[0][1], nx)
    y = np.linspace(bounds[1][0], bounds[1][1], ny)
    z = np.linspace(bounds[2][0], bounds[2][1], nz)

    grid_data = {
        'gamma_xx': np.zeros((nx, ny, nz)),
        'gamma_xy': np.zeros((nx, ny, nz)),
        'gamma_xz': np.zeros((nx, ny, nz)),
        'gamma_yy': np.zeros((nx, ny, nz)),
        'gamma_yz': np.zeros((nx, ny, nz)),
        'gamma_zz': np.zeros((nx, ny, nz)),
        'K_xx': np.zeros((nx, ny, nz)),
        'K_xy': np.zeros((nx, ny, nz)),
        'K_xz': np.zeros((nx, ny, nz)),
        'K_yy': np.zeros((nx, ny, nz)),
        'K_yz': np.zeros((nx, ny, nz)),
        'K_zz': np.zeros((nx, ny, nz)),
        'alpha': np.zeros((nx, ny, nz)),
        'beta_x': np.zeros((nx, ny, nz)),
        'beta_y': np.zeros((nx, ny, nz)),
        'beta_z': np.zeros((nx, ny, nz)),
    }

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                g = schw.gamma(xi, yj, zk)
                K = schw.extrinsic_curvature(xi, yj, zk)
                beta = schw.shift(xi, yj, zk)

                grid_data['gamma_xx'][i, j, k] = g[0, 0]
                grid_data['gamma_xy'][i, j, k] = g[0, 1]
                grid_data['gamma_xz'][i, j, k] = g[0, 2]
                grid_data['gamma_yy'][i, j, k] = g[1, 1]
                grid_data['gamma_yz'][i, j, k] = g[1, 2]
                grid_data['gamma_zz'][i, j, k] = g[2, 2]

                grid_data['K_xx'][i, j, k] = K[0, 0]
                grid_data['K_xy'][i, j, k] = K[0, 1]
                grid_data['K_xz'][i, j, k] = K[0, 2]
                grid_data['K_yy'][i, j, k] = K[1, 1]
                grid_data['K_yz'][i, j, k] = K[1, 2]
                grid_data['K_zz'][i, j, k] = K[2, 2]

                grid_data['alpha'][i, j, k] = schw.lapse(xi, yj, zk)
                grid_data['beta_x'][i, j, k] = beta[0]
                grid_data['beta_y'][i, j, k] = beta[1]
                grid_data['beta_z'][i, j, k] = beta[2]

    return NumericalMetric(grid_data, bounds)
