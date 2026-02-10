"""
Vectorized local Lagrange interpolation for surface functions.

Uses a 4×4 stencil for 4th order accuracy with truly local coupling.
Each interpolation query only depends on 16 nearby grid points.
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def _lagrange_weights_1d(x, x_nodes):
    """
    Compute Lagrange interpolation weights for a single point.

    Args:
        x: Query point
        x_nodes: Array of node positions (length n)

    Returns:
        Array of weights (length n)
    """
    n = len(x_nodes)
    weights = np.ones(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

    return weights


@jit(nopython=True, cache=True)
def _find_stencil_base(x, x_grid, n_grid, stencil_size):
    """Find the starting index for the stencil centered on x."""
    dx = x_grid[1] - x_grid[0]
    idx = int(x / dx) - stencil_size // 2 + 1

    # Clip to valid range
    if idx < 0:
        idx = 0
    elif idx > n_grid - stencil_size:
        idx = n_grid - stencil_size

    return idx


@jit(nopython=True, cache=True)
def _interp_single_point(theta, phi, rho, theta_grid, phi_grid, N_s):
    """
    Interpolate at a single point using 4×4 Lagrange stencil.

    Returns interpolated value.
    """
    stencil_size = 4

    # Find stencil bases
    i_th_base = _find_stencil_base(theta, theta_grid, N_s, stencil_size)

    d_phi = phi_grid[1] - phi_grid[0]
    i_ph_base = int(phi / d_phi) - stencil_size // 2 + 1

    # Get theta nodes (no wrapping)
    theta_nodes = theta_grid[i_th_base:i_th_base + stencil_size]

    # Get phi nodes (with wrapping)
    phi_nodes = np.empty(stencil_size)
    for k in range(stencil_size):
        phi_nodes[k] = phi_grid[(i_ph_base + k) % N_s]

    # Handle phi wrapping for interpolation
    phi_eval = phi
    wrapped = False
    for k in range(stencil_size - 1):
        if phi_nodes[k + 1] < phi_nodes[k]:
            # Wrapped around
            for m in range(k + 1, stencil_size):
                phi_nodes[m] += 2 * np.pi
            if phi_eval < np.pi:
                phi_eval += 2 * np.pi
            wrapped = True
            break

    # Extract stencil values
    stencil = np.empty((stencil_size, stencil_size))
    for i in range(stencil_size):
        for j in range(stencil_size):
            i_th = i_th_base + i
            i_ph = (i_ph_base + j) % N_s
            stencil[i, j] = rho[i_th, i_ph]

    # Compute Lagrange weights
    w_theta = _lagrange_weights_1d(theta, theta_nodes)
    w_phi = _lagrange_weights_1d(phi_eval, phi_nodes)

    # Interpolate: first in phi, then in theta
    phi_interp = np.zeros(stencil_size)
    for i in range(stencil_size):
        for j in range(stencil_size):
            phi_interp[i] += w_phi[j] * stencil[i, j]

    result = 0.0
    for i in range(stencil_size):
        result += w_theta[i] * phi_interp[i]

    return result


@jit(nopython=True, cache=True, parallel=True)
def interpolate_batch_lagrange(theta_arr, phi_arr, rho, theta_grid, phi_grid):
    """
    Batch interpolation using local 4×4 Lagrange stencils.

    Args:
        theta_arr: Array of theta query points
        phi_arr: Array of phi query points
        rho: Grid values (N_s, N_s)
        theta_grid: Theta grid points
        phi_grid: Phi grid points

    Returns:
        Array of interpolated values
    """
    n = len(theta_arr)
    N_s = len(theta_grid)
    result = np.empty(n)

    for i in prange(n):
        th = theta_arr[i]
        ph = phi_arr[i] % (2 * np.pi)

        # Handle poles
        if th <= 0:
            result[i] = rho[0, 0]
        elif th >= np.pi:
            result[i] = rho[-1, 0]
        else:
            result[i] = _interp_single_point(th, ph, rho, theta_grid, phi_grid, N_s)

    return result


@jit(nopython=True, cache=True)
def _get_stencil_indices(theta, phi, theta_grid, phi_grid, N_s):
    """
    Get the (i_theta, i_phi) indices of the 4×4 stencil for a query point.

    Returns:
        i_th_base: Starting theta index
        i_ph_indices: Array of 4 phi indices (with wrapping)
    """
    stencil_size = 4

    i_th_base = _find_stencil_base(theta, theta_grid, N_s, stencil_size)

    d_phi = phi_grid[1] - phi_grid[0]
    i_ph_base = int(phi / d_phi) - stencil_size // 2 + 1

    i_ph_indices = np.empty(stencil_size, dtype=np.int64)
    for k in range(stencil_size):
        i_ph_indices[k] = (i_ph_base + k) % N_s

    return i_th_base, i_ph_indices


class LagrangeInterpolator:
    """
    Fast local Lagrange interpolator using 4×4 stencils.

    Each interpolation depends on only 16 grid points, enabling
    sparse Jacobian computation.
    """

    def __init__(self, mesh):
        """
        Initialize with mesh information.

        Args:
            mesh: SurfaceMesh instance
        """
        self.mesh = mesh
        self.theta = mesh.theta.copy()
        self.phi = mesh.phi.copy()
        self.N_s = mesh.N_s
        self._warmed_up = False

    def _warmup(self, rho):
        """Warm up JIT compilation."""
        if not self._warmed_up:
            # Single point to trigger compilation
            _ = interpolate_batch_lagrange(
                np.array([1.0]), np.array([1.0]),
                rho, self.theta, self.phi
            )
            self._warmed_up = True

    def interpolate_batch(self, rho, theta_arr, phi_arr):
        """
        Interpolate at multiple points.

        Args:
            rho: Grid values (N_s, N_s)
            theta_arr: Query theta values
            phi_arr: Query phi values

        Returns:
            Interpolated values
        """
        self._warmup(rho)
        return interpolate_batch_lagrange(
            theta_arr, phi_arr, rho, self.theta, self.phi
        )

    def interpolate(self, rho, theta, phi):
        """Interpolate at a single point."""
        return self.interpolate_batch(rho, np.array([theta]), np.array([phi]))[0]

    def get_stencil_indices(self, theta, phi):
        """
        Get the grid indices that affect interpolation at (theta, phi).

        Returns:
            List of (i_theta, i_phi) tuples for the 16 stencil points
        """
        i_th_base, i_ph_indices = _get_stencil_indices(
            theta, phi, self.theta, self.phi, self.N_s
        )

        indices = []
        for di in range(4):
            for dj in range(4):
                indices.append((i_th_base + di, int(i_ph_indices[dj])))

        return indices
