"""
Biquartic interpolation for surface functions.

Provides O(h⁴) accurate interpolation of ρ(θ, φ) at arbitrary points
using a 4×4 (16-point) stencil.

Special handling near poles where the standard stencil cannot be used.

Reference: Huq, Choptuik & Matzner (2000), Section II.B and Fig. 22
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from .surface import SurfaceMesh


def lagrange_weights(x: float, x_nodes: np.ndarray) -> np.ndarray:
    """
    Compute Lagrange interpolation weights.

    Args:
        x: Point at which to interpolate
        x_nodes: Array of node positions

    Returns:
        Array of weights w_i such that f(x) ≈ Σ w_i f(x_i)
    """
    n = len(x_nodes)
    weights = np.ones(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

    return weights


def lagrange_derivative_weights(x: float, x_nodes: np.ndarray) -> np.ndarray:
    """
    Compute Lagrange interpolation derivative weights.

    For the Lagrange basis polynomial L_i(x), the derivative is:
    L'_i(x) = Σ_{k≠i} [Π_{j≠i,k} (x - x_j) / (x_i - x_j)]

    Args:
        x: Point at which to compute derivative
        x_nodes: Array of node positions

    Returns:
        Array of weights w'_i such that f'(x) ≈ Σ w'_i f(x_i)
    """
    n = len(x_nodes)
    deriv_weights = np.zeros(n)

    for i in range(n):
        # Compute L'_i(x)
        for k in range(n):
            if k == i:
                continue
            # This term contributes: Π_{j≠i,k} (x - x_j) / (x_i - x_j)
            term = 1.0
            for j in range(n):
                if j != i:
                    term /= (x_nodes[i] - x_nodes[j])
                    if j != k:
                        term *= (x - x_nodes[j])
            deriv_weights[i] += term

    return deriv_weights


def lagrange_weights_vectorized(x_arr: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
    """
    Compute Lagrange interpolation weights for multiple query points.

    Args:
        x_arr: Array of query points, shape (n_queries,)
        x_nodes: Array of node positions, shape (n_nodes,)

    Returns:
        Array of weights, shape (n_queries, n_nodes)
    """
    n_queries = len(x_arr)
    n_nodes = len(x_nodes)

    # Shape: (n_queries, n_nodes)
    x_diff = x_arr[:, np.newaxis] - x_nodes[np.newaxis, :]

    # Shape: (n_nodes, n_nodes) - differences between nodes
    node_diff = x_nodes[:, np.newaxis] - x_nodes[np.newaxis, :]
    np.fill_diagonal(node_diff, 1.0)  # Avoid division by zero

    # Compute weights
    weights = np.ones((n_queries, n_nodes))
    for j in range(n_nodes):
        for k in range(n_nodes):
            if j != k:
                weights[:, j] *= x_diff[:, k] / node_diff[j, k]

    return weights


class BiquarticInterpolator:
    """
    Biquartic interpolation for functions on the (θ, φ) mesh.

    Uses a 4×4 stencil for O(h⁴) accuracy. Near poles, the stencil
    is shifted to avoid crossing the pole.
    """

    def __init__(self, mesh: SurfaceMesh):
        """
        Initialize interpolator with mesh information.

        Args:
            mesh: SurfaceMesh instance defining the grid
        """
        self.mesh = mesh
        self.stencil_size = 4

    def _get_stencil_indices(
        self,
        theta: float,
        phi: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get stencil indices for a given (θ, φ) point.

        Returns shifted stencil near poles and the local indices
        of the query point within the stencil.

        Args:
            theta: θ coordinate
            phi: φ coordinate

        Returns:
            Tuple of (i_th, i_ph) index arrays
        """
        mesh = self.mesh
        N = self.stencil_size

        # Find the cell containing the point
        i_theta_float = theta / mesh.d_theta
        i_phi_float = phi / mesh.d_phi

        # Base stencil position (try to center on query point)
        i_th_base = int(np.floor(i_theta_float)) - N // 2 + 1
        i_ph_base = int(np.floor(i_phi_float)) - N // 2 + 1

        # Shift stencil if it would cross boundaries in θ
        if i_th_base < 0:
            i_th_base = 0
        elif i_th_base + N > mesh.N_s:
            i_th_base = mesh.N_s - N

        # θ indices (no wrapping)
        i_th = np.arange(i_th_base, i_th_base + N)

        # φ indices (with periodic wrapping)
        i_ph = np.arange(i_ph_base, i_ph_base + N) % mesh.N_s

        return i_th, i_ph

    def interpolate(
        self,
        rho: np.ndarray,
        theta: float,
        phi: float
    ) -> float:
        """
        Interpolate ρ at an arbitrary (θ, φ) point.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta: θ coordinate in [0, π]
            phi: φ coordinate in [0, 2π)

        Returns:
            Interpolated value of ρ
        """
        mesh = self.mesh

        # Handle pole cases directly
        if theta <= 0:
            return rho[0, 0]
        if theta >= np.pi:
            return rho[-1, 0]

        # Normalize phi to [0, 2π)
        phi = phi % (2 * np.pi)

        # Get stencil
        i_th, i_ph = self._get_stencil_indices(theta, phi)

        # Get theta and phi values at stencil points
        theta_nodes = mesh.theta[i_th]
        phi_nodes = mesh.phi[i_ph]

        # Handle periodic phi wrapping for interpolation
        # Shift phi values if stencil wraps around
        phi_eval = phi
        if np.any(np.diff(phi_nodes) < 0):
            phi_nodes = phi_nodes.copy()
            for k in range(len(phi_nodes)):
                if phi_nodes[k] < np.pi:
                    phi_nodes[k] += 2 * np.pi
            if phi_eval < np.pi:
                phi_eval += 2 * np.pi

        # Compute Lagrange weights
        w_theta = lagrange_weights(theta, theta_nodes)
        w_phi = lagrange_weights(phi_eval, phi_nodes)

        # Extract stencil values
        stencil_values = rho[np.ix_(i_th, i_ph)]

        # Biquartic interpolation: first in φ, then in θ
        phi_interp = np.dot(stencil_values, w_phi)
        result = np.dot(w_theta, phi_interp)

        return result

    def interpolate_batch(
        self,
        rho: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate ρ at multiple points efficiently using vectorized operations.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta_arr: Array of θ coordinates, shape (n_points,)
            phi_arr: Array of φ coordinates, shape (n_points,)

        Returns:
            Array of interpolated values, shape (n_points,)
        """
        mesh = self.mesh
        n_points = len(theta_arr)
        results = np.zeros(n_points)

        # Handle poles
        north_pole = theta_arr <= 0
        south_pole = theta_arr >= np.pi
        interior = ~(north_pole | south_pole)

        results[north_pole] = rho[0, 0]
        results[south_pole] = rho[-1, 0]

        if not np.any(interior):
            return results

        # Process interior points
        theta_int = theta_arr[interior]
        phi_int = phi_arr[interior] % (2 * np.pi)
        n_int = len(theta_int)

        N = self.stencil_size

        # Compute stencil base indices for all points
        i_theta_float = theta_int / mesh.d_theta
        i_phi_float = phi_int / mesh.d_phi

        i_th_base = np.floor(i_theta_float).astype(int) - N // 2 + 1
        i_ph_base = np.floor(i_phi_float).astype(int) - N // 2 + 1

        # Clip theta indices to valid range
        i_th_base = np.clip(i_th_base, 0, mesh.N_s - N)

        # Compute Lagrange weights for all points
        # For each point, we need weights for the 4 theta nodes and 4 phi nodes
        results_int = np.zeros(n_int)

        for k in range(n_int):
            theta = theta_int[k]
            phi = phi_int[k]
            i_th_b = i_th_base[k]
            i_ph_b = i_ph_base[k]

            # θ and φ stencil indices
            i_th = np.arange(i_th_b, i_th_b + N)
            i_ph = np.arange(i_ph_b, i_ph_b + N) % mesh.N_s

            # Get node positions
            theta_nodes = mesh.theta[i_th]
            phi_nodes = mesh.phi[i_ph]

            # Handle periodic phi wrapping
            phi_eval = phi
            if np.any(np.diff(phi_nodes) < 0):
                phi_nodes = phi_nodes.copy()
                for j in range(N):
                    if phi_nodes[j] < np.pi:
                        phi_nodes[j] += 2 * np.pi
                if phi_eval < np.pi:
                    phi_eval += 2 * np.pi

            # Compute Lagrange weights
            w_theta = lagrange_weights(theta, theta_nodes)
            w_phi = lagrange_weights(phi_eval, phi_nodes)

            # Extract stencil values and interpolate
            stencil_values = rho[np.ix_(i_th, i_ph)]
            phi_interp = np.dot(stencil_values, w_phi)
            results_int[k] = np.dot(w_theta, phi_interp)

        results[interior] = results_int
        return results

    def interpolate_gradient(
        self,
        rho: np.ndarray,
        theta: float,
        phi: float
    ) -> Tuple[float, float, float]:
        """
        Interpolate ρ and its gradients at an arbitrary point.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta: θ coordinate
            phi: φ coordinate

        Returns:
            Tuple of (rho, drho/dtheta, drho/dphi)
        """
        mesh = self.mesh

        # Handle poles
        if theta <= mesh.d_theta / 10:
            val = rho[0, 0]
            # Derivative estimate at north pole
            drho_dtheta = (rho[1, :].mean() - val) / mesh.d_theta
            return val, drho_dtheta, 0.0

        if theta >= np.pi - mesh.d_theta / 10:
            val = rho[-1, 0]
            drho_dtheta = (val - rho[-2, :].mean()) / mesh.d_theta
            return val, drho_dtheta, 0.0

        phi = phi % (2 * np.pi)

        # Use finite differences for derivatives
        h_theta = mesh.d_theta / 10
        h_phi = mesh.d_phi / 10

        val = self.interpolate(rho, theta, phi)

        # θ derivative
        if theta - h_theta > 0 and theta + h_theta < np.pi:
            val_p = self.interpolate(rho, theta + h_theta, phi)
            val_m = self.interpolate(rho, theta - h_theta, phi)
            drho_dtheta = (val_p - val_m) / (2 * h_theta)
        else:
            drho_dtheta = 0.0

        # φ derivative
        val_p = self.interpolate(rho, theta, (phi + h_phi) % (2 * np.pi))
        val_m = self.interpolate(rho, theta, (phi - h_phi) % (2 * np.pi))
        drho_dphi = (val_p - val_m) / (2 * h_phi)

        return val, drho_dtheta, drho_dphi

    def interpolate_array(
        self,
        rho: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate ρ at multiple points.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta_arr: Array of θ coordinates
            phi_arr: Array of φ coordinates (same shape as theta_arr)

        Returns:
            Array of interpolated values with same shape as input
        """
        shape = theta_arr.shape
        return self.interpolate_batch(rho, theta_arr.ravel(), phi_arr.ravel()).reshape(shape)


def create_interpolator(mesh: SurfaceMesh) -> BiquarticInterpolator:
    """
    Create a biquartic interpolator for the given mesh.

    Args:
        mesh: SurfaceMesh instance

    Returns:
        BiquarticInterpolator instance
    """
    return BiquarticInterpolator(mesh)


class FastInterpolator:
    """
    Fast interpolator using SciPy's RectBivariateSpline.

    Optimized for batch interpolation of many points at once.
    Uses quintic (5th order) spline interpolation for high accuracy.
    """

    def __init__(self, mesh: SurfaceMesh, spline_order: int = 5):
        """
        Initialize fast interpolator with mesh information.

        Args:
            mesh: SurfaceMesh instance defining the grid
            spline_order: Order of the spline (default 5 for quintic)
        """
        self.mesh = mesh
        self.spline_order = spline_order
        self._spline = None
        self._rho_id = None  # Track which rho array is cached

    def _build_interpolator(self, rho: np.ndarray):
        """
        Build the SciPy spline interpolator for the given rho values.

        Handles periodicity in φ by extending the grid.
        """
        mesh = self.mesh

        # Extend grid in φ direction for periodicity
        # Add extra points to handle periodic wrapping smoothly
        n_extend = min(3, mesh.N_s // 2)  # Extend by a few points on each side
        theta_extended = mesh.theta
        phi_extended = np.concatenate([
            mesh.phi[-n_extend:] - 2*np.pi,  # Points before 0
            mesh.phi,                         # Main grid
            mesh.phi[:n_extend] + 2*np.pi    # Points after 2π
        ])
        rho_extended = np.column_stack([
            rho[:, -n_extend:],  # Wrap from end
            rho,                  # Main grid
            rho[:, :n_extend]    # Wrap to beginning
        ])

        self._spline = RectBivariateSpline(
            theta_extended,
            phi_extended,
            rho_extended,
            kx=self.spline_order,
            ky=self.spline_order,
            s=0  # No smoothing, exact interpolation
        )
        self._rho_id = id(rho)

    def interpolate(self, rho: np.ndarray, theta: float, phi: float) -> float:
        """
        Interpolate ρ at an arbitrary (θ, φ) point.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta: θ coordinate in [0, π]
            phi: φ coordinate in [0, 2π)

        Returns:
            Interpolated value of ρ
        """
        # Rebuild interpolator if rho changed
        if self._spline is None or self._rho_id != id(rho):
            self._build_interpolator(rho)

        # Handle poles
        if theta <= 0:
            return rho[0, 0]
        if theta >= np.pi:
            return rho[-1, 0]

        # Normalize phi
        phi = phi % (2 * np.pi)

        return float(self._spline(theta, phi, grid=False))

    def interpolate_batch(
        self,
        rho: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate ρ at multiple points efficiently.

        This is the main performance advantage over BiquarticInterpolator.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta_arr: Array of θ coordinates, shape (n_points,)
            phi_arr: Array of φ coordinates, shape (n_points,)

        Returns:
            Array of interpolated values, shape (n_points,)
        """
        # Rebuild interpolator if rho changed
        if self._spline is None or self._rho_id != id(rho):
            self._build_interpolator(rho)

        n_points = len(theta_arr)
        results = np.zeros(n_points)

        # Handle poles
        north_pole = theta_arr <= 0
        south_pole = theta_arr >= np.pi
        interior = ~(north_pole | south_pole)

        results[north_pole] = rho[0, 0]
        results[south_pole] = rho[-1, 0]

        if np.any(interior):
            # Normalize phi to [0, 2π)
            theta_int = theta_arr[interior]
            phi_int = phi_arr[interior] % (2 * np.pi)

            # Evaluate spline at all interior points
            results[interior] = self._spline(theta_int, phi_int, grid=False)

        return results

    def interpolate_array(
        self,
        rho: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate ρ at multiple points.

        Args:
            rho: Array of shape (N_s, N_s) with grid values
            theta_arr: Array of θ coordinates
            phi_arr: Array of φ coordinates (same shape as theta_arr)

        Returns:
            Array of interpolated values with same shape as input
        """
        shape = theta_arr.shape
        return self.interpolate_batch(rho, theta_arr.ravel(), phi_arr.ravel()).reshape(shape)


def create_fast_interpolator(mesh: SurfaceMesh) -> FastInterpolator:
    """
    Create a fast SciPy-based interpolator for the given mesh.

    Args:
        mesh: SurfaceMesh instance

    Returns:
        FastInterpolator instance
    """
    return FastInterpolator(mesh)
