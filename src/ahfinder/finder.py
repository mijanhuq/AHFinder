"""
High-level API for apparent horizon finding.

Provides the ApparentHorizonFinder class as the main user-facing interface.
"""

import numpy as np
from typing import Tuple, Optional, List

from .surface import SurfaceMesh, create_sphere, create_ellipsoid
from .interpolation import BiquarticInterpolator, FastInterpolator
from .solver import NewtonSolver, ConvergenceError
from .metrics.base import Metric


class ApparentHorizonFinder:
    """
    High-level interface for finding apparent horizons.

    This class wraps the Newton solver and provides convenient methods
    for horizon finding, area computation, and coordinate extraction.

    Example:
        >>> from ahfinder import ApparentHorizonFinder
        >>> from ahfinder.metrics import SchwarzschildMetric
        >>>
        >>> metric = SchwarzschildMetric(M=1.0)
        >>> finder = ApparentHorizonFinder(metric, N_s=33)
        >>> rho = finder.find()
        >>> print(f"Horizon area: {finder.horizon_area(rho):.6f}")
        >>> print(f"Expected: {16 * np.pi:.6f}")  # 4π(2M)² for Schwarzschild
    """

    def __init__(
        self,
        metric: Metric,
        N_s: int = 33,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        epsilon: float = 1e-5,
        spacing_factor: float = 0.5,
        use_fast_interpolator: bool = True,
        use_jfnk: bool = False,
        jfnk_maxiter: int = 50,
        jfnk_tol: float = 1e-6
    ):
        """
        Initialize the apparent horizon finder.

        Args:
            metric: Metric object providing geometric data
            N_s: Number of grid points in each direction (default 33)
            center: Center of the coordinate system (default origin)
            epsilon: Perturbation size for Jacobian computation
            spacing_factor: Cartesian stencil spacing factor
            use_fast_interpolator: Use SciPy-based fast interpolator (default True)
            use_jfnk: Use Jacobian-Free Newton-Krylov solver (default False).
                      JFNK avoids computing the full Jacobian, reducing
                      complexity from O(n²) to O(n × k) where k is GMRES iterations.
            jfnk_maxiter: Maximum GMRES iterations for JFNK
            jfnk_tol: Relative tolerance for GMRES in JFNK
        """
        self.metric = metric
        self.N_s = N_s
        self.center = center
        self.epsilon = epsilon
        self.spacing_factor = spacing_factor
        self.use_fast_interpolator = use_fast_interpolator
        self.use_jfnk = use_jfnk
        self.jfnk_maxiter = jfnk_maxiter
        self.jfnk_tol = jfnk_tol

        # Create mesh
        self.mesh = SurfaceMesh(N_s)
        if use_fast_interpolator:
            self.interpolator = FastInterpolator(self.mesh)
        else:
            self.interpolator = BiquarticInterpolator(self.mesh)

        # Solver will be created when needed
        self._solver: Optional[NewtonSolver] = None

        # Store last result
        self._last_rho: Optional[np.ndarray] = None

    def find(
        self,
        initial_guess: Optional[np.ndarray] = None,
        initial_radius: Optional[float] = None,
        tol: float = 1e-9,
        max_iter: int = 20,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Find the apparent horizon.

        Args:
            initial_guess: Initial surface ρ(θ, φ), shape (N_s, N_s)
            initial_radius: If no guess provided, use this radius
            tol: Convergence tolerance for ||δρ||
            max_iter: Maximum Newton iterations
            verbose: Print convergence information

        Returns:
            Converged surface ρ(θ, φ) as (N_s, N_s) array

        Raises:
            ConvergenceError: If Newton iteration fails to converge
        """
        # Create solver
        self._solver = NewtonSolver(
            self.mesh,
            self.metric,
            self.center,
            tol,
            max_iter,
            self.epsilon,
            self.spacing_factor,
            verbose,
            self.use_fast_interpolator,
            self.use_jfnk,
            self.jfnk_maxiter,
            self.jfnk_tol
        )

        # Find horizon
        self._last_rho = self._solver.solve(initial_guess, initial_radius)

        return self._last_rho.copy()

    def horizon_coordinates(
        self,
        rho: Optional[np.ndarray] = None,
        n_theta: Optional[int] = None,
        n_phi: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Cartesian coordinates of the horizon surface.

        Args:
            rho: Surface values (uses last result if None)
            n_theta: Number of points in θ direction (default N_s)
            n_phi: Number of points in φ direction (default N_s)

        Returns:
            Tuple of (x, y, z) coordinate arrays
        """
        if rho is None:
            if self._last_rho is None:
                raise ValueError("No horizon found yet. Call find() first.")
            rho = self._last_rho

        if n_theta is None:
            n_theta = self.N_s
        if n_phi is None:
            n_phi = self.N_s

        if n_theta == self.N_s and n_phi == self.N_s:
            # Use grid directly
            return self.mesh.xyz_from_rho(rho, self.center)

        # Interpolate to new resolution
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

        rho_interp = self.interpolator.interpolate_array(rho, theta_grid, phi_grid)

        cx, cy, cz = self.center
        sin_theta = np.sin(theta_grid)
        cos_theta = np.cos(theta_grid)
        sin_phi = np.sin(phi_grid)
        cos_phi = np.cos(phi_grid)

        x = cx + rho_interp * sin_theta * cos_phi
        y = cy + rho_interp * sin_theta * sin_phi
        z = cz + rho_interp * cos_theta

        return x, y, z

    def horizon_area(self, rho: Optional[np.ndarray] = None) -> float:
        """
        Compute the proper area of the horizon.

        The area is computed by integrating the induced metric on the surface.

        Args:
            rho: Surface values (uses last result if None)

        Returns:
            Proper area of the horizon
        """
        if rho is None:
            if self._last_rho is None:
                raise ValueError("No horizon found yet. Call find() first.")
            rho = self._last_rho

        mesh = self.mesh
        cx, cy, cz = self.center

        total_area = 0.0

        # Integrate over the surface using trapezoidal rule
        for i_theta in range(mesh.N_s):
            theta = mesh.theta[i_theta]
            sin_theta = np.sin(theta)

            # Skip exact poles (zero area contribution)
            if sin_theta < 1e-10:
                continue

            for i_phi in range(mesh.N_s):
                phi = mesh.phi[i_phi]
                r = rho[i_theta, i_phi]

                # Surface point
                x = cx + r * sin_theta * np.cos(phi)
                y = cy + r * sin_theta * np.sin(phi)
                z = cz + r * np.cos(theta)

                # Get metric at this point
                gamma = self.metric.gamma(x, y, z)

                # Compute surface tangent vectors
                # ∂X/∂θ and ∂X/∂φ

                # Numerical derivatives of rho
                if i_theta > 0 and i_theta < mesh.N_s - 1:
                    drho_dtheta = (rho[i_theta + 1, i_phi] - rho[i_theta - 1, i_phi]) / (2 * mesh.d_theta)
                elif i_theta == 0:
                    drho_dtheta = (rho[1, i_phi] - rho[0, i_phi]) / mesh.d_theta
                else:
                    drho_dtheta = (rho[-1, i_phi] - rho[-2, i_phi]) / mesh.d_theta

                i_phi_p = (i_phi + 1) % mesh.N_s
                i_phi_m = (i_phi - 1) % mesh.N_s
                drho_dphi = (rho[i_theta, i_phi_p] - rho[i_theta, i_phi_m]) / (2 * mesh.d_phi)

                cos_theta = np.cos(theta)
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                # Tangent vector in θ direction
                e_theta = np.array([
                    drho_dtheta * sin_theta * cos_phi + r * cos_theta * cos_phi,
                    drho_dtheta * sin_theta * sin_phi + r * cos_theta * sin_phi,
                    drho_dtheta * cos_theta - r * sin_theta
                ])

                # Tangent vector in φ direction
                e_phi = np.array([
                    drho_dphi * sin_theta * cos_phi - r * sin_theta * sin_phi,
                    drho_dphi * sin_theta * sin_phi + r * sin_theta * cos_phi,
                    drho_dphi * cos_theta
                ])

                # Induced metric components
                g_theta_theta = np.einsum('i,ij,j->', e_theta, gamma, e_theta)
                g_theta_phi = np.einsum('i,ij,j->', e_theta, gamma, e_phi)
                g_phi_phi = np.einsum('i,ij,j->', e_phi, gamma, e_phi)

                # Determinant of induced metric
                det_h = g_theta_theta * g_phi_phi - g_theta_phi**2

                if det_h > 0:
                    dA = np.sqrt(det_h) * mesh.d_theta * mesh.d_phi
                    total_area += dA

        return total_area

    def horizon_radius_average(self, rho: Optional[np.ndarray] = None) -> float:
        """
        Compute the average coordinate radius of the horizon.

        Args:
            rho: Surface values (uses last result if None)

        Returns:
            Average radius
        """
        if rho is None:
            if self._last_rho is None:
                raise ValueError("No horizon found yet. Call find() first.")
            rho = self._last_rho

        return np.mean(rho)

    def horizon_radius_equatorial(self, rho: Optional[np.ndarray] = None) -> float:
        """
        Compute the equatorial radius of the horizon.

        Args:
            rho: Surface values (uses last result if None)

        Returns:
            Equatorial radius (at θ = π/2)
        """
        if rho is None:
            if self._last_rho is None:
                raise ValueError("No horizon found yet. Call find() first.")
            rho = self._last_rho

        i_eq = self.mesh.N_s // 2
        return np.mean(rho[i_eq, :])

    def horizon_radius_polar(self, rho: Optional[np.ndarray] = None) -> float:
        """
        Compute the polar radius of the horizon.

        Args:
            rho: Surface values (uses last result if None)

        Returns:
            Average of north and south pole radii
        """
        if rho is None:
            if self._last_rho is None:
                raise ValueError("No horizon found yet. Call find() first.")
            rho = self._last_rho

        r_north = rho[0, 0]
        r_south = rho[-1, 0]
        return 0.5 * (r_north + r_south)

    def irreducible_mass(self, rho: Optional[np.ndarray] = None) -> float:
        """
        Compute the irreducible mass from the horizon area.

        M_irr = √(A / 16π)

        Args:
            rho: Surface values (uses last result if None)

        Returns:
            Irreducible mass
        """
        A = self.horizon_area(rho)
        return np.sqrt(A / (16 * np.pi))

    @property
    def convergence_history(self) -> Tuple[List[float], List[float]]:
        """
        Get convergence history from last solve.

        Returns:
            Tuple of (residual_norms, delta_rho_norms) lists
        """
        if self._solver is None:
            return [], []
        return self._solver.residual_history, self._solver.delta_rho_history

    def refine(
        self,
        rho: np.ndarray,
        new_N_s: int,
        tol: float = 1e-9,
        max_iter: int = 10,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Refine a solution to higher resolution.

        Interpolates the current solution to a finer mesh and runs
        a few more Newton iterations.

        Args:
            rho: Current solution at current resolution
            new_N_s: New (higher) resolution
            tol: Convergence tolerance
            max_iter: Maximum iterations for refinement
            verbose: Print convergence information

        Returns:
            Refined solution at new resolution
        """
        if new_N_s <= self.N_s:
            raise ValueError("new_N_s must be greater than current N_s")

        # Create new mesh
        new_mesh = SurfaceMesh(new_N_s)
        new_theta, new_phi = new_mesh.theta_phi_grid()

        # Interpolate current solution to new mesh
        new_rho = self.interpolator.interpolate_array(rho, new_theta, new_phi)

        # Create new finder at higher resolution
        new_finder = ApparentHorizonFinder(
            self.metric,
            new_N_s,
            self.center,
            self.epsilon,
            self.spacing_factor,
            self.use_fast_interpolator
        )

        # Solve with interpolated solution as initial guess
        refined_rho = new_finder.find(
            initial_guess=new_rho,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose
        )

        return refined_rho
