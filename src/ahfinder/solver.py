"""
Newton solver for finding the apparent horizon.

Implements the Newton iteration:
    ρ_{n+1} = ρ_n - J⁻¹ F[ρ_n]

where F[ρ] is the residual and J is the Jacobian.

The linear system J δρ = -F is solved using iterative methods
(GMRES or BiCGSTAB with ILU preconditioning) for efficiency.

Reference: Huq, Choptuik & Matzner (2000), Section II.E
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres, spilu, LinearOperator
from typing import Tuple, Optional, Callable, List
import warnings

from .surface import SurfaceMesh, create_sphere
from .interpolation import BiquarticInterpolator, FastInterpolator
from .stencil import CartesianStencil
from .residual import ResidualEvaluator, create_residual_evaluator
from .jacobian import JacobianComputer
from .metrics.base import Metric


class ConvergenceError(Exception):
    """Raised when Newton iteration fails to converge."""
    pass


class NewtonSolver:
    """
    Newton solver for the apparent horizon equation F[ρ] = 0.

    Attributes:
        mesh: SurfaceMesh instance
        metric: Metric providing geometric data
        center: Center of coordinate system
        tol: Convergence tolerance
        max_iter: Maximum Newton iterations
        epsilon: Perturbation for Jacobian computation
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        tol: float = 1e-9,
        max_iter: int = 20,
        epsilon: float = 1e-5,
        spacing_factor: float = 0.5,
        verbose: bool = True,
        use_fast_interpolator: bool = True
    ):
        """
        Initialize Newton solver.

        Args:
            mesh: SurfaceMesh instance
            metric: Metric providing geometric data
            center: Center of coordinate system
            tol: Convergence tolerance for ||δρ||
            max_iter: Maximum Newton iterations
            epsilon: Perturbation for Jacobian computation
            spacing_factor: Stencil spacing factor
            verbose: Print convergence information
            use_fast_interpolator: Use SciPy-based fast interpolator (default True)
        """
        self.mesh = mesh
        self.metric = metric
        self.center = center
        self.tol = tol
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose

        # Set up interpolator and residual evaluator
        if use_fast_interpolator:
            self.interpolator = FastInterpolator(mesh)
        else:
            self.interpolator = BiquarticInterpolator(mesh)
        self.residual_evaluator = create_residual_evaluator(
            mesh, self.interpolator, metric, center, spacing_factor
        )
        self.jacobian_computer = JacobianComputer(
            mesh, self.residual_evaluator, epsilon
        )

        # Storage for convergence history
        self.residual_history: List[float] = []
        self.delta_rho_history: List[float] = []

    def solve(
        self,
        initial_guess: Optional[np.ndarray] = None,
        initial_radius: Optional[float] = None
    ) -> np.ndarray:
        """
        Find the apparent horizon using Newton iteration.

        Implements the algorithm from Huq, Choptuik & Matzner (2000):
            Start with initial guess ρ = ρ0
            while ||F|| > tolerance:
                Compute Jacobian J for current ρ
                Evaluate F[ρ]
                Solve J · δρ = -F[ρ] for δρ
                Update surface: ρ = ρ + δρ

        Args:
            initial_guess: Initial surface ρ(θ, φ), shape (N_s, N_s)
            initial_radius: If no initial_guess, use this radius for sphere

        Returns:
            Converged surface ρ(θ, φ)

        Raises:
            ConvergenceError: If iteration fails to converge
        """
        # Set up initial guess
        if initial_guess is not None:
            rho = initial_guess.copy()
        elif initial_radius is not None:
            rho = create_sphere(self.mesh, initial_radius)
        else:
            # Default: try to guess horizon radius from metric
            rho = self._default_initial_guess()

        # Clear history
        self.residual_history = []
        self.delta_rho_history = []

        if self.verbose:
            print("Newton iteration for apparent horizon:")
            print(f"  N_s = {self.mesh.N_s}, tol = {self.tol}")
            print("-" * 50)

        for iteration in range(self.max_iter):
            # Evaluate F[ρ] everywhere on S
            F = self.residual_evaluator.evaluate(rho)
            F_norm = np.linalg.norm(F)
            self.residual_history.append(F_norm)

            if self.verbose:
                print(f"  Iter {iteration:3d}: ||F|| = {F_norm:.6e}", end="")

            # Check for convergence: ||F|| < tolerance
            if F_norm < self.tol:
                if self.verbose:
                    print()
                    print("-" * 50)
                    print(f"Converged in {iteration + 1} iterations")
                return rho

            # Compute the Jacobian J for current ρ
            # Use dense Jacobian - the sparse version misses important couplings
            J = self.jacobian_computer.compute_dense(rho)

            # Solve J · δρ = -F[ρ] for δρ
            delta_rho_flat = np.linalg.solve(J, -F)

            # Convert flat array to grid
            delta_rho = self.mesh.flat_to_grid(delta_rho_flat)

            # Update the surface: ρ = ρ + δρ
            rho = rho + delta_rho

            # Track step size for diagnostics
            delta_norm = np.linalg.norm(delta_rho_flat)
            self.delta_rho_history.append(delta_norm)

            if self.verbose:
                print(f", ||δρ|| = {delta_norm:.6e}")

        # Failed to converge
        if self.verbose:
            print("-" * 50)
            print("WARNING: Failed to converge!")

        raise ConvergenceError(
            f"Newton iteration did not converge after {self.max_iter} iterations. "
            f"Final ||F|| = {self.residual_history[-1]:.2e}"
        )

    def _solve_linear_system(
        self,
        J: sparse.csr_matrix,
        b: np.ndarray
    ) -> np.ndarray:
        """
        Solve the linear system J x = b using iterative methods.

        Uses GMRES with ILU preconditioning.

        Args:
            J: Sparse Jacobian matrix
            b: Right-hand side vector

        Returns:
            Solution vector x
        """
        n = len(b)

        try:
            # Try ILU preconditioner
            ilu = spilu(J.tocsc(), drop_tol=1e-4)
            M = LinearOperator((n, n), lambda x: ilu.solve(x))

            # Solve with GMRES
            x, info = gmres(J, b, M=M, rtol=1e-10, maxiter=100)

            if info != 0:
                warnings.warn(f"GMRES did not converge (info={info})")
        except Exception:
            # Fallback to unpreconditioned GMRES
            x, info = gmres(J, b, rtol=1e-10, maxiter=200)

            if info != 0:
                # Last resort: dense solve
                warnings.warn("Using dense linear solve as fallback")
                x = np.linalg.solve(J.toarray(), b)

        return x

    def _default_initial_guess(self) -> np.ndarray:
        """
        Generate a default initial guess based on the metric.

        For Schwarzschild-like metrics, uses r = 2M.
        """
        # Try to get characteristic radius from metric
        if hasattr(self.metric, 'horizon_radius'):
            r0 = self.metric.horizon_radius()
        elif hasattr(self.metric, 'horizon_radius_equatorial'):
            r0 = self.metric.horizon_radius_equatorial()
        elif hasattr(self.metric, 'M'):
            r0 = 2.0 * self.metric.M
        else:
            r0 = 1.0

        return create_sphere(self.mesh, r0)


def find_horizon(
    metric: Metric,
    N_s: int = 33,
    initial_guess: Optional[np.ndarray] = None,
    initial_radius: Optional[float] = None,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    tol: float = 1e-9,
    max_iter: int = 20,
    verbose: bool = True
) -> Tuple[np.ndarray, SurfaceMesh]:
    """
    Find the apparent horizon for a given metric.

    This is the main entry point for horizon finding.

    Args:
        metric: Metric providing geometric data
        N_s: Number of grid points in each direction
        initial_guess: Initial surface guess (optional)
        initial_radius: Initial sphere radius if no guess provided
        center: Center of coordinate system
        tol: Convergence tolerance
        max_iter: Maximum Newton iterations
        verbose: Print convergence information

    Returns:
        Tuple of (rho, mesh) where rho is the converged surface
        and mesh is the SurfaceMesh instance

    Raises:
        ConvergenceError: If iteration fails to converge

    Example:
        >>> from ahfinder.metrics import SchwarzschildMetric
        >>> metric = SchwarzschildMetric(M=1.0)
        >>> rho, mesh = find_horizon(metric, N_s=33)
        >>> print(f"Horizon radius: {rho.mean():.6f}")
    """
    mesh = SurfaceMesh(N_s)
    solver = NewtonSolver(
        mesh, metric, center, tol, max_iter,
        verbose=verbose
    )

    rho = solver.solve(initial_guess, initial_radius)

    return rho, mesh
