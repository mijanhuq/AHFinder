"""
Newton solver for finding the apparent horizon.

Implements the Newton iteration:
    ρ_{n+1} = ρ_n - J⁻¹ F[ρ_n]

where F[ρ] is the residual and J is the Jacobian.

Two solver modes are available:
1. Dense Jacobian: Compute full J and solve with numpy.linalg.solve
2. Jacobian-Free Newton-Krylov (JFNK): Use iterative GMRES with
   matrix-vector products computed via finite differences:
   J @ v ≈ (F(ρ + εv) - F(ρ)) / ε

JFNK reduces complexity from O(n²) to O(n × k) where k is the number
of Krylov iterations, providing significant speedup for larger problems.

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
        use_fast_interpolator: bool = True,
        use_jfnk: bool = False,
        jfnk_maxiter: int = 50,
        jfnk_tol: float = 1e-6
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
            use_jfnk: Use Jacobian-Free Newton-Krylov method (default False)
            jfnk_maxiter: Maximum GMRES iterations for JFNK
            jfnk_tol: Relative tolerance for GMRES in JFNK
        """
        self.mesh = mesh
        self.metric = metric
        self.center = center
        self.tol = tol
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        self.use_jfnk = use_jfnk
        self.jfnk_maxiter = jfnk_maxiter
        self.jfnk_tol = jfnk_tol

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
        self.jfnk_iterations: List[int] = []  # Track GMRES iterations

        # Cached preconditioner for JFNK (lagged Jacobian)
        self._jfnk_precond = None
        self._jfnk_precond_lu = None

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
        self.jfnk_iterations = []

        # Clear preconditioner cache for fresh solve
        self._jfnk_precond = None
        self._jfnk_precond_lu = None

        if self.verbose:
            print("Newton iteration for apparent horizon:")
            mode = "JFNK" if self.use_jfnk else "Dense Jacobian"
            print(f"  N_s = {self.mesh.N_s}, tol = {self.tol}, mode = {mode}")
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

            # Solve J · δρ = -F[ρ] for δρ
            if self.use_jfnk:
                # Jacobian-Free Newton-Krylov: use GMRES with matrix-free matvec
                delta_rho_flat, gmres_iters = self._solve_jfnk(rho, F, iteration)
                self.jfnk_iterations.append(gmres_iters)
            else:
                # Dense Jacobian: compute full J and solve directly
                J = self.jacobian_computer.compute_dense(rho)
                delta_rho_flat = np.linalg.solve(J, -F)

            # Convert flat array to grid
            delta_rho = self.mesh.flat_to_grid(delta_rho_flat)

            # Update the surface: ρ = ρ + δρ
            rho = rho + delta_rho

            # Track step size for diagnostics
            delta_norm = np.linalg.norm(delta_rho_flat)
            self.delta_rho_history.append(delta_norm)

            if self.verbose:
                if self.use_jfnk:
                    print(f", ||δρ|| = {delta_norm:.6e} (GMRES: {self.jfnk_iterations[-1]} iters)")
                else:
                    print(f", ||δρ|| = {delta_norm:.6e}")

        # Failed to converge
        if self.verbose:
            print("-" * 50)
            print("WARNING: Failed to converge!")

        raise ConvergenceError(
            f"Newton iteration did not converge after {self.max_iter} iterations. "
            f"Final ||F|| = {self.residual_history[-1]:.2e}"
        )

    def _solve_jfnk(
        self,
        rho: np.ndarray,
        F: np.ndarray,
        iteration: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Solve J · δρ = -F using Jacobian-Free Newton-Krylov method.

        Uses GMRES with matrix-vector products computed via finite differences:
            J @ v ≈ (F(ρ + εv) - F(ρ)) / ε

        This avoids forming the full Jacobian, reducing complexity from
        O(n²) to O(n × k) where k is the number of GMRES iterations.

        A lagged Jacobian preconditioner is used to accelerate convergence:
        - On the first Newton iteration, we compute the full Jacobian
        - Its LU factorization is used as a preconditioner for GMRES
        - This reduces GMRES iterations from O(n) to O(1) typically

        Args:
            rho: Current surface (grid form)
            F: Current residual (flat form)
            iteration: Current Newton iteration number

        Returns:
            Tuple of (delta_rho_flat, gmres_iterations)
        """
        # Make a copy of F to avoid potential aliasing issues in the matvec closure
        # (This prevents a subtle bug where F could be modified between iterations)
        F = F.copy()
        n = len(F)

        # Counter for GMRES iterations
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        # Pre-compute rho norm for epsilon scaling
        rho_flat = self.mesh.grid_to_flat(rho)
        rho_norm = np.linalg.norm(rho_flat)

        # Pre-compute constants for matvec
        sqrt_eps = np.sqrt(np.finfo(float).eps)  # ~1.5e-8

        def matvec(v):
            """
            Compute J @ v using finite difference approximation.

            J @ v ≈ (F(ρ + ε*v̂) - F(ρ)) / ε

            Uses the formula from Knoll & Keyes (2004):
            ε = sqrt(machine_eps) * (1 + ||ρ||) / ||v||

            This balances truncation error and roundoff error.

            Note: We re-evaluate F(ρ) each call to avoid subtle caching bugs.
            """
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-14:
                return np.zeros(n)

            # Optimal epsilon from Knoll & Keyes (2004), Eq. 5
            eps = sqrt_eps * (1.0 + rho_norm) / v_norm

            # Convert v to grid form, perturb rho, evaluate F
            v_grid = self.mesh.flat_to_grid(v)
            rho_perturbed = rho + eps * v_grid

            # Evaluate both F(rho) and F(rho_perturbed) fresh to avoid caching issues
            F_base = self.residual_evaluator.evaluate(rho)
            F_perturbed = self.residual_evaluator.evaluate(rho_perturbed)

            # J @ v ≈ (F(ρ + εv) - F(ρ)) / ε
            return (F_perturbed - F_base) / eps

        # Create LinearOperator for GMRES
        J_op = LinearOperator((n, n), matvec=matvec, dtype=float)

        # Build or update preconditioner
        # We compute a full Jacobian on the first iteration and use it
        # as a lagged preconditioner for subsequent iterations
        M = None
        if iteration == 0 or self._jfnk_precond_lu is None:
            # Compute full Jacobian for preconditioning
            if self.verbose:
                print(" [computing preconditioner]", end="", flush=True)
            J_dense = self.jacobian_computer.compute_dense(rho)
            self._jfnk_precond = J_dense

            # LU factorization for fast solves
            from scipy.linalg import lu_factor
            self._jfnk_precond_lu = lu_factor(J_dense)

        # Create preconditioner as LinearOperator
        if self._jfnk_precond_lu is not None:
            from scipy.linalg import lu_solve
            lu, piv = self._jfnk_precond_lu

            def precond_solve(v):
                return lu_solve((lu, piv), v)

            M = LinearOperator((n, n), matvec=precond_solve, dtype=float)

        # Solve with GMRES
        # Use restart to help with convergence
        delta_rho_flat, info = gmres(
            J_op, -F,
            M=M,  # Preconditioner
            rtol=self.jfnk_tol,
            atol=self.tol * 0.1,  # Absolute tolerance based on Newton tolerance
            maxiter=self.jfnk_maxiter,
            restart=min(30, n),  # Restart GMRES for better convergence
            callback=callback,
            callback_type='x'
        )

        if info != 0:
            warnings.warn(f"JFNK GMRES did not fully converge (info={info})")

        return delta_rho_flat, iteration_count[0]

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
