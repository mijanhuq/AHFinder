"""
Implicit time stepping for Level Flow evolution.

Implements backward Euler (implicit) time stepping which allows
larger time steps without instability:

    ρ^{n+1} - ρ^n + dt·Θ(ρ^{n+1}) = 0

Each step requires solving a nonlinear system using Newton iteration
with the Jacobian: J = I + dt·(∂Θ/∂ρ)

Reference: Shoemaker, Huq & Matzner (2000) - arXiv:gr-qc/0004062
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, spilu, LinearOperator, bicgstab
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
import warnings

from ..surface import SurfaceMesh
from ..metrics.base import Metric
from ..residual_vectorized import create_vectorized_residual_evaluator
from ..jacobian_vectorized import create_vectorized_jacobian_computer


@dataclass
class ImplicitStepResult:
    """Result of a single implicit time step."""
    rho: np.ndarray              # Updated surface shape
    n_newton_iters: int          # Newton iterations taken
    converged: bool              # Whether Newton converged
    final_theta_norm: float      # ||Θ|| at new position


class ImplicitLevelFlowStepper:
    """
    Backward Euler time stepper for Level Flow evolution.

    Uses implicit time stepping which is unconditionally stable,
    allowing much larger time steps than explicit methods.

    Each step solves:
        G(ρ^{n+1}) = (ρ^{n+1} - ρ^n) + dt·Θ(ρ^{n+1}) = 0

    Using Newton iteration:
        J·δρ = -G
        J = I + dt·(∂Θ/∂ρ)

    Args:
        mesh: SurfaceMesh instance
        metric: Metric providing geometric data
        center: Center of coordinate system
        spacing_factor: Stencil spacing factor for residual evaluation
        epsilon: Perturbation for Jacobian computation
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing_factor: float = 0.5,
        epsilon: float = 1e-5
    ):
        self.mesh = mesh
        self.metric = metric
        self.center = center
        self.N_s = mesh.N_s

        # Create vectorized residual evaluator and Jacobian computer
        self.residual_evaluator = create_vectorized_residual_evaluator(
            mesh, metric, center, spacing_factor
        )
        self.jacobian_computer = create_vectorized_jacobian_computer(
            mesh, metric, center, spacing_factor, epsilon
        )

        # Index mapping
        self._indices = mesh.independent_indices()
        self._n_independent = len(self._indices)

        # ILU preconditioner cache
        self._ilu_cache = None

    def _residual_to_grid(self, residual_flat: np.ndarray) -> np.ndarray:
        """Convert flat residual array to full (N_s, N_s) grid."""
        rho_grid = np.zeros((self.N_s, self.N_s))
        for k, (i, j) in enumerate(self._indices):
            rho_grid[i, j] = residual_flat[k]

        # Handle poles
        rho_grid[0, :] = rho_grid[0, 0]
        rho_grid[-1, :] = rho_grid[-1, 0]

        return rho_grid

    def _grid_to_flat(self, grid: np.ndarray) -> np.ndarray:
        """Convert (N_s, N_s) grid to flat array of independent values."""
        return np.array([grid[i, j] for i, j in self._indices])

    def _compute_theta(self, rho: np.ndarray) -> np.ndarray:
        """Compute expansion Θ at all independent grid points."""
        return self.residual_evaluator.evaluate(rho)

    def _compute_jacobian(self, rho: np.ndarray, verbose: bool = False) -> sparse.csr_matrix:
        """Compute sparse Jacobian ∂Θ/∂ρ."""
        return self.jacobian_computer.compute_sparse(rho, verbose=verbose)

    def step(
        self,
        rho_n: np.ndarray,
        dt: float,
        tol: float = 1e-8,
        max_newton_iter: int = 10,
        verbose: bool = False
    ) -> ImplicitStepResult:
        """
        Take one backward Euler step.

        Solves the implicit equation:
            G(ρ^{n+1}) = (ρ^{n+1} - ρ^n) + dt·Θ(ρ^{n+1}) = 0

        Using Newton iteration with Jacobian:
            J = I + dt·(∂Θ/∂ρ)

        Args:
            rho_n: Current surface shape (N_s, N_s)
            dt: Time step
            tol: Newton convergence tolerance
            max_newton_iter: Maximum Newton iterations per step

        Returns:
            ImplicitStepResult with updated surface
        """
        n = self._n_independent
        rho = rho_n.copy()
        rho_n_flat = self._grid_to_flat(rho_n)

        converged = False
        n_iter = 0

        for newton_iter in range(max_newton_iter):
            # Evaluate Θ at current position
            theta_flat = self._compute_theta(rho)

            # Compute implicit residual: G = (ρ - ρ_n) + dt·Θ
            rho_flat = self._grid_to_flat(rho)
            G = (rho_flat - rho_n_flat) + dt * theta_flat

            G_norm = np.linalg.norm(G)

            if verbose:
                print(f"    Newton iter {newton_iter}: ||G|| = {G_norm:.6e}")

            # Check convergence
            if G_norm < tol:
                converged = True
                n_iter = newton_iter + 1
                break

            # Compute Jacobian of Θ
            J_theta = self._compute_jacobian(rho, verbose=False)

            # Implicit Jacobian: J = I + dt·(∂Θ/∂ρ)
            J = sparse.eye(n, format='csr') + dt * J_theta

            # Solve J·δρ = -G
            try:
                # Try ILU-preconditioned iterative solve
                if self._ilu_cache is None or newton_iter == 0:
                    self._ilu_cache = spilu(J.tocsc(), drop_tol=1e-4)

                M = LinearOperator((n, n), matvec=self._ilu_cache.solve)
                delta_flat, info = bicgstab(J, -G, M=M, rtol=1e-10, maxiter=100)

                if info != 0:
                    # Fallback to direct solve
                    delta_flat = spsolve(J.tocsc(), -G)
            except Exception:
                # Direct solve as fallback
                delta_flat = spsolve(J.tocsc(), -G)

            # Update surface
            delta_grid = self._residual_to_grid(delta_flat)
            rho = rho + delta_grid

            # Enforce pole conditions
            rho[0, :] = rho[0, 0]
            rho[-1, :] = rho[-1, 0]

            n_iter = newton_iter + 1

        # Final Θ evaluation
        final_theta = self._compute_theta(rho)
        final_theta_norm = np.linalg.norm(final_theta)

        return ImplicitStepResult(
            rho=rho,
            n_newton_iters=n_iter,
            converged=converged,
            final_theta_norm=final_theta_norm
        )


@dataclass
class ImplicitLevelFlowResult:
    """Result of implicit Level Flow evolution."""
    rho: np.ndarray              # Final surface shape
    converged: bool              # Whether Θ → 0 was achieved
    n_steps: int                 # Number of time steps
    final_residual_norm: float   # Final ||Θ||
    total_newton_iters: int      # Total Newton iterations
    history: List[dict]          # Evolution history


class ImplicitLevelFlowFinder:
    """
    Find apparent horizons using implicit Level Flow.

    Uses backward Euler time stepping for unconditional stability,
    allowing large time steps that would cause explicit methods to diverge.

    Advantages:
    - Unconditionally stable (can use arbitrarily large dt)
    - Faster convergence to steady state
    - No CFL restriction

    Disadvantages:
    - Each step requires Newton iteration (more work per step)
    - Requires Jacobian computation

    Args:
        metric: Metric object providing geometric data
        N_s: Grid resolution (N_s x N_s grid)
        center: Center of coordinate system
    """

    def __init__(
        self,
        metric: Metric,
        N_s: int = 21,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.metric = metric
        self.N_s = N_s
        self.center = center

        self.mesh = SurfaceMesh(N_s)
        self.stepper = ImplicitLevelFlowStepper(
            self.mesh, metric, center
        )

        # Also keep a residual evaluator for computing Θ
        self.residual_evaluator = self.stepper.residual_evaluator

    def _compute_theta_grid(self, rho: np.ndarray) -> np.ndarray:
        """Compute expansion Θ on the full grid."""
        theta_flat = self.residual_evaluator.evaluate(rho)
        return self.stepper._residual_to_grid(theta_flat)

    def evolve(
        self,
        initial_radius: float = 2.0,
        initial_shape: Optional[np.ndarray] = None,
        dt: float = 1.0,
        t_final: float = 50.0,
        tol: float = 1e-8,
        max_steps: int = 1000,
        newton_tol: float = 1e-10,
        max_newton_iter: int = 10,
        save_history: bool = False,
        history_interval: int = 1,
        verbose: bool = True,
        callback: Optional[Callable] = None
    ) -> ImplicitLevelFlowResult:
        """
        Evolve surface toward apparent horizon using implicit stepping.

        Args:
            initial_radius: Starting radius (if initial_shape not provided)
            initial_shape: Custom initial surface shape (N_s, N_s)
            dt: Time step (can be large due to implicit stability)
            t_final: Maximum evolution time
            tol: Convergence tolerance for ||Θ||
            max_steps: Maximum number of time steps
            newton_tol: Convergence tolerance for Newton iteration
            max_newton_iter: Maximum Newton iterations per step
            save_history: Whether to save evolution history
            history_interval: Steps between history snapshots
            verbose: Print progress
            callback: Optional callback(step, t, rho, theta, dt)

        Returns:
            ImplicitLevelFlowResult with final surface
        """
        # Initialize surface
        if initial_shape is not None:
            rho = initial_shape.copy()
        else:
            rho = np.full((self.N_s, self.N_s), initial_radius)

        # Initial expansion
        theta = self._compute_theta_grid(rho)
        residual_norm = np.linalg.norm(theta.ravel())

        history = []
        t = 0.0
        step = 0
        total_newton_iters = 0

        if verbose:
            print(f"Implicit Level Flow: Starting evolution")
            print(f"  dt = {dt}, N_s = {self.N_s}")
            print(f"  Initial ||Θ|| = {residual_norm:.6e}")

        while step < max_steps and t < t_final:
            # Check convergence
            if residual_norm < tol:
                if verbose:
                    print(f"  Converged at step {step}, t = {t:.4f}")
                break

            # Take implicit step
            result = self.stepper.step(
                rho, dt,
                tol=newton_tol,
                max_newton_iter=max_newton_iter,
                verbose=False
            )

            if not result.converged:
                warnings.warn(f"Newton iteration did not converge at step {step}")

            rho = result.rho
            total_newton_iters += result.n_newton_iters
            t += dt
            step += 1

            # Update residual norm
            theta = self._compute_theta_grid(rho)
            residual_norm = np.linalg.norm(theta.ravel())

            # Callback
            if callback is not None:
                callback(step, t, rho, theta, dt)

            # Save history
            if save_history and step % history_interval == 0:
                history.append({
                    'step': step,
                    't': t,
                    'dt': dt,
                    'residual_norm': residual_norm,
                    'newton_iters': result.n_newton_iters,
                    'rho_mean': np.mean(rho),
                    'rho_min': np.min(rho),
                    'rho_max': np.max(rho)
                })

            # Progress
            if verbose and step % 10 == 0:
                print(f"  Step {step}: t = {t:.4f}, ||Θ|| = {residual_norm:.6e}, "
                      f"Newton iters = {result.n_newton_iters}")

        converged = residual_norm < tol

        if verbose:
            status = "CONVERGED" if converged else "NOT CONVERGED"
            print(f"  {status} after {step} steps")
            print(f"  Final ||Θ|| = {residual_norm:.6e}")
            print(f"  Total Newton iterations: {total_newton_iters}")
            print(f"  Final ρ: mean = {np.mean(rho):.4f}, "
                  f"range = [{np.min(rho):.4f}, {np.max(rho):.4f}]")

        return ImplicitLevelFlowResult(
            rho=rho,
            converged=converged,
            n_steps=step,
            final_residual_norm=residual_norm,
            total_newton_iters=total_newton_iters,
            history=history
        )

    def find(
        self,
        initial_radius: float = 2.0,
        dt: float = 1.0,
        tol: float = 1e-8,
        **kwargs
    ) -> np.ndarray:
        """
        Find apparent horizon (simplified interface).

        Args:
            initial_radius: Starting radius
            dt: Time step
            tol: Convergence tolerance

        Returns:
            Final surface shape ρ(θ, φ)
        """
        result = self.evolve(
            initial_radius=initial_radius,
            dt=dt,
            tol=tol,
            verbose=False,
            **kwargs
        )
        return result.rho

    def find_hybrid(
        self,
        initial_radius: float = 2.0,
        flow_dt: float = 1.0,
        flow_tol: float = 1e-1,
        newton_tol: float = 1e-8,
        max_flow_steps: int = 100,
        verbose: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Hybrid method: Implicit Level Flow to get close, then Newton to converge.

        This combines the robustness of Level Flow with the fast convergence
        of Newton's method near the solution.

        Args:
            initial_radius: Starting radius
            flow_dt: Time step for Level Flow
            flow_tol: Stop Level Flow when ||Θ|| < flow_tol
            newton_tol: Final tolerance for Newton's method
            max_flow_steps: Maximum Level Flow steps
            verbose: Print progress

        Returns:
            (rho, info): Final surface and info dict
        """
        import time
        from ..finder import ApparentHorizonFinder

        info = {}

        # Phase 1: Implicit Level Flow
        if verbose:
            print("Phase 1: Implicit Level Flow (finding approximate location)")

        t0 = time.perf_counter()
        flow_result = self.evolve(
            initial_radius=initial_radius,
            dt=flow_dt,
            tol=flow_tol,
            max_steps=max_flow_steps,
            verbose=verbose
        )
        t1 = time.perf_counter()

        info['flow_time'] = t1 - t0
        info['flow_steps'] = flow_result.n_steps
        info['flow_residual'] = flow_result.final_residual_norm
        info['flow_newton_iters'] = flow_result.total_newton_iters

        if verbose:
            print(f"\n  Level Flow: {flow_result.n_steps} steps, "
                  f"||Θ|| = {flow_result.final_residual_norm:.2e}")

        # Phase 2: Newton
        if verbose:
            print("\nPhase 2: Newton's method (precise convergence)")

        newton_finder = ApparentHorizonFinder(
            self.metric,
            N_s=self.N_s,
            center=self.center,
            use_vectorized_jacobian=True
        )

        t0 = time.perf_counter()
        try:
            rho_final = newton_finder.find(
                initial_guess=flow_result.rho,
                tol=newton_tol,
                max_iter=30
            )
            info['newton_converged'] = True
        except RuntimeError as e:
            if verbose:
                print(f"  Newton failed: {e}")
            rho_final = flow_result.rho
            info['newton_converged'] = False
        t1 = time.perf_counter()

        info['newton_time'] = t1 - t0
        info['total_time'] = info['flow_time'] + info['newton_time']

        if verbose:
            print(f"\n  Newton time: {info['newton_time']:.2f} s")
            print(f"  Total time: {info['total_time']:.2f} s")
            print(f"  Final mean radius: {np.mean(rho_final):.6f}")

        return rho_final, info
