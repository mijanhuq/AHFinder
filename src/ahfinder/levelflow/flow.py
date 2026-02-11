"""
Level Flow evolution for finding apparent horizons.

Implements the parabolic PDE:
    ∂ρ/∂t = -Θ

where Θ is the expansion of outgoing null normals.
The surface flows toward Θ = 0 (the apparent horizon).

Reference: Shoemaker, Huq & Matzner (2000) - arXiv:gr-qc/0004062
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from ..surface import SurfaceMesh
from ..metrics.base import Metric
from ..residual_vectorized import create_vectorized_residual_evaluator
from ..finder import ApparentHorizonFinder


@dataclass
class LevelFlowResult:
    """Result of Level Flow evolution."""
    rho: np.ndarray              # Final surface shape (N_s, N_s)
    converged: bool              # Whether Θ → 0 was achieved
    n_steps: int                 # Number of time steps taken
    final_residual_norm: float   # Final ||Θ||
    history: List[dict]          # History of evolution (optional)


class LevelFlowFinder:
    """
    Find apparent horizons using the Level Flow method.

    The Level Flow method evolves a surface according to:
        ∂ρ/∂t = -Θ

    where Θ is the expansion scalar. The surface flows toward
    regions where Θ = 0 (the apparent horizon).

    Advantages over Newton's method:
    - More robust to initial guess
    - Can find multiple horizons
    - Handles topological changes

    Disadvantages:
    - Slower convergence (O(1/dt) vs O(n_iter))
    - Requires choosing time step

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

        # Create mesh and residual evaluator (shared with Newton method)
        self.mesh = SurfaceMesh(N_s)
        self.residual_evaluator = create_vectorized_residual_evaluator(
            self.mesh, metric, center
        )

        # Index mapping for converting between flat residual and full grid
        self._indices = self.mesh.independent_indices()
        self._n_independent = len(self._indices)

    def _residual_to_grid(self, residual_flat: np.ndarray) -> np.ndarray:
        """Convert flat residual array to full (N_s, N_s) grid."""
        rho_grid = np.zeros((self.N_s, self.N_s))
        for k, (i, j) in enumerate(self._indices):
            rho_grid[i, j] = residual_flat[k]

        # Handle poles (same value around each pole)
        rho_grid[0, :] = rho_grid[0, 0]
        rho_grid[-1, :] = rho_grid[-1, 0]

        return rho_grid

    def _compute_theta_grid(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute expansion Θ on the full grid.

        Args:
            rho: Surface shape (N_s, N_s)

        Returns:
            Θ on full grid (N_s, N_s)
        """
        theta_flat = self.residual_evaluator.evaluate(rho)
        return self._residual_to_grid(theta_flat)

    def _compute_dt(
        self,
        rho: np.ndarray,
        theta: np.ndarray,
        cfl: float = 0.1
    ) -> float:
        """
        Compute adaptive time step based on CFL condition.

        Args:
            rho: Current surface shape
            theta: Current expansion field
            cfl: CFL number (stability factor)

        Returns:
            Time step dt
        """
        # Maximum "speed" - use RMS instead of max for stability
        rms_theta = np.sqrt(np.mean(theta**2))

        if rms_theta < 1e-10:
            return 0.1  # Already converged

        # Simple time step: limit change to 1% of mean radius per step
        rho_mean = np.mean(rho)
        dt = cfl * rho_mean / (rms_theta + 1e-10)

        # Hard cap on dt
        dt = min(dt, 0.1)

        return dt

    def _smooth_surface(self, rho: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Apply light smoothing to prevent high-frequency oscillations.

        Uses simple averaging with neighbors.
        """
        rho_smooth = rho.copy()
        N = self.N_s

        # Interior points only
        for i in range(1, N - 1):
            for j in range(N):
                jp = (j + 1) % N
                jm = (j - 1) % N

                # Average of 4 neighbors
                avg = 0.25 * (rho[i-1, j] + rho[i+1, j] + rho[i, jp] + rho[i, jm])
                rho_smooth[i, j] = (1 - alpha) * rho[i, j] + alpha * avg

        # Poles stay as is (they're already enforced to be uniform)
        return rho_smooth

    def _regularized_velocity(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute regularized flow velocity to prevent instability.

        Uses: v = -Θ / (1 + |Θ|) to bound velocity magnitude.
        """
        return -theta / (1.0 + np.abs(theta))

    def _rk4_step(
        self,
        rho: np.ndarray,
        dt: float,
        regularize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take one RK4 time step.

        Solves: ∂ρ/∂t = v(Θ) where v is optionally regularized.

        Args:
            rho: Current surface shape
            dt: Time step
            regularize: Use regularized velocity

        Returns:
            (new_rho, theta): Updated surface and final expansion
        """
        # k1
        theta1 = self._compute_theta_grid(rho)
        if regularize:
            k1 = self._regularized_velocity(theta1)
        else:
            k1 = -theta1

        # k2
        rho2 = rho + 0.5 * dt * k1
        theta2 = self._compute_theta_grid(rho2)
        if regularize:
            k2 = self._regularized_velocity(theta2)
        else:
            k2 = -theta2

        # k3
        rho3 = rho + 0.5 * dt * k2
        theta3 = self._compute_theta_grid(rho3)
        if regularize:
            k3 = self._regularized_velocity(theta3)
        else:
            k3 = -theta3

        # k4
        rho4 = rho + dt * k3
        theta4 = self._compute_theta_grid(rho4)
        if regularize:
            k4 = self._regularized_velocity(theta4)
        else:
            k4 = -theta4

        # Combine
        rho_new = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Apply smoothing to prevent high-frequency oscillations
        rho_new = self._smooth_surface(rho_new, alpha=0.05)

        # Enforce pole conditions
        rho_new[0, :] = rho_new[0, 0]
        rho_new[-1, :] = rho_new[-1, 0]

        # Evaluate Θ at new position
        theta_new = self._compute_theta_grid(rho_new)

        return rho_new, theta_new

    def _euler_step(
        self,
        rho: np.ndarray,
        dt: float,
        regularize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take one forward Euler step.

        Args:
            rho: Current surface shape
            dt: Time step
            regularize: Use regularized velocity

        Returns:
            (new_rho, theta): Updated surface and expansion
        """
        theta = self._compute_theta_grid(rho)

        if regularize:
            velocity = self._regularized_velocity(theta)
        else:
            velocity = -theta

        rho_new = rho + dt * velocity

        # Apply smoothing
        rho_new = self._smooth_surface(rho_new, alpha=0.05)

        # Enforce pole conditions
        rho_new[0, :] = rho_new[0, 0]
        rho_new[-1, :] = rho_new[-1, 0]

        theta_new = self._compute_theta_grid(rho_new)
        return rho_new, theta_new

    def evolve(
        self,
        initial_radius: float = 2.0,
        initial_shape: Optional[np.ndarray] = None,
        t_final: float = 50.0,
        tol: float = 1e-8,
        max_steps: int = 10000,
        cfl: float = 0.1,
        method: str = 'euler',
        regularize: bool = True,
        save_history: bool = False,
        history_interval: int = 10,
        verbose: bool = True,
        callback: Optional[Callable] = None
    ) -> LevelFlowResult:
        """
        Evolve surface toward apparent horizon.

        Args:
            initial_radius: Starting radius (if initial_shape not provided)
            initial_shape: Custom initial surface shape (N_s, N_s)
            t_final: Maximum evolution time
            tol: Convergence tolerance for ||Θ||
            max_steps: Maximum number of time steps
            cfl: CFL number for time step selection
            method: Integration method ('euler' or 'rk4')
            regularize: Use regularized velocity v=-Θ/(1+|Θ|) for stability
            save_history: Whether to save evolution history
            history_interval: Steps between history snapshots
            verbose: Print progress
            callback: Optional callback(step, t, rho, theta, dt) called each step

        Returns:
            LevelFlowResult with final surface and convergence info
        """
        # Initialize surface
        if initial_shape is not None:
            rho = initial_shape.copy()
        else:
            rho = np.full((self.N_s, self.N_s), initial_radius)

        # Choose integration method
        if method == 'euler':
            step_fn = lambda rho, dt: self._euler_step(rho, dt, regularize)
        elif method == 'rk4':
            step_fn = lambda rho, dt: self._rk4_step(rho, dt, regularize)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Initial expansion
        theta = self._compute_theta_grid(rho)
        residual_norm = np.linalg.norm(theta.ravel())

        history = []
        t = 0.0
        step = 0

        if verbose:
            print(f"Level Flow: Starting evolution")
            print(f"  Initial ||Θ|| = {residual_norm:.6e}")

        # Main evolution loop
        while step < max_steps and t < t_final:
            # Check convergence
            if residual_norm < tol:
                if verbose:
                    print(f"  Converged at step {step}, t = {t:.4f}")
                break

            # Compute adaptive time step
            dt = self._compute_dt(rho, theta, cfl)

            # Don't overshoot t_final
            if t + dt > t_final:
                dt = t_final - t

            # Take step
            rho_new, theta_new = step_fn(rho, dt)

            # Update
            rho = rho_new
            theta = theta_new
            t += dt
            step += 1
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
                    'rho_mean': np.mean(rho),
                    'rho_min': np.min(rho),
                    'rho_max': np.max(rho),
                    'theta_min': np.min(theta),
                    'theta_max': np.max(theta)
                })

            # Progress
            if verbose and step % 100 == 0:
                print(f"  Step {step}: t = {t:.4f}, dt = {dt:.4e}, ||Θ|| = {residual_norm:.6e}")

        converged = residual_norm < tol

        if verbose:
            status = "CONVERGED" if converged else "NOT CONVERGED"
            print(f"  {status} after {step} steps")
            print(f"  Final ||Θ|| = {residual_norm:.6e}")
            print(f"  Final ρ: mean = {np.mean(rho):.4f}, "
                  f"range = [{np.min(rho):.4f}, {np.max(rho):.4f}]")

        return LevelFlowResult(
            rho=rho,
            converged=converged,
            n_steps=step,
            final_residual_norm=residual_norm,
            history=history
        )

    def find(
        self,
        initial_radius: float = 2.0,
        tol: float = 1e-8,
        **kwargs
    ) -> np.ndarray:
        """
        Find apparent horizon (simplified interface matching Newton finder).

        Args:
            initial_radius: Starting radius
            tol: Convergence tolerance

        Returns:
            Final surface shape ρ(θ, φ)
        """
        result = self.evolve(
            initial_radius=initial_radius,
            tol=tol,
            verbose=False,
            **kwargs
        )
        return result.rho

    def find_hybrid(
        self,
        initial_radius: float = 2.0,
        flow_tol: float = 1e-1,
        newton_tol: float = 1e-8,
        max_flow_steps: int = 1000,
        verbose: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Hybrid method: Level Flow to get close, then Newton to converge.

        This combines the robustness of Level Flow (works from any initial guess)
        with the speed of Newton's method (quadratic convergence near solution).

        Args:
            initial_radius: Starting radius for Level Flow
            flow_tol: Stop Level Flow when ||Θ|| < flow_tol
            newton_tol: Final tolerance for Newton's method
            max_flow_steps: Maximum Level Flow steps
            verbose: Print progress

        Returns:
            (rho, info): Final surface and info dict with timings
        """
        import time

        info = {}

        # Phase 1: Level Flow to get approximate location
        if verbose:
            print("Phase 1: Level Flow (finding approximate location)")

        t0 = time.perf_counter()
        flow_result = self.evolve(
            initial_radius=initial_radius,
            tol=flow_tol,
            max_steps=max_flow_steps,
            verbose=verbose
        )
        t1 = time.perf_counter()

        info['flow_time'] = t1 - t0
        info['flow_steps'] = flow_result.n_steps
        info['flow_residual'] = flow_result.final_residual_norm

        if verbose:
            print(f"\n  Level Flow: {flow_result.n_steps} steps, "
                  f"||Θ|| = {flow_result.final_residual_norm:.2e}")
            print(f"  Mean radius: {np.mean(flow_result.rho):.4f}")

        # Phase 2: Newton to converge precisely
        if verbose:
            print("\nPhase 2: Newton's method (precise convergence)")

        # Create Newton finder with same parameters
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
