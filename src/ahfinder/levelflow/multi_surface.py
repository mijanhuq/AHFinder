"""
Multi-surface Level Flow for tracking multiple apparent horizons.

Tracks multiple horizon components through Level Flow evolution,
handling topology changes (mergers, splits) as they occur.

This is useful for:
- Binary black hole mergers (two horizons → one)
- Finding all horizons in multi-body systems
- Tracking horizon evolution through topology changes

Reference: Shoemaker, Huq & Matzner (2000) - arXiv:gr-qc/0004062
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import warnings

from ..surface import SurfaceMesh
from ..metrics.base import Metric
from .implicit import ImplicitLevelFlowStepper, ImplicitStepResult
from .topology import TopologyDetector, HorizonCandidate, detect_horizon_topology
from .regularization import SurfaceSmoother


@dataclass
class SurfaceState:
    """State of a single tracked surface."""
    rho: np.ndarray              # Current shape ρ(θ, φ)
    surface_id: int              # Unique identifier
    parent_id: Optional[int]     # ID of parent (if split from another)
    theta_norm: float            # Current ||Θ||
    mean_radius: float           # Current mean radius
    is_active: bool = True       # Whether this surface is still evolving
    history: List[dict] = field(default_factory=list)


@dataclass
class MultiSurfaceResult:
    """Result of multi-surface Level Flow evolution."""
    surfaces: List[np.ndarray]   # Final surface shapes
    converged: List[bool]        # Convergence status for each
    n_steps: int                 # Total evolution steps
    n_topology_changes: int      # Number of splits/mergers detected
    final_n_surfaces: int        # Number of surfaces at end
    history: List[dict]          # Evolution history


class MultiSurfaceLevelFlow:
    """
    Multi-surface Level Flow finder for multiple apparent horizons.

    Tracks multiple surfaces through Level Flow evolution, detecting
    topology changes (mergers/splits) and adjusting accordingly.

    Algorithm:
    1. Start with initial guess(es) or detect topology automatically
    2. Evolve each surface using implicit Level Flow
    3. Periodically check for topology changes
    4. Handle mergers (combine surfaces) or splits (spawn new surfaces)
    5. Continue until all surfaces converge or max time reached

    Args:
        metric: Metric providing geometric data
        N_s: Grid resolution (N_s x N_s for each surface)
        center: Center of coordinate system
        topology_check_interval: Steps between topology checks
    """

    def __init__(
        self,
        metric: Metric,
        N_s: int = 21,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        topology_check_interval: int = 10
    ):
        self.metric = metric
        self.N_s = N_s
        self.center = center
        self.topology_check_interval = topology_check_interval

        # Shared mesh and stepper
        self.mesh = SurfaceMesh(N_s)
        self.stepper = ImplicitLevelFlowStepper(
            self.mesh, metric, center
        )
        self.smoother = SurfaceSmoother(self.mesh)

        # Surface tracking
        self.surfaces: Dict[int, SurfaceState] = {}
        self._next_surface_id = 0

        # Topology detection
        self.topology_detector: Optional[TopologyDetector] = None

    def _create_surface(
        self,
        rho: np.ndarray,
        parent_id: Optional[int] = None
    ) -> int:
        """Create a new tracked surface and return its ID."""
        surface_id = self._next_surface_id
        self._next_surface_id += 1

        theta_flat = self.stepper.residual_evaluator.evaluate(rho)
        theta_norm = np.linalg.norm(theta_flat)

        self.surfaces[surface_id] = SurfaceState(
            rho=rho.copy(),
            surface_id=surface_id,
            parent_id=parent_id,
            theta_norm=theta_norm,
            mean_radius=np.mean(rho),
            is_active=True
        )

        return surface_id

    def _evolve_surface(
        self,
        surface_id: int,
        dt: float,
        newton_tol: float = 1e-10
    ) -> ImplicitStepResult:
        """Evolve a single surface one time step."""
        state = self.surfaces[surface_id]

        result = self.stepper.step(
            state.rho, dt,
            tol=newton_tol,
            max_newton_iter=10
        )

        # Update state
        state.rho = result.rho
        state.theta_norm = result.final_theta_norm
        state.mean_radius = np.mean(result.rho)

        return result

    def _check_surfaces_overlap(
        self,
        id1: int,
        id2: int,
        threshold: float = 0.1
    ) -> bool:
        """Check if two surfaces overlap significantly (indicating merger)."""
        rho1 = self.surfaces[id1].rho
        rho2 = self.surfaces[id2].rho

        # Compute relative difference
        diff = np.abs(rho1 - rho2)
        mean_r = 0.5 * (np.mean(rho1) + np.mean(rho2))

        rel_diff = np.max(diff) / mean_r

        return rel_diff < threshold

    def _merge_surfaces(self, id1: int, id2: int) -> int:
        """Merge two surfaces into one."""
        state1 = self.surfaces[id1]
        state2 = self.surfaces[id2]

        # Average the two surfaces
        merged_rho = 0.5 * (state1.rho + state2.rho)

        # Deactivate old surfaces
        state1.is_active = False
        state2.is_active = False

        # Create new merged surface
        new_id = self._create_surface(merged_rho, parent_id=id1)

        return new_id

    def _check_topology_changes(
        self,
        r_range: Tuple[float, float] = None,
        merge_threshold: float = 0.1
    ) -> List[Tuple[str, List[int]]]:
        """
        Check for topology changes (mergers, splits).

        Returns:
            List of (change_type, [affected_ids]) tuples
        """
        changes = []
        active_ids = [sid for sid, s in self.surfaces.items() if s.is_active]

        # Check for mergers: surfaces that have become very similar
        for i, id1 in enumerate(active_ids):
            for id2 in active_ids[i + 1:]:
                if self._check_surfaces_overlap(id1, id2, merge_threshold):
                    changes.append(('merge', [id1, id2]))

        # TODO: Check for splits using topology detection
        # This would require rebuilding the Θ field and checking
        # if a single surface has evolved to have multiple zero crossings

        return changes

    def _apply_topology_changes(
        self,
        changes: List[Tuple[str, List[int]]]
    ) -> int:
        """Apply detected topology changes. Returns number of changes made."""
        n_changes = 0

        for change_type, affected_ids in changes:
            if change_type == 'merge':
                if len(affected_ids) >= 2:
                    # Merge first two
                    id1, id2 = affected_ids[0], affected_ids[1]
                    if (self.surfaces[id1].is_active and
                        self.surfaces[id2].is_active):
                        self._merge_surfaces(id1, id2)
                        n_changes += 1

        return n_changes

    def initialize_from_topology(
        self,
        r_range: Tuple[float, float] = (0.5, 5.0),
        n_r: int = 30,
        verbose: bool = True
    ) -> int:
        """
        Initialize surfaces by detecting topology in the Θ field.

        Uses TopologyDetector to find all apparent horizons and
        creates a tracked surface for each.

        Args:
            r_range: Radial range to search
            n_r: Number of radial grid points
            verbose: Print progress

        Returns:
            Number of surfaces initialized
        """
        self.topology_detector = TopologyDetector(
            self.mesh, self.metric, self.center, r_range, n_r
        )

        candidates = self.topology_detector.find_horizons(verbose=verbose)

        for candidate in candidates:
            if candidate.is_valid:
                self._create_surface(candidate.rho)

        if verbose:
            print(f"Initialized {len(self.surfaces)} surface(s) from topology")

        return len(self.surfaces)

    def initialize_from_radii(
        self,
        radii: List[float],
        verbose: bool = True
    ) -> int:
        """
        Initialize surfaces as spheres at given radii.

        Args:
            radii: List of initial radii
            verbose: Print progress

        Returns:
            Number of surfaces initialized
        """
        for r in radii:
            rho = np.full((self.N_s, self.N_s), r)
            self._create_surface(rho)

        if verbose:
            print(f"Initialized {len(self.surfaces)} surface(s) at radii {radii}")

        return len(self.surfaces)

    def evolve(
        self,
        dt: float = 1.0,
        t_final: float = 50.0,
        tol: float = 1e-8,
        max_steps: int = 500,
        check_topology: bool = True,
        regularize: bool = True,
        regularization_strength: float = 0.01,
        verbose: bool = True
    ) -> MultiSurfaceResult:
        """
        Evolve all surfaces toward apparent horizons.

        Args:
            dt: Time step for implicit evolution
            t_final: Maximum evolution time
            tol: Convergence tolerance for ||Θ||
            max_steps: Maximum evolution steps
            check_topology: Whether to check for topology changes
            regularize: Apply mean curvature regularization
            regularization_strength: Strength of regularization
            verbose: Print progress

        Returns:
            MultiSurfaceResult with all final surfaces
        """
        if not self.surfaces:
            raise ValueError("No surfaces initialized. Call initialize_* first.")

        t = 0.0
        step = 0
        n_topology_changes = 0
        history = []

        if verbose:
            print(f"Multi-Surface Level Flow: {len(self.surfaces)} surface(s)")
            print(f"  dt = {dt}, tol = {tol}")

        while step < max_steps and t < t_final:
            active_surfaces = [
                sid for sid, s in self.surfaces.items() if s.is_active
            ]

            if not active_surfaces:
                if verbose:
                    print("All surfaces converged or deactivated")
                break

            # Check if all active surfaces have converged
            all_converged = all(
                self.surfaces[sid].theta_norm < tol
                for sid in active_surfaces
            )

            if all_converged:
                if verbose:
                    print(f"All surfaces converged at step {step}")
                break

            # Evolve each active surface
            for sid in active_surfaces:
                state = self.surfaces[sid]

                # Skip if already converged
                if state.theta_norm < tol:
                    continue

                # Take implicit step
                result = self._evolve_surface(sid, dt)

                # Apply smoothing if requested
                if regularize:
                    state.rho = self.smoother.smooth(state.rho, regularization_strength)
                    # Recompute theta after smoothing
                    theta_flat = self.stepper.residual_evaluator.evaluate(state.rho)
                    state.theta_norm = np.linalg.norm(theta_flat)

            # Check for topology changes periodically
            if check_topology and step % self.topology_check_interval == 0:
                changes = self._check_topology_changes()
                if changes:
                    n_applied = self._apply_topology_changes(changes)
                    n_topology_changes += n_applied
                    if verbose and n_applied > 0:
                        print(f"  Step {step}: Applied {n_applied} topology change(s)")

            t += dt
            step += 1

            # Record history
            history.append({
                'step': step,
                't': t,
                'n_active': len(active_surfaces),
                'theta_norms': {
                    sid: self.surfaces[sid].theta_norm
                    for sid in active_surfaces
                }
            })

            # Progress output
            if verbose and step % 10 == 0:
                norms = [self.surfaces[sid].theta_norm for sid in active_surfaces]
                print(f"  Step {step}: t = {t:.2f}, "
                      f"active = {len(active_surfaces)}, "
                      f"max ||Θ|| = {max(norms):.2e}")

        # Gather results
        final_active = [
            sid for sid, s in self.surfaces.items() if s.is_active
        ]
        final_surfaces = [self.surfaces[sid].rho for sid in final_active]
        final_converged = [
            self.surfaces[sid].theta_norm < tol for sid in final_active
        ]

        if verbose:
            print(f"\nEvolution complete:")
            print(f"  Final surfaces: {len(final_surfaces)}")
            print(f"  Converged: {sum(final_converged)} / {len(final_surfaces)}")
            print(f"  Topology changes: {n_topology_changes}")

        return MultiSurfaceResult(
            surfaces=final_surfaces,
            converged=final_converged,
            n_steps=step,
            n_topology_changes=n_topology_changes,
            final_n_surfaces=len(final_surfaces),
            history=history
        )

    def find_all(
        self,
        initial_radius: float = 3.0,
        r_range: Optional[Tuple[float, float]] = None,
        auto_detect: bool = True,
        **evolve_kwargs
    ) -> List[np.ndarray]:
        """
        Find all apparent horizons.

        This is the main entry point for multi-horizon finding.

        Args:
            initial_radius: Starting radius (if not auto-detecting)
            r_range: Radial range for topology detection
            auto_detect: Whether to auto-detect initial surfaces
            **evolve_kwargs: Additional arguments for evolve()

        Returns:
            List of final surface shapes ρ(θ, φ)

        Example:
            >>> finder = MultiSurfaceLevelFlow(metric, N_s=21)
            >>> horizons = finder.find_all(initial_radius=3.0)
            >>> print(f"Found {len(horizons)} horizon(s)")
        """
        # Initialize surfaces
        if auto_detect and r_range is not None:
            n_init = self.initialize_from_topology(r_range, verbose=True)
            if n_init == 0:
                # Fallback to single sphere
                warnings.warn("Auto-detection found no surfaces. Using sphere.")
                self.initialize_from_radii([initial_radius])
        else:
            self.initialize_from_radii([initial_radius])

        # Evolve
        result = self.evolve(**evolve_kwargs)

        return result.surfaces


class MultiHorizonFinder:
    """
    High-level interface for finding multiple apparent horizons.

    This class provides a simplified API that handles:
    - Automatic topology detection
    - Multi-surface evolution
    - Newton refinement for precise convergence

    Args:
        metric: Metric providing geometric data
        N_s: Grid resolution
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

    def find_all(
        self,
        initial_radius: float = 3.0,
        r_range: Tuple[float, float] = (0.5, 5.0),
        flow_tol: float = 1e-2,
        newton_tol: float = 1e-8,
        verbose: bool = True
    ) -> List[np.ndarray]:
        """
        Find all apparent horizons in the spacetime.

        Uses a three-phase approach:
        1. Topology detection to find initial surfaces
        2. Level Flow evolution to approach horizons
        3. Newton refinement for precise convergence

        Args:
            initial_radius: Fallback initial radius
            r_range: Radial range for topology search
            flow_tol: Tolerance for Level Flow phase
            newton_tol: Final Newton tolerance
            verbose: Print progress

        Returns:
            List of converged horizon surfaces ρ(θ, φ)
        """
        from ..finder import ApparentHorizonFinder

        # Phase 1: Detect topology
        if verbose:
            print("Phase 1: Topology detection")

        candidates = detect_horizon_topology(
            self.metric, self.N_s, self.center, r_range,
            verbose=verbose
        )

        if not candidates:
            if verbose:
                print("  No horizons detected. Trying single surface.")
            # Fallback: try single sphere
            initial_surfaces = [np.full((self.N_s, self.N_s), initial_radius)]
        else:
            initial_surfaces = [c.rho for c in candidates if c.is_valid]

        if verbose:
            print(f"  Initial surfaces: {len(initial_surfaces)}")

        # Phase 2: Level Flow evolution
        if verbose:
            print("\nPhase 2: Level Flow evolution")

        multi_flow = MultiSurfaceLevelFlow(
            self.metric, self.N_s, self.center
        )

        for rho in initial_surfaces:
            multi_flow._create_surface(rho)

        result = multi_flow.evolve(
            dt=1.0,
            tol=flow_tol,
            max_steps=100,
            verbose=verbose
        )

        # Phase 3: Newton refinement
        if verbose:
            print("\nPhase 3: Newton refinement")

        newton_finder = ApparentHorizonFinder(
            self.metric,
            N_s=self.N_s,
            center=self.center,
            use_vectorized_jacobian=True
        )

        final_horizons = []
        for i, rho in enumerate(result.surfaces):
            try:
                refined = newton_finder.find(
                    initial_guess=rho,
                    tol=newton_tol,
                    max_iter=30
                )
                final_horizons.append(refined)
                if verbose:
                    print(f"  Surface {i}: converged at r = {np.mean(refined):.4f}")
            except RuntimeError as e:
                warnings.warn(f"Newton refinement failed for surface {i}: {e}")
                final_horizons.append(rho)

        if verbose:
            print(f"\nFound {len(final_horizons)} horizon(s)")

        return final_horizons
