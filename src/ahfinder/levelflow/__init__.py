"""
Level Flow method for finding apparent horizons.

This module implements the Level Flow algorithm from:
Shoemaker, Huq & Matzner (2000) - arXiv:gr-qc/0004062

The Level Flow method evolves a surface toward the apparent horizon
by following the gradient of the expansion scalar Θ:

    ∂ρ/∂t = -Θ

This approach:
- Is robust to initial guess variations
- Can find multiple apparent horizons
- Handles topological changes (horizons merging/splitting)

Available classes:

- LevelFlowFinder: Explicit time stepping (original method)
- ImplicitLevelFlowFinder: Implicit time stepping (stable, large dt)
- MultiHorizonFinder: Multi-surface tracking with topology detection

Usage:

    # Simple explicit flow
    from ahfinder.levelflow import LevelFlowFinder
    finder = LevelFlowFinder(metric, N_s=21)
    rho, history = finder.evolve(initial_radius=3.0, t_final=10.0)

    # Implicit flow (stable with large time steps)
    from ahfinder.levelflow import ImplicitLevelFlowFinder
    finder = ImplicitLevelFlowFinder(metric, N_s=21)
    rho = finder.find(initial_radius=3.0, dt=1.0)

    # Multi-horizon finding
    from ahfinder.levelflow import MultiHorizonFinder
    finder = MultiHorizonFinder(metric, N_s=21)
    horizons = finder.find_all(initial_radius=3.0)

    # Topology detection only
    from ahfinder.levelflow import detect_horizon_topology
    candidates = detect_horizon_topology(metric, r_range=(0.5, 5.0))
"""

from .flow import LevelFlowFinder, LevelFlowResult

from .implicit import (
    ImplicitLevelFlowFinder,
    ImplicitLevelFlowResult,
    ImplicitLevelFlowStepper,
    ImplicitStepResult
)

from .regularization import (
    smooth_surface_average,
    regularized_velocity,
    SurfaceSmoother,
    MeanCurvatureRegularizer,  # Alias for backwards compatibility
    estimate_optimal_epsilon
)

from .topology import (
    TopologyDetector,
    HorizonCandidate,
    detect_horizon_topology
)

from .multi_surface import (
    MultiSurfaceLevelFlow,
    MultiSurfaceResult,
    MultiHorizonFinder
)

__all__ = [
    # Original explicit method
    'LevelFlowFinder',
    'LevelFlowResult',

    # Implicit time stepping
    'ImplicitLevelFlowFinder',
    'ImplicitLevelFlowResult',
    'ImplicitLevelFlowStepper',
    'ImplicitStepResult',

    # Smoothing/Regularization
    'smooth_surface_average',
    'regularized_velocity',
    'SurfaceSmoother',
    'MeanCurvatureRegularizer',  # Alias for SurfaceSmoother
    'estimate_optimal_epsilon',

    # Topology detection
    'TopologyDetector',
    'HorizonCandidate',
    'detect_horizon_topology',

    # Multi-surface tracking
    'MultiSurfaceLevelFlow',
    'MultiSurfaceResult',
    'MultiHorizonFinder',
]
