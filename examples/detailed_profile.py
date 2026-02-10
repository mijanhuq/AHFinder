#!/usr/bin/env python3
"""
Detailed function-level profiling of AHFinder.
"""

import cProfile
import pstats
import io
import numpy as np
from pstats import SortKey

def run_horizon_find():
    """Run a horizon find for profiling."""
    from ahfinder import ApparentHorizonFinder
    from ahfinder.metrics.boosted_kerr_fast import FastBoostedKerrMetric

    metric = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.3, 0.0, 0.0]))
    finder = ApparentHorizonFinder(metric, N_s=25)

    # Warm up JIT
    _ = finder.find(initial_radius=1.9, tol=1e-5, max_iter=5)

    # Profile this run
    rho = finder.find(initial_radius=1.9, tol=1e-5, max_iter=20)
    return rho

def profile_with_cprofile():
    """Run cProfile and print results."""
    print("=" * 80)
    print("Detailed Function Profile - FastBoostedKerrMetric (a=0.5, v=0.3), N_s=25")
    print("=" * 80)

    # First warm up
    print("\nWarming up JIT compilation...")
    run_horizon_find()

    # Now profile
    print("Profiling...")
    profiler = cProfile.Profile()
    profiler.enable()

    rho = run_horizon_find()

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)

    print("\n" + "=" * 80)
    print("Top 40 functions by CUMULATIVE time:")
    print("=" * 80)
    ps.print_stats(40)
    print(s.getvalue())

    # Also show by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats(SortKey.TIME)
    print("\n" + "=" * 80)
    print("Top 40 functions by TOTAL (internal) time:")
    print("=" * 80)
    ps2.print_stats(40)
    print(s2.getvalue())

    # Callers for key functions
    print("\n" + "=" * 80)
    print("Call graph for key functions:")
    print("=" * 80)

    s3 = io.StringIO()
    ps3 = pstats.Stats(profiler, stream=s3)

    # Find the top time consumers and show their callers
    ps3.print_callers('spatial_metric', 10)
    ps3.print_callers('extrinsic_curvature', 10)
    ps3.print_callers('compute_residual', 10)
    ps3.print_callers('__call__', 10)  # RectBivariateSpline
    print(s3.getvalue())

def profile_schwarzschild():
    """Profile Schwarzschild for comparison."""
    from ahfinder import ApparentHorizonFinder
    from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast

    print("\n" + "=" * 80)
    print("Comparison: SchwarzschildMetricFast, N_s=25")
    print("=" * 80)

    metric = SchwarzschildMetricFast(M=1.0)
    finder = ApparentHorizonFinder(metric, N_s=25)

    # Warm up
    _ = finder.find(initial_radius=2.5, tol=1e-5, max_iter=5)

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()

    rho = finder.find(initial_radius=2.5, tol=1e-5, max_iter=20)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.TIME)
    print("\nTop 30 functions by TOTAL time:")
    ps.print_stats(30)
    print(s.getvalue())

def main():
    profile_with_cprofile()
    profile_schwarzschild()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
To interpret these results:

- 'tottime': Time spent IN this function (excluding subcalls)
- 'cumtime': Time spent in this function AND all functions it calls
- 'ncalls': Number of times the function was called
- 'percall': Time per call

Key functions to look for:
- spatial_metric, spatial_metric_derivative, extrinsic_curvature: Metric evaluation
- compute_residual: Expansion Θ computation
- compute_jacobian_dense: Jacobian matrix construction
- RectBivariateSpline.__call__: Interpolation (often a bottleneck)
- np.linalg.solve: Linear system solve

GPU acceleration helps most where:
1. High 'tottime' AND high 'ncalls' (parallel opportunity)
2. Function is pure computation (not I/O or memory-bound)
""")

if __name__ == "__main__":
    main()
