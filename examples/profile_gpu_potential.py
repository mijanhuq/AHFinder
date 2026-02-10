#!/usr/bin/env python3
"""
Profile AHFinder to evaluate GPU acceleration potential.

Measures time spent in different components and estimates parallelism.
"""

import numpy as np
import time
from functools import wraps

# Timing storage
timings = {}

def timed(name):
    """Decorator to time function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if name not in timings:
                timings[name] = {'total': 0, 'calls': 0}
            timings[name]['total'] += elapsed
            timings[name]['calls'] += 1
            return result
        return wrapper
    return decorator

# Import after defining decorator
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.boosted_kerr_fast import FastBoostedKerrMetric
from ahfinder.metrics import SchwarzschildMetric
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast

def profile_single_find(metric, N_s=25, initial_radius=2.0):
    """Profile a single horizon find."""
    finder = ApparentHorizonFinder(metric, N_s=N_s)

    # Time the full find
    start = time.perf_counter()
    rho = finder.find(initial_radius=initial_radius, tol=1e-5, max_iter=20)
    total_time = time.perf_counter() - start

    return total_time, rho

def profile_components(metric, N_s=25):
    """Profile individual components."""
    from ahfinder.surface import SurfaceMesh
    from ahfinder.stencil import CartesianStencil
    from ahfinder.residual import compute_residual
    from ahfinder.jacobian import compute_jacobian_dense

    # Create surface
    surface = SurfaceMesh(N_s)
    rho = np.ones(surface.N_points) * 2.0

    # Time mesh update
    start = time.perf_counter()
    for _ in range(10):
        surface.update(rho)
    mesh_time = (time.perf_counter() - start) / 10

    # Time stencil computation (includes interpolation)
    stencil = CartesianStencil(surface)
    start = time.perf_counter()
    for _ in range(10):
        stencil.compute(rho)
    stencil_time = (time.perf_counter() - start) / 10

    # Time residual computation
    start = time.perf_counter()
    for _ in range(10):
        F = compute_residual(surface, stencil, metric)
    residual_time = (time.perf_counter() - start) / 10

    # Time Jacobian computation
    start = time.perf_counter()
    for _ in range(3):
        J = compute_jacobian_dense(surface, stencil, metric, rho)
    jacobian_time = (time.perf_counter() - start) / 3

    return {
        'mesh_update': mesh_time,
        'stencil': stencil_time,
        'residual': residual_time,
        'jacobian': jacobian_time,
    }

def profile_metric_calls(metric, N_s=25):
    """Profile metric evaluation overhead."""
    from ahfinder.surface import SurfaceMesh

    surface = SurfaceMesh(N_s)
    rho = np.ones(surface.N_points) * 2.0
    surface.update(rho)

    # Get sample points
    points = surface.xyz  # (N_points, 3)

    # Time individual metric calls
    n_calls = 100

    # Single point evaluation
    x, y, z = points[0]
    start = time.perf_counter()
    for _ in range(n_calls):
        gamma = metric.spatial_metric(x, y, z)
    single_gamma_time = (time.perf_counter() - start) / n_calls

    start = time.perf_counter()
    for _ in range(n_calls):
        dgamma = metric.spatial_metric_derivative(x, y, z)
    single_dgamma_time = (time.perf_counter() - start) / n_calls

    start = time.perf_counter()
    for _ in range(n_calls):
        K = metric.extrinsic_curvature(x, y, z)
    single_K_time = (time.perf_counter() - start) / n_calls

    # All points evaluation (simulating batch)
    start = time.perf_counter()
    for _ in range(10):
        for i in range(len(points)):
            x, y, z = points[i]
            gamma = metric.spatial_metric(x, y, z)
            dgamma = metric.spatial_metric_derivative(x, y, z)
            K = metric.extrinsic_curvature(x, y, z)
    all_points_time = (time.perf_counter() - start) / 10

    return {
        'single_gamma': single_gamma_time,
        'single_dgamma': single_dgamma_time,
        'single_K': single_K_time,
        'all_points': all_points_time,
        'n_points': len(points),
    }

def estimate_gpu_speedup(component_times, metric_times):
    """Estimate potential GPU speedup."""

    # GPU assumptions:
    # - Memory transfer overhead: ~0.1-1ms per transfer
    # - GPU kernel launch overhead: ~10-50 microseconds
    # - GPU can parallelize across all points
    # - Typical GPU speedup for parallel math: 10-100x per operation

    n_points = metric_times['n_points']

    # Current sequential time for metric at all points
    current_metric_time = metric_times['all_points']

    # GPU estimate: parallelize across points, but add transfer overhead
    # Assume 20x speedup on compute, but 0.5ms transfer overhead
    transfer_overhead = 0.0005  # 0.5 ms
    gpu_compute_speedup = 20  # Conservative estimate

    gpu_metric_time = current_metric_time / gpu_compute_speedup + transfer_overhead

    # Jacobian: currently N_points evaluations, each with perturbations
    # This is embarrassingly parallel
    current_jacobian = component_times['jacobian']
    gpu_jacobian_time = current_jacobian / gpu_compute_speedup + transfer_overhead * 2

    # Stencil/interpolation: harder to GPU-ify due to scipy
    # Would need custom CUDA implementation
    current_stencil = component_times['stencil']
    # Optimistic: 5x speedup with custom GPU spline
    gpu_stencil_time = current_stencil / 5 + transfer_overhead

    # Linear solve: stays on CPU (or use cuBLAS)
    # For small matrices (N_points x N_points), CPU is often faster

    return {
        'metric': {
            'current': current_metric_time,
            'gpu_estimate': gpu_metric_time,
            'speedup': current_metric_time / gpu_metric_time if gpu_metric_time > 0 else 0,
        },
        'jacobian': {
            'current': current_jacobian,
            'gpu_estimate': gpu_jacobian_time,
            'speedup': current_jacobian / gpu_jacobian_time if gpu_jacobian_time > 0 else 0,
        },
        'stencil': {
            'current': current_stencil,
            'gpu_estimate': gpu_stencil_time,
            'speedup': current_stencil / gpu_stencil_time if gpu_stencil_time > 0 else 0,
        },
    }

def main():
    print("=" * 70)
    print("AHFinder GPU Acceleration Potential Analysis")
    print("=" * 70)

    # Test configurations
    configs = [
        ("Schwarzschild (fast)", SchwarzschildMetricFast(M=1.0), 2.0),
        ("Boosted Kerr (a=0.5, v=0.3)",
         FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.3, 0.0, 0.0])), 1.9),
    ]

    for N_s in [17, 25, 33]:
        n_points = N_s * N_s
        print(f"\n{'=' * 70}")
        print(f"N_s = {N_s} ({n_points} points)")
        print("=" * 70)

        for name, metric, r0 in configs:
            print(f"\n{name}:")
            print("-" * 50)

            # Warm up JIT
            _ = profile_single_find(metric, N_s=N_s, initial_radius=r0)

            # Profile components
            comp_times = profile_components(metric, N_s=N_s)
            metric_times = profile_metric_calls(metric, N_s=N_s)

            # Full find timing
            total_time, _ = profile_single_find(metric, N_s=N_s, initial_radius=r0)

            print(f"\nComponent breakdown (per iteration):")
            print(f"  Mesh update:    {comp_times['mesh_update']*1000:8.3f} ms")
            print(f"  Stencil:        {comp_times['stencil']*1000:8.3f} ms")
            print(f"  Residual:       {comp_times['residual']*1000:8.3f} ms")
            print(f"  Jacobian:       {comp_times['jacobian']*1000:8.3f} ms")

            iter_total = sum(comp_times.values())
            print(f"  --------------------------")
            print(f"  Iteration sum:  {iter_total*1000:8.3f} ms")

            print(f"\nMetric evaluation:")
            print(f"  Single γ_ij:    {metric_times['single_gamma']*1e6:8.2f} μs")
            print(f"  Single ∂γ_ij:   {metric_times['single_dgamma']*1e6:8.2f} μs")
            print(f"  Single K_ij:    {metric_times['single_K']*1e6:8.2f} μs")
            print(f"  All {n_points} points: {metric_times['all_points']*1000:8.3f} ms")

            # GPU estimates
            gpu_est = estimate_gpu_speedup(comp_times, metric_times)

            print(f"\nGPU speedup estimates:")
            print(f"  {'Component':<15} {'Current':>10} {'GPU Est':>10} {'Speedup':>10}")
            print(f"  {'-'*45}")
            for comp, data in gpu_est.items():
                print(f"  {comp:<15} {data['current']*1000:>9.2f}ms {data['gpu_estimate']*1000:>9.2f}ms {data['speedup']:>9.1f}x")

            # Overall estimate
            current_total = sum(d['current'] for d in gpu_est.values())
            gpu_total = sum(d['gpu_estimate'] for d in gpu_est.values())
            overall_speedup = current_total / gpu_total if gpu_total > 0 else 0

            print(f"  {'-'*45}")
            print(f"  {'TOTAL':<15} {current_total*1000:>9.2f}ms {gpu_total*1000:>9.2f}ms {overall_speedup:>9.1f}x")

            print(f"\n  Full find time: {total_time*1000:.1f} ms")

    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    print("""
Key findings:

1. MEMORY TRANSFER OVERHEAD: GPU transfers cost ~0.5-1ms each.
   For small problems (N_s ≤ 25), this can dominate.

2. PARALLELISM:
   - Metric evaluation at N points: Embarrassingly parallel ✓
   - Jacobian computation: Embarrassingly parallel ✓
   - Stencil/interpolation: Requires custom CUDA splines

3. BOTTLENECKS:
   - If Jacobian dominates → GPU helps significantly
   - If stencil dominates → Need custom GPU interpolation
   - If linear solve dominates → cuBLAS may help for large N_s

4. BREAK-EVEN POINT:
   - GPU likely beneficial only for N_s ≥ 33 (1000+ points)
   - For N_s = 17-25, CPU with Numba is likely faster

5. IMPLEMENTATION EFFORT:
   - Metric batching: Medium (rewrite to process arrays)
   - Jacobian GPU: Medium (parallelize across columns)
   - Interpolation GPU: High (custom CUDA spline)
""")

if __name__ == "__main__":
    main()
