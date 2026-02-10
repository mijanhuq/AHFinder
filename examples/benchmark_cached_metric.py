#!/usr/bin/env python3
"""
Benchmark cached vs non-cached metric computation.
"""

import numpy as np
import time

def benchmark_metric_calls():
    """Compare cached vs non-cached metric."""
    from ahfinder.metrics.boosted_kerr_fast import FastBoostedKerrMetric
    from ahfinder.metrics.boosted_kerr_fast_cached import FastBoostedKerrMetricCached

    print("=" * 70)
    print("Metric Computation Benchmark: Cached vs Non-Cached")
    print("=" * 70)

    # Create metrics
    velocity = np.array([0.3, 0.0, 0.0])
    metric_orig = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=velocity)
    metric_cached = FastBoostedKerrMetricCached(M=1.0, a=0.5, velocity=velocity)

    # Test points
    n_points = 10000
    np.random.seed(42)
    # Random points on a sphere of radius ~2
    theta = np.random.uniform(0.1, np.pi - 0.1, n_points)
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    r = np.random.uniform(1.8, 2.2, n_points)
    x_pts = r * np.sin(theta) * np.cos(phi)
    y_pts = r * np.sin(theta) * np.sin(phi)
    z_pts = r * np.cos(theta)

    # Warm up JIT
    print("\nWarming up JIT...")
    for i in range(100):
        x, y, z = x_pts[i], y_pts[i], z_pts[i]
        _ = metric_orig.gamma_inv(x, y, z)
        _ = metric_orig.dgamma(x, y, z)
        _ = metric_orig.extrinsic_curvature(x, y, z)
        _ = metric_orig.K_trace(x, y, z)

        _ = metric_cached.gamma_inv(x, y, z)
        _ = metric_cached.dgamma(x, y, z)
        _ = metric_cached.extrinsic_curvature(x, y, z)
        _ = metric_cached.K_trace(x, y, z)

        # Also warm up the single-call method
        _ = metric_cached.compute_all_geometric(x, y, z)

    # Benchmark: calling all 4 methods per point (like residual.py does)
    print(f"\nBenchmarking {n_points} points, 4 metric calls each...")
    print("-" * 50)

    # Original (non-cached)
    start = time.perf_counter()
    for i in range(n_points):
        x, y, z = x_pts[i], y_pts[i], z_pts[i]
        gamma_inv = metric_orig.gamma_inv(x, y, z)
        dgamma = metric_orig.dgamma(x, y, z)
        K = metric_orig.extrinsic_curvature(x, y, z)
        K_trace = metric_orig.K_trace(x, y, z)
    t_orig = time.perf_counter() - start

    # Cached version
    start = time.perf_counter()
    for i in range(n_points):
        x, y, z = x_pts[i], y_pts[i], z_pts[i]
        gamma_inv = metric_cached.gamma_inv(x, y, z)
        dgamma = metric_cached.dgamma(x, y, z)
        K = metric_cached.extrinsic_curvature(x, y, z)
        K_trace = metric_cached.K_trace(x, y, z)
    t_cached = time.perf_counter() - start

    # Cached with single call
    start = time.perf_counter()
    for i in range(n_points):
        x, y, z = x_pts[i], y_pts[i], z_pts[i]
        gamma_inv, dgamma, K, K_trace = metric_cached.compute_all_geometric(x, y, z)
    t_single = time.perf_counter() - start

    print(f"\n{'Method':<35} {'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'Original (7× _compute_all)':<35} {t_orig*1000:>12.2f} {'1.0x':>10}")
    print(f"{'Cached (1× _compute_all)':<35} {t_cached*1000:>12.2f} {t_orig/t_cached:>10.1f}x")
    print(f"{'Cached + single call':<35} {t_single*1000:>12.2f} {t_orig/t_single:>10.1f}x")

    # Verify results match
    print("\nVerifying results match...")
    x, y, z = x_pts[0], y_pts[0], z_pts[0]

    gi_orig = metric_orig.gamma_inv(x, y, z)
    dg_orig = metric_orig.dgamma(x, y, z)
    K_orig = metric_orig.extrinsic_curvature(x, y, z)
    Kt_orig = metric_orig.K_trace(x, y, z)

    gi_cached, dg_cached, K_cached, Kt_cached = metric_cached.compute_all_geometric(x, y, z)

    print(f"  gamma_inv max diff: {np.max(np.abs(gi_orig - gi_cached)):.2e}")
    print(f"  dgamma max diff:    {np.max(np.abs(dg_orig - dg_cached)):.2e}")
    print(f"  K max diff:         {np.max(np.abs(K_orig - K_cached)):.2e}")
    print(f"  K_trace diff:       {abs(Kt_orig - Kt_cached):.2e}")

    # Breakdown: how much time in each part?
    print("\n" + "=" * 70)
    print("Breakdown of computation time")
    print("=" * 70)

    # Time just _compute_all equivalent
    start = time.perf_counter()
    for i in range(n_points):
        x, y, z = x_pts[i], y_pts[i], z_pts[i]
        metric_cached._invalidate_cache()
        metric_cached._ensure_base_computed(x, y, z)
    t_base = time.perf_counter() - start

    print(f"\n{'Operation':<40} {'Time (ms)':>12} {'% of cached':>12}")
    print("-" * 65)
    print(f"{'Base (H, l, dH, dl)':<40} {t_base*1000:>12.2f} {100*t_base/t_cached:>11.1f}%")
    print(f"{'Derived (gamma_inv, dgamma, K, K_trace)':<40} {(t_cached-t_base)*1000:>12.2f} {100*(t_cached-t_base)/t_cached:>11.1f}%")

    # Estimate full horizon find speedup
    print("\n" + "=" * 70)
    print("Estimated Full Horizon Find Speedup")
    print("=" * 70)

    # From profile: metric was 39% of total time
    # With 7x speedup on metric portion:
    metric_fraction = 0.39
    metric_speedup = t_orig / t_cached

    # New total = non-metric time + (metric time / speedup)
    # = (1 - 0.39) + 0.39/speedup
    new_fraction = (1 - metric_fraction) + metric_fraction / metric_speedup
    overall_speedup = 1.0 / new_fraction

    print(f"\nMetric computation speedup: {metric_speedup:.1f}x")
    print(f"Metric was {metric_fraction*100:.0f}% of total time")
    print(f"Estimated overall speedup: {overall_speedup:.1f}x")

if __name__ == "__main__":
    benchmark_metric_calls()
