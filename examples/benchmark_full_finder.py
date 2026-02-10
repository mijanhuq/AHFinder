#!/usr/bin/env python3
"""
Benchmark full horizon finder with optimized vs original metric.
"""

import numpy as np
import time

def benchmark_full_finder():
    from ahfinder import ApparentHorizonFinder
    from ahfinder.metrics.boosted_kerr_fast import FastBoostedKerrMetric
    from ahfinder.metrics.boosted_kerr_fast_cached import FastBoostedKerrMetricCached

    print("=" * 70)
    print("Full Horizon Finder Benchmark")
    print("=" * 70)

    velocity = np.array([0.3, 0.0, 0.0])

    for N_s in [17, 25]:
        print(f"\nN_s = {N_s}")
        print("-" * 50)

        # Original metric
        metric_orig = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=velocity)
        finder_orig = ApparentHorizonFinder(metric_orig, N_s=N_s)

        # Cached metric with fast single-call
        metric_fast = FastBoostedKerrMetricCached(M=1.0, a=0.5, velocity=velocity)
        finder_fast = ApparentHorizonFinder(metric_fast, N_s=N_s)

        # Warm up JIT
        print("Warming up JIT...")
        _ = finder_orig.find(initial_radius=1.9, tol=1e-5, max_iter=10)
        _ = finder_fast.find(initial_radius=1.9, tol=1e-5, max_iter=10)

        # Benchmark original
        n_runs = 3
        times_orig = []
        for _ in range(n_runs):
            start = time.perf_counter()
            rho = finder_orig.find(initial_radius=1.9, tol=1e-5, max_iter=20)
            times_orig.append(time.perf_counter() - start)
        t_orig = np.mean(times_orig)

        # Benchmark fast
        times_fast = []
        for _ in range(n_runs):
            start = time.perf_counter()
            rho = finder_fast.find(initial_radius=1.9, tol=1e-5, max_iter=20)
            times_fast.append(time.perf_counter() - start)
        t_fast = np.mean(times_fast)

        print(f"\n{'Method':<35} {'Time (s)':>12} {'Speedup':>10}")
        print("-" * 60)
        print(f"{'Original FastBoostedKerrMetric':<35} {t_orig:>12.3f} {'1.0x':>10}")
        print(f"{'Cached + single JIT call':<35} {t_fast:>12.3f} {t_orig/t_fast:>10.1f}x")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

if __name__ == "__main__":
    benchmark_full_finder()
