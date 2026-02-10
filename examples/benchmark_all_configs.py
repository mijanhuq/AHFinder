"""
Comprehensive benchmark: all configurations from baseline to best.
"""

import numpy as np
import time
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics import SchwarzschildMetric
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast


def run_finder(metric, N_s, use_sparse=False, use_vectorized=False):
    """Run horizon finder and return time."""
    finder = ApparentHorizonFinder(
        metric, N_s=N_s,
        use_sparse_jacobian=use_sparse,
        use_vectorized_jacobian=use_vectorized
    )

    t0 = time.time()
    rho = finder.find(initial_radius=2.0, tol=1e-9, max_iter=20, verbose=False)
    return time.time() - t0


if __name__ == "__main__":
    N_s = 25
    M = 1.0

    print(f"Comprehensive Performance Benchmark (N_s={N_s})")
    print("="*70)

    configs = [
        ("Dense Jacobian + Regular Metric", SchwarzschildMetric(M=M), False, False),
        ("Dense Jacobian + Fast Metric", SchwarzschildMetricFast(M=M), False, False),
        ("Sparse Jacobian + Regular Metric", SchwarzschildMetric(M=M), True, False),
        ("Sparse Jacobian + Fast Metric", SchwarzschildMetricFast(M=M), True, False),
        ("Vectorized Jacobian + Fast Metric", SchwarzschildMetricFast(M=M), False, True),
    ]

    results = []
    baseline = None

    for name, metric, use_sparse, use_vec in configs:
        print(f"\n{name}...")

        # Warmup
        t = run_finder(metric, N_s, use_sparse, use_vec)

        # Benchmark
        times = []
        for _ in range(3):
            t = run_finder(metric, N_s, use_sparse, use_vec)
            times.append(t)

        t_avg = np.mean(times)

        if baseline is None:
            baseline = t_avg
            speedup = 1.0
        else:
            speedup = baseline / t_avg

        print(f"  Time: {t_avg:.2f}s, Speedup: {speedup:.1f}x")
        results.append((name, t_avg, speedup))

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Configuration':<45} | {'Time':>8} | {'Speedup':>8}")
    print("-"*70)
    for name, t, s in results:
        print(f"{name:<45} | {t:>7.2f}s | {s:>7.1f}x")
