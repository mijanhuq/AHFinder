"""
Benchmark full horizon finding: original sparse vs vectorized.
"""

import numpy as np
import time
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast


def benchmark_finder(N_s: int, use_vectorized: bool):
    """Benchmark full horizon finding."""
    M = 1.0
    metric = SchwarzschildMetricFast(M=M)

    if use_vectorized:
        finder = ApparentHorizonFinder(
            metric, N_s=N_s,
            use_vectorized_jacobian=True
        )
    else:
        finder = ApparentHorizonFinder(
            metric, N_s=N_s,
            use_sparse_jacobian=True
        )

    t0 = time.time()
    rho = finder.find(initial_radius=2.0, tol=1e-9, max_iter=20, verbose=False)
    t_total = time.time() - t0

    r_horizon = np.mean(rho)
    error = abs(r_horizon - 2*M) / (2*M)

    return t_total, r_horizon, error


if __name__ == "__main__":
    print("Comparing Original Sparse vs Vectorized Sparse Jacobian")
    print("="*70)

    results = []

    for N_s in [17, 21, 25]:
        print(f"\nN_s = {N_s}")
        print("-"*40)

        # Warmup
        _ = benchmark_finder(N_s, use_vectorized=False)
        _ = benchmark_finder(N_s, use_vectorized=True)

        # Benchmark
        n_runs = 3
        times_orig = []
        times_vec = []

        for _ in range(n_runs):
            t, _, _ = benchmark_finder(N_s, use_vectorized=False)
            times_orig.append(t)

        for _ in range(n_runs):
            t, r, e = benchmark_finder(N_s, use_vectorized=True)
            times_vec.append(t)

        t_orig = np.mean(times_orig)
        t_vec = np.mean(times_vec)
        speedup = t_orig / t_vec

        print(f"  Original sparse:   {t_orig:.2f}s")
        print(f"  Vectorized sparse: {t_vec:.2f}s")
        print(f"  Speedup:           {speedup:.2f}x")
        print(f"  Horizon radius:    {r:.6f} (expected 2.0)")

        results.append({
            'N_s': N_s,
            't_orig': t_orig,
            't_vec': t_vec,
            'speedup': speedup
        })

    print("\n" + "="*70)
    print("SUMMARY: Full Horizon Finding Performance")
    print("="*70)
    print(f"{'N_s':>5} | {'Original':>10} | {'Vectorized':>11} | {'Speedup':>7}")
    print("-"*50)
    for r in results:
        print(f"{r['N_s']:>5} | {r['t_orig']:>9.2f}s | {r['t_vec']:>10.2f}s | {r['speedup']:>6.2f}x")
