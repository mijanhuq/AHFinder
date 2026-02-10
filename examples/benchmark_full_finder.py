"""
Benchmark full horizon finding with vectorized residual evaluation.
"""

import numpy as np
import time
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast


def benchmark_finder(N_s: int, use_sparse: bool = True):
    """Benchmark full horizon finding."""
    print(f"\n{'='*60}")
    print(f"Full horizon finding: N_s={N_s}, sparse_jacobian={use_sparse}")
    print(f"{'='*60}")

    M = 1.0
    metric = SchwarzschildMetricFast(M=M)

    finder = ApparentHorizonFinder(
        metric, N_s=N_s,
        use_sparse_jacobian=use_sparse
    )

    # Time the full solve
    t0 = time.time()
    rho = finder.find(initial_radius=2.0, tol=1e-9, max_iter=20, verbose=True)
    t_total = time.time() - t0

    # Verify result
    r_horizon = np.mean(rho)
    expected = 2 * M
    error = abs(r_horizon - expected) / expected

    print(f"\nResults:")
    print(f"  Total time:       {t_total:.2f}s")
    print(f"  Horizon radius:   {r_horizon:.6f} (expected {expected:.6f})")
    print(f"  Relative error:   {error:.2e}")

    return t_total, r_horizon


if __name__ == "__main__":
    print("Benchmarking with current sparse Jacobian implementation")
    print("="*60)

    for N_s in [17, 21, 25]:
        benchmark_finder(N_s, use_sparse=True)
