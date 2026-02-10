"""
Benchmark vectorized vs point-by-point residual evaluation.
"""

import numpy as np
import time
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast
from ahfinder.surface import SurfaceMesh
from ahfinder.jacobian_sparse import create_sparse_residual_evaluator
from ahfinder.residual_vectorized import create_vectorized_residual_evaluator


def benchmark_residual(N_s: int):
    """Compare point-by-point vs vectorized residual evaluation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking residual evaluation for N_s={N_s}")
    print(f"{'='*60}")

    M = 1.0
    r_guess = 2 * M

    metric = SchwarzschildMetricFast(M=M)
    mesh = SurfaceMesh(N_s=N_s)

    rho = np.full((N_s, 2*N_s - 1), r_guess)
    n = mesh.n_independent

    print(f"Grid: {N_s}x{2*N_s-1}, Independent DOFs: {n}")

    # Create evaluators
    print("\nCreating evaluators...")
    sparse_residual = create_sparse_residual_evaluator(
        mesh, metric, center=(0.0, 0.0, 0.0), spacing_factor=0.5
    )
    vec_residual = create_vectorized_residual_evaluator(
        mesh, metric, center=(0.0, 0.0, 0.0), spacing_factor=0.5
    )

    # Warmup JIT
    print("Warming up JIT...")
    _ = sparse_residual.evaluate(rho)
    _ = vec_residual.evaluate(rho)

    # Benchmark point-by-point
    n_runs = 10
    print(f"\nBenchmarking ({n_runs} runs each)...")

    times_sparse = []
    for _ in range(n_runs):
        t0 = time.time()
        F_sparse = sparse_residual.evaluate(rho)
        times_sparse.append(time.time() - t0)

    times_vec = []
    for _ in range(n_runs):
        t0 = time.time()
        F_vec = vec_residual.evaluate(rho)
        times_vec.append(time.time() - t0)

    t_sparse = np.mean(times_sparse)
    t_vec = np.mean(times_vec)

    # Verify results match
    max_diff = np.max(np.abs(F_sparse - F_vec))
    rel_diff = max_diff / (np.max(np.abs(F_sparse)) + 1e-15)

    print(f"\nResults:")
    print(f"  Point-by-point: {1000*t_sparse:.2f}ms")
    print(f"  Vectorized:     {1000*t_vec:.2f}ms")
    print(f"  Speedup:        {t_sparse/t_vec:.2f}x")
    print(f"\nVerification:")
    print(f"  Max |F_sparse - F_vec|: {max_diff:.2e}")
    print(f"  Relative difference:    {rel_diff:.2e}")

    return {
        'N_s': N_s,
        'n': n,
        't_sparse': t_sparse,
        't_vec': t_vec,
        'speedup': t_sparse / t_vec,
        'max_diff': max_diff
    }


if __name__ == "__main__":
    results = []

    for N_s in [13, 17, 21, 25]:
        result = benchmark_residual(N_s)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Vectorized Residual Performance")
    print("="*70)
    print(f"{'N_s':>5} | {'DOFs':>6} | {'Point-by-point':>14} | {'Vectorized':>11} | {'Speedup':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['N_s']:>5} | {r['n']:>6} | {1000*r['t_sparse']:>12.2f}ms | "
              f"{1000*r['t_vec']:>9.2f}ms | {r['speedup']:>6.2f}x")
