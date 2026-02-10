"""
Benchmark vectorized vs original sparse Jacobian computation.
"""

import numpy as np
import time
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast
from ahfinder.surface import SurfaceMesh
from ahfinder.jacobian_sparse import (
    create_sparse_residual_evaluator,
    SparseJacobianComputer
)
from ahfinder.jacobian_vectorized import create_vectorized_jacobian_computer


def benchmark_jacobian(N_s: int):
    """Compare original vs vectorized Jacobian."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Jacobian computation for N_s={N_s}")
    print(f"{'='*60}")

    M = 1.0
    r_guess = 2 * M

    metric = SchwarzschildMetricFast(M=M)
    mesh = SurfaceMesh(N_s=N_s)

    rho = np.full((N_s, 2*N_s - 1), r_guess)
    n = mesh.n_independent

    print(f"Grid: {N_s}x{2*N_s-1}, Independent DOFs: {n}")

    # Create computers
    print("\nCreating Jacobian computers...")
    sparse_residual = create_sparse_residual_evaluator(
        mesh, metric, center=(0.0, 0.0, 0.0), spacing_factor=0.5
    )
    sparse_jac = SparseJacobianComputer(mesh, sparse_residual)

    vec_jac = create_vectorized_jacobian_computer(
        mesh, metric, center=(0.0, 0.0, 0.0), spacing_factor=0.5
    )

    # Warmup
    print("Warming up JIT...")
    _ = sparse_jac.compute_sparse(rho)
    _ = vec_jac.compute_sparse(rho, verbose=False)

    # Benchmark
    n_runs = 3
    print(f"\nBenchmarking ({n_runs} runs each)...")

    times_orig = []
    for _ in range(n_runs):
        t0 = time.time()
        J_orig = sparse_jac.compute_sparse(rho)
        times_orig.append(time.time() - t0)

    times_vec = []
    for _ in range(n_runs):
        t0 = time.time()
        J_vec = vec_jac.compute_sparse(rho, verbose=False)
        times_vec.append(time.time() - t0)

    t_orig = np.mean(times_orig)
    t_vec = np.mean(times_vec)

    # Verify
    max_diff = np.max(np.abs(J_orig.toarray() - J_vec.toarray()))

    print(f"\nResults:")
    print(f"  Original sparse:   {t_orig:.3f}s")
    print(f"  Vectorized sparse: {t_vec:.3f}s")
    print(f"  Speedup:           {t_orig/t_vec:.2f}x")
    print(f"  Max |J_orig - J_vec|: {max_diff:.2e}")

    return {
        'N_s': N_s,
        'n': n,
        't_orig': t_orig,
        't_vec': t_vec,
        'speedup': t_orig / t_vec
    }


if __name__ == "__main__":
    results = []

    for N_s in [13, 17, 21, 25]:
        result = benchmark_jacobian(N_s)
        results.append(result)

    print("\n" + "="*70)
    print("SUMMARY: Vectorized Jacobian Performance")
    print("="*70)
    print(f"{'N_s':>5} | {'DOFs':>6} | {'Original':>10} | {'Vectorized':>11} | {'Speedup':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['N_s']:>5} | {r['n']:>6} | {r['t_orig']:>9.3f}s | "
              f"{r['t_vec']:>10.3f}s | {r['speedup']:>6.2f}x")
