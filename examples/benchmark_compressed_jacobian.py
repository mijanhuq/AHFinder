"""
Benchmark compressed Jacobian computation using graph coloring.

Compares:
1. Regular sparse Jacobian (one evaluation per column)
2. Compressed Jacobian (one evaluation per color group)
"""

import numpy as np
import time
from ahfinder.metrics import SchwarzschildMetric
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast
from ahfinder.surface import SurfaceMesh
from ahfinder.jacobian_sparse import (
    create_sparse_residual_evaluator,
    SparseJacobianComputer
)
from ahfinder.jacobian_compressed import (
    compute_column_colors,
    compute_compressed_jacobian,
    CompressedJacobianComputer
)


def benchmark_jacobian(N_s: int, use_fast_metric: bool = True):
    """Benchmark regular vs compressed sparse Jacobian."""
    print(f"\n{'='*60}")
    print(f"Benchmarking N_s={N_s}, fast_metric={use_fast_metric}")
    print(f"{'='*60}")

    # Setup
    M = 1.0
    r_guess = 2 * M

    if use_fast_metric:
        metric = SchwarzschildMetricFast(M=M)
    else:
        metric = SchwarzschildMetric(M=M)

    mesh = SurfaceMesh(N_s=N_s)

    # Initial surface
    rho = np.full((N_s, 2*N_s - 1), r_guess)

    # Create sparse residual evaluator
    sparse_residual = create_sparse_residual_evaluator(
        mesh, metric, center=(0.0, 0.0, 0.0), spacing_factor=0.5
    )

    n = mesh.n_independent
    print(f"Grid: {N_s}x{2*N_s-1}, Independent DOFs: {n}")

    # 1. Compute graph coloring
    print("\nComputing graph coloring...")
    t0 = time.time()
    colors, num_colors = compute_column_colors(mesh, sparse_residual, rho)
    t_coloring = time.time() - t0
    print(f"  Coloring time: {t_coloring:.3f}s")
    print(f"  {n} columns → {num_colors} color groups ({n/num_colors:.1f}x compression)")

    # 2. Benchmark regular sparse Jacobian
    print("\nRegular sparse Jacobian (one eval per column)...")
    sparse_jac = SparseJacobianComputer(mesh, sparse_residual)

    t0 = time.time()
    J_regular = sparse_jac.compute_sparse(rho)
    t_regular = time.time() - t0
    print(f"  Time: {t_regular:.3f}s")
    print(f"  Non-zeros: {J_regular.nnz} ({100*J_regular.nnz/n**2:.1f}% density)")

    # 3. Benchmark compressed Jacobian
    print("\nCompressed Jacobian (one eval per color group)...")
    t0 = time.time()
    J_compressed = compute_compressed_jacobian(mesh, sparse_residual, rho, verbose=False)
    t_compressed = time.time() - t0
    print(f"  Time: {t_compressed:.3f}s")
    print(f"  Non-zeros: {J_compressed.nnz} ({100*J_compressed.nnz/n**2:.1f}% density)")

    # 4. Verify results match
    diff = np.abs(J_regular.toarray() - J_compressed.toarray())
    max_diff = np.max(diff)
    print(f"\nVerification: max |J_regular - J_compressed| = {max_diff:.2e}")

    # 5. Summary
    speedup = t_regular / t_compressed
    print(f"\n{'='*60}")
    print(f"SUMMARY for N_s={N_s}")
    print(f"{'='*60}")
    print(f"  Regular sparse Jacobian:    {t_regular:.3f}s ({n} evaluations)")
    print(f"  Compressed sparse Jacobian: {t_compressed:.3f}s ({num_colors} evaluations)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Coloring overhead: {t_coloring:.3f}s (one-time cost)")

    return {
        'N_s': N_s,
        'n': n,
        'num_colors': num_colors,
        't_regular': t_regular,
        't_compressed': t_compressed,
        'speedup': speedup,
        't_coloring': t_coloring
    }


if __name__ == "__main__":
    results = []

    # Test multiple grid sizes
    for N_s in [13, 17, 21, 25]:
        result = benchmark_jacobian(N_s, use_fast_metric=True)
        results.append(result)

    # Final summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY: Compressed Jacobian Performance")
    print("="*70)
    print(f"{'N_s':>5} | {'DOFs':>6} | {'Colors':>7} | {'Regular':>10} | {'Compressed':>11} | {'Speedup':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['N_s']:>5} | {r['n']:>6} | {r['num_colors']:>7} | "
              f"{r['t_regular']:>9.3f}s | {r['t_compressed']:>10.3f}s | {r['speedup']:>6.1f}x")
