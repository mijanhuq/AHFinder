"""
Comprehensive benchmark: Dense vs Sparse Jacobian horizon finding.
"""

import numpy as np
import time
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.kerr import KerrMetric


def benchmark_schwarzschild():
    """Benchmark Schwarzschild horizon finding."""
    print("=" * 70)
    print("SCHWARZSCHILD HORIZON FINDING BENCHMARK")
    print("=" * 70)

    metric = SchwarzschildMetric(M=1.0)
    expected_r = 2.0
    expected_area = 16 * np.pi  # 4π(2M)²

    sizes = [17, 21, 25, 29, 33]

    print(f"\n{'N_s':>5} {'n_indep':>8} {'Dense (s)':>12} {'Sparse (s)':>12} "
          f"{'Speedup':>10} {'Radius':>10} {'Area err':>12}")
    print("-" * 85)

    for N_s in sizes:
        # Dense Jacobian
        finder_dense = ApparentHorizonFinder(metric, N_s=N_s, use_sparse_jacobian=False)

        t0 = time.perf_counter()
        rho_dense = finder_dense.find(verbose=False)
        t_dense = time.perf_counter() - t0

        r_dense = np.mean(rho_dense)
        area_dense = finder_dense.horizon_area(rho_dense)
        area_err_dense = abs(area_dense - expected_area) / expected_area

        # Sparse Jacobian
        finder_sparse = ApparentHorizonFinder(metric, N_s=N_s, use_sparse_jacobian=True)

        t0 = time.perf_counter()
        rho_sparse = finder_sparse.find(verbose=False)
        t_sparse = time.perf_counter() - t0

        r_sparse = np.mean(rho_sparse)
        area_sparse = finder_sparse.horizon_area(rho_sparse)

        speedup = t_dense / t_sparse

        n_indep = finder_dense.mesh.n_independent

        print(f"{N_s:>5} {n_indep:>8} {t_dense:>12.3f} {t_sparse:>12.3f} "
              f"{speedup:>10.1f}x {r_sparse:>10.4f} {area_err_dense:>12.2e}")


def benchmark_kerr():
    """Benchmark Kerr horizon finding."""
    print("\n" + "=" * 70)
    print("KERR HORIZON FINDING BENCHMARK (a=0.5)")
    print("=" * 70)

    a = 0.5
    M = 1.0
    metric = KerrMetric(M=M, a=a)

    # Expected horizon radius at equator for Kerr
    r_plus = M + np.sqrt(M**2 - a**2)

    sizes = [17, 21, 25, 29]

    print(f"\n{'N_s':>5} {'n_indep':>8} {'Dense (s)':>12} {'Sparse (s)':>12} "
          f"{'Speedup':>10} {'r_eq':>10}")
    print("-" * 70)

    for N_s in sizes:
        # Dense Jacobian
        finder_dense = ApparentHorizonFinder(metric, N_s=N_s, use_sparse_jacobian=False)

        t0 = time.perf_counter()
        rho_dense = finder_dense.find(verbose=False)
        t_dense = time.perf_counter() - t0

        r_eq_dense = finder_dense.horizon_radius_equatorial(rho_dense)

        # Sparse Jacobian
        finder_sparse = ApparentHorizonFinder(metric, N_s=N_s, use_sparse_jacobian=True)

        t0 = time.perf_counter()
        rho_sparse = finder_sparse.find(verbose=False)
        t_sparse = time.perf_counter() - t0

        r_eq_sparse = finder_sparse.horizon_radius_equatorial(rho_sparse)

        speedup = t_dense / t_sparse

        n_indep = finder_dense.mesh.n_independent

        print(f"{N_s:>5} {n_indep:>8} {t_dense:>12.3f} {t_sparse:>12.3f} "
              f"{speedup:>10.1f}x {r_eq_sparse:>10.4f}")

    print(f"\nExpected equatorial radius: {r_plus:.4f}")


def detailed_timing():
    """Detailed timing breakdown for N_s=25."""
    print("\n" + "=" * 70)
    print("DETAILED TIMING BREAKDOWN (N_s=25, Schwarzschild)")
    print("=" * 70)

    metric = SchwarzschildMetric(M=1.0)
    N_s = 25

    # Dense
    print("\n--- Dense Jacobian ---")
    finder = ApparentHorizonFinder(metric, N_s=N_s, use_sparse_jacobian=False)
    rho = finder.find(verbose=True)

    # Sparse
    print("\n--- Sparse Jacobian (Lagrange) ---")
    finder_sparse = ApparentHorizonFinder(metric, N_s=N_s, use_sparse_jacobian=True)
    rho_sparse = finder_sparse.find(verbose=True)

    # Compare results
    print(f"\nRadius comparison:")
    print(f"  Dense:  {np.mean(rho):.6f}")
    print(f"  Sparse: {np.mean(rho_sparse):.6f}")
    print(f"  Diff:   {abs(np.mean(rho) - np.mean(rho_sparse)):.2e}")


if __name__ == "__main__":
    benchmark_schwarzschild()
    benchmark_kerr()
    detailed_timing()
