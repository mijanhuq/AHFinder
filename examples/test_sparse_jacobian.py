"""
Test sparse Jacobian computation with Lagrange interpolation.

Compares:
1. Accuracy of sparse vs dense Jacobian
2. Speed improvement
3. Convergence of Newton iteration with sparse Jacobian
"""

import numpy as np
import time
from ahfinder.surface import SurfaceMesh, create_sphere
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.jacobian_sparse import (
    create_sparse_residual_evaluator,
    SparseJacobianComputer
)
from ahfinder.interpolation import FastInterpolator
from ahfinder.residual import create_residual_evaluator
from ahfinder.jacobian import JacobianComputer


def test_jacobian_accuracy():
    """Compare sparse Jacobian to dense Jacobian."""
    print("=" * 60)
    print("JACOBIAN ACCURACY TEST")
    print("=" * 60)

    N_s = 17
    mesh = SurfaceMesh(N_s)
    metric = SchwarzschildMetric(M=1.0)
    center = (0.0, 0.0, 0.0)

    # Initial surface (near horizon)
    rho = create_sphere(mesh, 2.0)

    # Dense Jacobian (original method with splines)
    print("\nComputing dense Jacobian (spline interpolation)...")
    interpolator = FastInterpolator(mesh, spline_order=3)
    dense_residual = create_residual_evaluator(mesh, interpolator, metric, center)
    dense_jacobian = JacobianComputer(mesh, dense_residual)

    t0 = time.perf_counter()
    J_dense = dense_jacobian.compute_dense(rho)
    t_dense = time.perf_counter() - t0
    print(f"  Time: {t_dense:.3f} s")

    # Sparse Jacobian (Lagrange interpolation)
    print("\nComputing sparse Jacobian (Lagrange interpolation)...")
    sparse_residual = create_sparse_residual_evaluator(mesh, metric, center)
    sparse_jacobian = SparseJacobianComputer(mesh, sparse_residual)

    t0 = time.perf_counter()
    J_sparse = sparse_jacobian.compute_sparse(rho)
    t_sparse = time.perf_counter() - t0
    print(f"  Time: {t_sparse:.3f} s")

    # Compare
    J_sparse_dense = J_sparse.toarray()
    diff = np.abs(J_dense - J_sparse_dense)

    # Relative difference (avoiding division by zero)
    rel_diff = np.where(
        np.abs(J_dense) > 1e-10,
        diff / np.abs(J_dense),
        diff
    )

    print(f"\nJacobian comparison:")
    print(f"  Max absolute diff: {diff.max():.2e}")
    print(f"  Mean absolute diff: {diff.mean():.2e}")
    print(f"  Max relative diff: {rel_diff.max():.2e}")
    print(f"  Speedup: {t_dense/t_sparse:.1f}x")


def test_jacobian_speed():
    """Benchmark Jacobian computation for various grid sizes."""
    print("\n" + "=" * 60)
    print("JACOBIAN SPEED BENCHMARK")
    print("=" * 60)

    sizes = [13, 17, 21, 25]
    metric = SchwarzschildMetric(M=1.0)
    center = (0.0, 0.0, 0.0)

    print(f"\n{'N_s':>5} {'n_indep':>8} {'Dense (s)':>12} {'Sparse (s)':>12} {'Speedup':>10}")
    print("-" * 55)

    for N_s in sizes:
        mesh = SurfaceMesh(N_s)
        rho = create_sphere(mesh, 2.0)

        # Dense (warm up + measure)
        interpolator = FastInterpolator(mesh, spline_order=3)
        dense_residual = create_residual_evaluator(mesh, interpolator, metric, center)
        dense_jacobian = JacobianComputer(mesh, dense_residual)

        _ = dense_jacobian.compute_dense(rho)  # Warm up
        t0 = time.perf_counter()
        _ = dense_jacobian.compute_dense(rho)
        t_dense = time.perf_counter() - t0

        # Sparse (warm up + measure)
        sparse_residual = create_sparse_residual_evaluator(mesh, metric, center)
        sparse_jacobian = SparseJacobianComputer(mesh, sparse_residual)

        _ = sparse_jacobian.compute_sparse(rho)  # Warm up
        t0 = time.perf_counter()
        _ = sparse_jacobian.compute_sparse(rho)
        t_sparse = time.perf_counter() - t0

        speedup = t_dense / t_sparse
        print(f"{N_s:>5} {mesh.n_independent:>8} {t_dense:>12.3f} {t_sparse:>12.3f} {speedup:>10.1f}x")


def test_newton_convergence():
    """Test Newton iteration with sparse Jacobian."""
    print("\n" + "=" * 60)
    print("NEWTON CONVERGENCE TEST")
    print("=" * 60)

    N_s = 17
    mesh = SurfaceMesh(N_s)
    metric = SchwarzschildMetric(M=1.0)
    center = (0.0, 0.0, 0.0)

    # Initial guess
    rho = create_sphere(mesh, 2.5)  # Start a bit off

    # Create sparse residual and Jacobian
    sparse_residual = create_sparse_residual_evaluator(mesh, metric, center)
    sparse_jacobian = SparseJacobianComputer(mesh, sparse_residual)

    print("\nNewton iteration with sparse Jacobian:")
    print("-" * 50)

    tol = 1e-9
    max_iter = 15

    for iteration in range(max_iter):
        F = sparse_residual.evaluate(rho)
        F_norm = np.linalg.norm(F)

        print(f"  Iter {iteration:2d}: ||F|| = {F_norm:.6e}", end="")

        if F_norm < tol:
            print()
            print("-" * 50)
            print(f"Converged in {iteration + 1} iterations!")

            # Check horizon radius
            r_mean = np.mean(rho)
            print(f"\nHorizon radius: {r_mean:.6f} (expected: 2.0)")
            return True

        # Compute sparse Jacobian
        J = sparse_jacobian.compute_dense(rho)

        # Solve J @ delta_rho = -F
        delta_rho_flat = np.linalg.solve(J, -F)
        delta_rho = mesh.flat_to_grid(delta_rho_flat)

        delta_norm = np.linalg.norm(delta_rho_flat)
        print(f", ||δρ|| = {delta_norm:.6e}")

        # Update
        rho = rho + delta_rho

    print("-" * 50)
    print("WARNING: Did not converge!")
    return False


def test_residual_comparison():
    """Compare residual values between spline and Lagrange interpolation."""
    print("\n" + "=" * 60)
    print("RESIDUAL COMPARISON TEST")
    print("=" * 60)

    N_s = 17
    mesh = SurfaceMesh(N_s)
    metric = SchwarzschildMetric(M=1.0)
    center = (0.0, 0.0, 0.0)

    rho = create_sphere(mesh, 2.0)

    # Spline residual
    interpolator = FastInterpolator(mesh, spline_order=3)
    spline_residual = create_residual_evaluator(mesh, interpolator, metric, center)
    F_spline = spline_residual.evaluate(rho)

    # Lagrange residual
    lagrange_residual = create_sparse_residual_evaluator(mesh, metric, center)
    F_lagrange = lagrange_residual.evaluate(rho)

    diff = np.abs(F_spline - F_lagrange)
    rel_diff = np.where(
        np.abs(F_spline) > 1e-10,
        diff / np.abs(F_spline),
        diff
    )

    print(f"\nResidual comparison (N_s={N_s}, r=2.0):")
    print(f"  Max abs diff: {diff.max():.2e}")
    print(f"  Mean abs diff: {diff.mean():.2e}")
    print(f"  Max rel diff: {rel_diff.max():.2e}")
    print(f"  ||F_spline||: {np.linalg.norm(F_spline):.6e}")
    print(f"  ||F_lagrange||: {np.linalg.norm(F_lagrange):.6e}")


if __name__ == "__main__":
    test_residual_comparison()
    test_jacobian_accuracy()
    test_jacobian_speed()
    test_newton_convergence()
