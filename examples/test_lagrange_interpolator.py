"""
Benchmark and test the Lagrange interpolator.

Tests:
1. Interpolation accuracy vs spline
2. Speed comparison
3. Stencil locality (16 points)
4. Jacobian sparsity potential
"""

import numpy as np
import time
from ahfinder.surface import SurfaceMesh
from ahfinder.interpolation import FastInterpolator
from ahfinder.interpolation_lagrange import LagrangeInterpolator, interpolate_batch_lagrange


def test_accuracy():
    """Compare accuracy of Lagrange vs spline interpolation."""
    print("=" * 60)
    print("ACCURACY TEST")
    print("=" * 60)

    N_s = 33
    mesh = SurfaceMesh(N_s)

    # Test function: smooth spherical harmonic-like
    theta_grid, phi_grid = mesh.theta_phi_grid()
    rho = 2.0 + 0.3 * np.sin(2 * theta_grid) * np.cos(3 * phi_grid)

    # Random query points
    np.random.seed(42)
    n_query = 1000
    theta_query = np.random.uniform(0.1, np.pi - 0.1, n_query)
    phi_query = np.random.uniform(0, 2 * np.pi, n_query)

    # Spline interpolation
    spline_interp = FastInterpolator(mesh, spline_order=3)
    spline_result = spline_interp.interpolate_array(rho, theta_query, phi_query)

    # Lagrange interpolation
    lagrange_interp = LagrangeInterpolator(mesh)
    lagrange_result = lagrange_interp.interpolate_batch(rho, theta_query, phi_query)

    # Compare
    diff = np.abs(spline_result - lagrange_result)
    print(f"Max difference: {diff.max():.2e}")
    print(f"Mean difference: {diff.mean():.2e}")
    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.2e}")

    # Compare to analytical (true values at random points)
    # For this test function, we can compute exactly
    true_values = 2.0 + 0.3 * np.sin(2 * theta_query) * np.cos(3 * phi_query)

    spline_error = np.abs(spline_result - true_values)
    lagrange_error = np.abs(lagrange_result - true_values)

    print(f"\nSpline error (vs analytical):")
    print(f"  Max: {spline_error.max():.2e}, Mean: {spline_error.mean():.2e}")
    print(f"\nLagrange error (vs analytical):")
    print(f"  Max: {lagrange_error.max():.2e}, Mean: {lagrange_error.mean():.2e}")


def test_speed():
    """Compare speed of Lagrange vs spline interpolation."""
    print("\n" + "=" * 60)
    print("SPEED TEST")
    print("=" * 60)

    N_s = 33
    mesh = SurfaceMesh(N_s)

    theta_grid, phi_grid = mesh.theta_phi_grid()
    rho = 2.0 + 0.3 * np.sin(2 * theta_grid) * np.cos(3 * phi_grid)

    # Query points (typical for Jacobian computation)
    n_query = 10000
    theta_query = np.random.uniform(0.1, np.pi - 0.1, n_query)
    phi_query = np.random.uniform(0, 2 * np.pi, n_query)

    # Warm up
    spline_interp = FastInterpolator(mesh, spline_order=3)
    lagrange_interp = LagrangeInterpolator(mesh)

    _ = spline_interp.interpolate_array(rho, theta_query[:10], phi_query[:10])
    _ = lagrange_interp.interpolate_batch(rho, theta_query[:10], phi_query[:10])

    # Benchmark spline
    n_trials = 10
    times_spline = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = spline_interp.interpolate_array(rho, theta_query, phi_query)
        times_spline.append(time.perf_counter() - t0)

    # Benchmark Lagrange
    times_lagrange = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = lagrange_interp.interpolate_batch(rho, theta_query, phi_query)
        times_lagrange.append(time.perf_counter() - t0)

    t_spline = np.median(times_spline) * 1000
    t_lagrange = np.median(times_lagrange) * 1000

    print(f"Spline (k=3):  {t_spline:.3f} ms for {n_query} points")
    print(f"Lagrange:      {t_lagrange:.3f} ms for {n_query} points")
    print(f"Ratio: {t_lagrange/t_spline:.2f}x")


def test_stencil_locality():
    """Verify that Lagrange interpolation only depends on 16 grid points."""
    print("\n" + "=" * 60)
    print("STENCIL LOCALITY TEST")
    print("=" * 60)

    N_s = 33
    mesh = SurfaceMesh(N_s)
    lagrange_interp = LagrangeInterpolator(mesh)

    # Test points at various locations
    test_points = [
        (np.pi / 2, np.pi),       # Middle
        (0.3, 0.5),               # Near north pole
        (np.pi - 0.3, 5.5),       # Near south pole
        (np.pi / 2, 0.1),         # Near phi=0
        (np.pi / 2, 2 * np.pi - 0.1),  # Near phi=2π
    ]

    for theta, phi in test_points:
        indices = lagrange_interp.get_stencil_indices(theta, phi)
        print(f"Point (θ={theta:.2f}, φ={phi:.2f}): {len(indices)} stencil points")

        # Verify exactly 16 points
        assert len(indices) == 16, f"Expected 16 points, got {len(indices)}"

    print("\nAll tests passed: exactly 16 stencil points for each query point.")


def analyze_jacobian_sparsity():
    """Analyze potential Jacobian sparsity with Lagrange interpolation."""
    print("\n" + "=" * 60)
    print("JACOBIAN SPARSITY ANALYSIS")
    print("=" * 60)

    N_s = 17  # Smaller for analysis
    mesh = SurfaceMesh(N_s)
    lagrange_interp = LagrangeInterpolator(mesh)

    # For each grid point, find which other grid points affect its interpolation
    # when queried at the stencil positions used for residual computation

    n_independent = mesh.n_independent
    print(f"Grid size: {N_s}x{N_s}")
    print(f"Independent points: {n_independent}")

    # For each independent point, determine dependencies
    # This simulates what happens in residual computation

    # Get all independent points
    indices = mesh.independent_indices()

    # Estimate: each residual point queries interpolation at ~27 stencil positions
    # (3x3x3 Cartesian stencil)
    # Each query depends on 16 grid points
    # But many of these will overlap for nearby queries

    # Conservative estimate of dependencies per residual point
    total_deps = 0
    max_deps = 0

    for i_th, i_ph in indices:
        theta = mesh.theta[i_th]
        phi = mesh.phi[i_ph]

        # Simulate 27 stencil queries around this point
        all_deps = set()
        for dth in [-0.1, 0, 0.1]:
            for dph in [-0.1, 0, 0.1]:
                for dr in [-0.1, 0, 0.1]:
                    # Query point (approximate, actual stencil is Cartesian)
                    th_query = np.clip(theta + dth, 0.01, np.pi - 0.01)
                    ph_query = (phi + dph) % (2 * np.pi)

                    stencil = lagrange_interp.get_stencil_indices(th_query, ph_query)
                    all_deps.update(stencil)

        total_deps += len(all_deps)
        max_deps = max(max_deps, len(all_deps))

    avg_deps = total_deps / n_independent

    print(f"\nDependencies per residual point:")
    print(f"  Average: {avg_deps:.1f}")
    print(f"  Maximum: {max_deps}")
    print(f"\nJacobian density:")
    print(f"  With Lagrange: {100 * avg_deps / n_independent:.1f}%")
    print(f"  Dense (current): 100%")
    print(f"\nPotential speedup: {n_independent / avg_deps:.1f}x fewer residual evals")


def test_pole_handling():
    """Test interpolation near poles."""
    print("\n" + "=" * 60)
    print("POLE HANDLING TEST")
    print("=" * 60)

    N_s = 33
    mesh = SurfaceMesh(N_s)
    lagrange_interp = LagrangeInterpolator(mesh)

    theta_grid, phi_grid = mesh.theta_phi_grid()
    rho = 2.0 * np.ones_like(theta_grid)  # Constant function

    # Test at exact poles
    result_north = lagrange_interp.interpolate(rho, 0, 0)
    result_south = lagrange_interp.interpolate(rho, np.pi, 0)

    print(f"North pole (θ=0): {result_north:.6f} (expected 2.0)")
    print(f"South pole (θ=π): {result_south:.6f} (expected 2.0)")

    # Test near poles
    for theta in [0.01, 0.05, 0.1]:
        results = []
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            r = lagrange_interp.interpolate(rho, theta, phi)
            results.append(r)
        print(f"Near north pole (θ={theta:.2f}): {np.mean(results):.6f} ± {np.std(results):.2e}")


if __name__ == "__main__":
    test_accuracy()
    test_speed()
    test_stencil_locality()
    analyze_jacobian_sparsity()
    test_pole_handling()
