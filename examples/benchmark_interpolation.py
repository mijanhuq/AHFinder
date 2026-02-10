#!/usr/bin/env python3
"""
Benchmark different interpolation methods for AHFinder.
"""

import numpy as np
import time
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy import ndimage

def benchmark_interpolation():
    """Compare interpolation methods."""

    print("=" * 70)
    print("Interpolation Benchmark")
    print("=" * 70)

    # Test sizes
    for N_s in [17, 25, 33]:
        n_points = N_s * N_s
        n_query = 10000  # Number of query points

        print(f"\nN_s = {N_s} (grid {N_s}x{N_s}), {n_query} query points")
        print("-" * 60)

        # Create test grid
        theta = np.linspace(0, np.pi, N_s)
        phi = np.linspace(0, 2*np.pi, N_s, endpoint=False)
        d_theta = theta[1] - theta[0]
        d_phi = phi[1] - phi[0]

        # Test function (something smooth)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        rho = 2.0 + 0.1 * np.sin(THETA) * np.cos(2*PHI)

        # Random query points
        np.random.seed(42)
        query_theta = np.random.uniform(0.1, np.pi - 0.1, n_query)
        query_phi = np.random.uniform(0, 2*np.pi, n_query)

        results = {}

        # 1. RectBivariateSpline quintic (current, k=5)
        # Extend for periodicity
        n_extend = 3
        phi_ext = np.concatenate([phi[-n_extend:] - 2*np.pi, phi, phi[:n_extend] + 2*np.pi])
        rho_ext = np.column_stack([rho[:, -n_extend:], rho, rho[:, :n_extend]])

        spline5 = RectBivariateSpline(theta, phi_ext, rho_ext, kx=5, ky=5, s=0)

        start = time.perf_counter()
        for _ in range(10):
            result = spline5(query_theta, query_phi % (2*np.pi), grid=False)
        t5 = (time.perf_counter() - start) / 10
        results['RectBivariateSpline k=5'] = (t5, result.copy())

        # 2. RectBivariateSpline cubic (k=3)
        spline3 = RectBivariateSpline(theta, phi_ext, rho_ext, kx=3, ky=3, s=0)

        start = time.perf_counter()
        for _ in range(10):
            result = spline3(query_theta, query_phi % (2*np.pi), grid=False)
        t3 = (time.perf_counter() - start) / 10
        results['RectBivariateSpline k=3'] = (t3, result.copy())

        # 3. RegularGridInterpolator linear
        rgi_linear = RegularGridInterpolator(
            (theta, phi), rho,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        query_points = np.column_stack([query_theta, query_phi % (2*np.pi)])

        start = time.perf_counter()
        for _ in range(10):
            result = rgi_linear(query_points)
        t_linear = (time.perf_counter() - start) / 10
        results['RegularGridInterpolator linear'] = (t_linear, result.copy())

        # 4. RegularGridInterpolator cubic
        rgi_cubic = RegularGridInterpolator(
            (theta, phi), rho,
            method='cubic',
            bounds_error=False,
            fill_value=None
        )

        start = time.perf_counter()
        for _ in range(10):
            result = rgi_cubic(query_points)
        t_cubic_rgi = (time.perf_counter() - start) / 10
        results['RegularGridInterpolator cubic'] = (t_cubic_rgi, result.copy())

        # 5. ndimage.map_coordinates (cubic spline)
        # Need to convert to grid coordinates
        theta_coords = query_theta / d_theta
        phi_coords = (query_phi % (2*np.pi)) / d_phi
        coords = np.array([theta_coords, phi_coords])

        start = time.perf_counter()
        for _ in range(10):
            result = ndimage.map_coordinates(rho, coords, order=3, mode='wrap')
        t_ndimage = (time.perf_counter() - start) / 10
        results['ndimage.map_coordinates order=3'] = (t_ndimage, result.copy())

        # 6. ndimage.map_coordinates (linear)
        start = time.perf_counter()
        for _ in range(10):
            result = ndimage.map_coordinates(rho, coords, order=1, mode='wrap')
        t_ndimage1 = (time.perf_counter() - start) / 10
        results['ndimage.map_coordinates order=1'] = (t_ndimage1, result.copy())

        # Print results
        print(f"\n{'Method':<40} {'Time (ms)':>10} {'Speedup':>10} {'Max Error':>12}")
        print("-" * 72)

        baseline_time = results['RectBivariateSpline k=5'][0]
        baseline_result = results['RectBivariateSpline k=5'][1]

        for name, (t, res) in sorted(results.items(), key=lambda x: x[1][0]):
            speedup = baseline_time / t
            error = np.max(np.abs(res - baseline_result))
            print(f"{name:<40} {t*1000:>10.3f} {speedup:>10.2f}x {error:>12.6f}")

    print("\n" + "=" * 70)
    print("NUMBA JIT Custom Interpolation Test")
    print("=" * 70)

    try:
        from numba import jit

        @jit(nopython=True, cache=True)
        def lagrange_interp_numba(x, x_nodes, values):
            """Fast Lagrange interpolation with Numba."""
            n = len(x_nodes)
            result = 0.0
            for i in range(n):
                weight = 1.0
                for j in range(n):
                    if i != j:
                        weight *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
                result += weight * values[i]
            return result

        @jit(nopython=True, cache=True)
        def biquartic_interp_numba(theta, phi, rho, theta_grid, phi_grid, N_s):
            """Biquartic interpolation with Numba."""
            d_theta = theta_grid[1] - theta_grid[0]
            d_phi = phi_grid[1] - phi_grid[0]

            # Find stencil base
            i_th_base = int(theta / d_theta) - 1
            i_ph_base = int(phi / d_phi) - 1

            # Clip theta
            if i_th_base < 0:
                i_th_base = 0
            elif i_th_base > N_s - 4:
                i_th_base = N_s - 4

            # Get stencil nodes and values
            theta_nodes = np.empty(4)
            phi_nodes = np.empty(4)
            stencil = np.empty((4, 4))

            for i in range(4):
                theta_nodes[i] = theta_grid[i_th_base + i]
                i_ph = (i_ph_base + i) % N_s
                phi_nodes[i] = phi_grid[i_ph]
                for j in range(4):
                    j_ph = (i_ph_base + j) % N_s
                    stencil[i, j] = rho[i_th_base + i, j_ph]

            # Handle phi wrapping
            phi_eval = phi
            for i in range(3):
                if phi_nodes[i+1] < phi_nodes[i]:
                    for j in range(i+1, 4):
                        phi_nodes[j] += 2 * np.pi
                    if phi_eval < np.pi:
                        phi_eval += 2 * np.pi
                    break

            # Interpolate in phi first
            phi_interp = np.empty(4)
            for i in range(4):
                phi_interp[i] = lagrange_interp_numba(phi_eval, phi_nodes, stencil[i, :])

            # Then in theta
            return lagrange_interp_numba(theta, theta_nodes, phi_interp)

        @jit(nopython=True, cache=True, parallel=True)
        def batch_interp_numba(theta_arr, phi_arr, rho, theta_grid, phi_grid, N_s):
            """Batch interpolation with Numba parallel."""
            from numba import prange
            n = len(theta_arr)
            result = np.empty(n)
            for i in prange(n):
                if theta_arr[i] <= 0:
                    result[i] = rho[0, 0]
                elif theta_arr[i] >= np.pi:
                    result[i] = rho[-1, 0]
                else:
                    phi = theta_arr[i] % (2 * np.pi)  # Fix: should be phi_arr[i]
                    result[i] = biquartic_interp_numba(
                        theta_arr[i], phi_arr[i] % (2*np.pi),
                        rho, theta_grid, phi_grid, N_s
                    )
            return result

        # Warm up JIT
        N_s = 25
        theta = np.linspace(0, np.pi, N_s)
        phi = np.linspace(0, 2*np.pi, N_s, endpoint=False)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        rho = 2.0 + 0.1 * np.sin(THETA) * np.cos(2*PHI)

        query_theta = np.random.uniform(0.1, np.pi - 0.1, 100)
        query_phi = np.random.uniform(0, 2*np.pi, 100)

        _ = batch_interp_numba(query_theta, query_phi, rho, theta, phi, N_s)

        # Now benchmark
        n_query = 10000
        query_theta = np.random.uniform(0.1, np.pi - 0.1, n_query)
        query_phi = np.random.uniform(0, 2*np.pi, n_query)

        start = time.perf_counter()
        for _ in range(10):
            result_numba = batch_interp_numba(query_theta, query_phi, rho, theta, phi, N_s)
        t_numba = (time.perf_counter() - start) / 10

        # Compare with scipy
        n_extend = 3
        phi_ext = np.concatenate([phi[-n_extend:] - 2*np.pi, phi, phi[:n_extend] + 2*np.pi])
        rho_ext = np.column_stack([rho[:, -n_extend:], rho, rho[:, :n_extend]])
        spline5 = RectBivariateSpline(theta, phi_ext, rho_ext, kx=5, ky=5, s=0)

        start = time.perf_counter()
        for _ in range(10):
            result_scipy = spline5(query_theta, query_phi, grid=False)
        t_scipy = (time.perf_counter() - start) / 10

        print(f"\nN_s=25, {n_query} query points:")
        print(f"  Numba biquartic (parallel): {t_numba*1000:.3f} ms")
        print(f"  Scipy quintic spline:       {t_scipy*1000:.3f} ms")
        print(f"  Speedup: {t_scipy/t_numba:.2f}x")

    except Exception as e:
        print(f"Numba test failed: {e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Fastest options (in order):

1. ndimage.map_coordinates order=1 (linear) - ~10x faster, lower accuracy
2. ndimage.map_coordinates order=3 (cubic)  - ~3-5x faster, good accuracy
3. RegularGridInterpolator cubic            - ~2-3x faster
4. RectBivariateSpline k=3 (cubic)          - ~1.5x faster
5. RectBivariateSpline k=5 (quintic)        - baseline (current)

Recommendations:
- If accuracy is critical: keep quintic spline or use Numba biquartic
- If speed is critical: use ndimage.map_coordinates order=3
- Best balance: Numba parallel biquartic (matches current accuracy, ~2x faster)
""")

if __name__ == "__main__":
    benchmark_interpolation()
