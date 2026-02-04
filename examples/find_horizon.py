#!/usr/bin/env python3
"""
Example: Finding apparent horizons in various spacetimes.

This script demonstrates how to use the AHFinder package to locate
apparent horizons for Schwarzschild, Kerr, and boosted black holes.

Reference: Huq, Choptuik & Matzner (2000) - arXiv:gr-qc/0002076
"""

import numpy as np
import sys
import os

# Add src to path if running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ahfinder import ApparentHorizonFinder
from ahfinder.metrics import SchwarzschildMetric, KerrMetric, BoostedMetric


def example_schwarzschild():
    """Find the Schwarzschild horizon."""
    print("=" * 60)
    print("Schwarzschild Black Hole (M = 1)")
    print("=" * 60)

    M = 1.0
    metric = SchwarzschildMetric(M=M)

    # Create finder with moderate resolution
    finder = ApparentHorizonFinder(metric, N_s=33)

    # Find the horizon starting from initial guess r = 2M
    rho = finder.find(initial_radius=2.0, tol=1e-8, verbose=True)

    # Analyze results
    r_mean = finder.horizon_radius_average(rho)
    r_eq = finder.horizon_radius_equatorial(rho)
    r_polar = finder.horizon_radius_polar(rho)
    area = finder.horizon_area(rho)
    M_irr = finder.irreducible_mass(rho)

    print(f"\nResults:")
    print(f"  Mean radius:       {r_mean:.6f} (expected: {2*M:.6f})")
    print(f"  Equatorial radius: {r_eq:.6f}")
    print(f"  Polar radius:      {r_polar:.6f}")
    print(f"  Horizon area:      {area:.6f} (expected: {16*np.pi*M**2:.6f})")
    print(f"  Irreducible mass:  {M_irr:.6f} (expected: {M:.6f})")

    # Check error
    error = abs(r_mean - 2*M) / (2*M) * 100
    print(f"  Radius error:      {error:.4f}%")

    return rho, finder


def example_kerr():
    """Find the Kerr horizon for different spin parameters."""
    print("\n" + "=" * 60)
    print("Kerr Black Hole")
    print("=" * 60)

    M = 1.0
    spins = [0.0, 0.5, 0.9]

    for a in spins:
        print(f"\nSpin parameter a/M = {a}")
        print("-" * 40)

        metric = KerrMetric(M=M, a=a)

        # Analytical horizon radius
        r_plus = M + np.sqrt(M**2 - a**2)
        print(f"  Analytical r_+ = {r_plus:.6f}")

        finder = ApparentHorizonFinder(metric, N_s=33)

        try:
            rho = finder.find(initial_radius=r_plus, tol=1e-6, verbose=False)

            r_eq = finder.horizon_radius_equatorial(rho)
            r_polar = finder.horizon_radius_polar(rho)
            area = finder.horizon_area(rho)
            expected_area = metric.horizon_area()

            print(f"  Found equatorial:  {r_eq:.6f}")
            print(f"  Found polar:       {r_polar:.6f}")
            print(f"  Oblateness (r_eq - r_polar): {r_eq - r_polar:.6f}")
            print(f"  Horizon area:      {area:.6f} (expected: {expected_area:.6f})")
            print(f"  Area error:        {abs(area - expected_area)/expected_area*100:.2f}%")

        except Exception as e:
            print(f"  Failed to converge: {e}")


def example_boosted():
    """Find the horizon of a boosted Schwarzschild black hole."""
    print("\n" + "=" * 60)
    print("Boosted Schwarzschild Black Hole")
    print("=" * 60)

    M = 1.0
    base_metric = SchwarzschildMetric(M=M)

    velocities = [0.0, 0.3, 0.6]

    # Find unboosted area for comparison
    finder_base = ApparentHorizonFinder(base_metric, N_s=33)
    rho_base = finder_base.find(initial_radius=2.0, tol=1e-7, verbose=False)
    area_base = finder_base.horizon_area(rho_base)

    for v in velocities:
        print(f"\nBoost velocity v = {v}c")
        print("-" * 40)

        if v == 0:
            gamma = 1.0
        else:
            gamma = 1.0 / np.sqrt(1 - v**2)

        print(f"  Lorentz factor γ = {gamma:.4f}")

        if v > 0:
            velocity = np.array([v, 0.0, 0.0])
            metric = BoostedMetric(base_metric, velocity)
        else:
            metric = base_metric

        finder = ApparentHorizonFinder(metric, N_s=33)

        try:
            rho = finder.find(initial_radius=2.0, tol=1e-6, verbose=False)

            # Get extent in each direction
            x, y, z = finder.horizon_coordinates(rho)

            x_extent = x.max() - x.min()
            y_extent = y.max() - y.min()
            z_extent = z.max() - z.min()

            area = finder.horizon_area(rho)

            print(f"  Extent in x (boost dir): {x_extent:.4f} (expected: {4*M/gamma:.4f})")
            print(f"  Extent in y (perp):      {y_extent:.4f} (expected: {4*M:.4f})")
            print(f"  Extent in z (perp):      {z_extent:.4f} (expected: {4*M:.4f})")
            print(f"  Horizon area:            {area:.4f}")
            print(f"  Area change from v=0:    {(area - area_base)/area_base*100:.2f}%")

        except Exception as e:
            print(f"  Failed: {e}")


def example_convergence():
    """Demonstrate convergence with mesh refinement."""
    print("\n" + "=" * 60)
    print("Convergence Study")
    print("=" * 60)

    M = 1.0
    metric = SchwarzschildMetric(M=M)

    resolutions = [17, 25, 33, 41, 49]
    expected_radius = 2.0 * M

    print("\n  N_s    Mean Radius    Error        Iterations")
    print("-" * 50)

    for N_s in resolutions:
        finder = ApparentHorizonFinder(metric, N_s=N_s)
        finder._solver = None  # Force new solver

        try:
            rho = finder.find(initial_radius=2.0, tol=1e-9, max_iter=30, verbose=False)
            r_mean = finder.horizon_radius_average(rho)
            error = abs(r_mean - expected_radius)

            res_hist, delta_hist = finder.convergence_history
            n_iter = len(res_hist)

            print(f"  {N_s:3d}    {r_mean:.8f}   {error:.2e}     {n_iter}")

        except Exception as e:
            print(f"  {N_s:3d}    Failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# Apparent Horizon Finder - Examples")
    print("# Based on Huq, Choptuik & Matzner (2000)")
    print("#" * 60)

    # Run examples
    example_schwarzschild()
    example_kerr()
    example_boosted()
    example_convergence()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
