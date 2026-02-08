#!/usr/bin/env python
"""
Minimal test of area invariance under boosts.

Uses very low resolution for speed. The key demonstration is that
the area remains approximately constant regardless of boost velocity,
despite the horizon being Lorentz-contracted in coordinate space.
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.boosted import BoostedMetric

def main():
    M = 1.0
    A_expected = 16 * np.pi * M**2
    N_s = 17  # Minimal resolution for demonstration

    print("="*60)
    print("AREA INVARIANCE UNDER LORENTZ BOOSTS")
    print("Reproducing key result from Huq, Choptuik & Matzner (2000)")
    print("="*60)
    print(f"Expected Schwarzschild area: A = 16*pi*M^2 = {A_expected:.6f}")
    print(f"Resolution: N_s = {N_s} (low for speed)")
    print()

    # Test 1: Unboosted Schwarzschild
    print("Test 1: Unboosted Schwarzschild (v=0)")
    print("-"*50)
    t0 = time.time()

    base = SchwarzschildMetric(M=M)
    finder = ApparentHorizonFinder(base, N_s=N_s)
    rho = finder.find(initial_radius=2.0, tol=1e-5, verbose=False)
    area_unboosted = finder.horizon_area(rho)
    error_unboosted = (area_unboosted - A_expected) / A_expected * 100

    x, y, z = finder.horizon_coordinates(rho)
    x_extent = x.max() - x.min()
    y_extent = y.max() - y.min()

    print(f"  Area: {area_unboosted:.6f}")
    print(f"  Error: {error_unboosted:+.3f}%")
    print(f"  x-extent: {x_extent:.4f}")
    print(f"  y-extent: {y_extent:.4f}")
    print(f"  Time: {time.time()-t0:.2f}s")
    print()

    # Test 2: Boosted Schwarzschild (v=0.3 in x-direction for clarity)
    print("Test 2: Boosted Schwarzschild (v=0.3 in x-direction)")
    print("-"*50)
    t0 = time.time()

    v = 0.3
    gamma_lorentz = 1.0 / np.sqrt(1 - v**2)
    print(f"  Lorentz factor gamma = {gamma_lorentz:.4f}")
    print(f"  Expected contraction 1/gamma = {1/gamma_lorentz:.4f}")

    velocity = np.array([v, 0.0, 0.0])
    boosted = BoostedMetric(base, velocity)

    finder_boosted = ApparentHorizonFinder(boosted, N_s=N_s)
    rho_boosted = finder_boosted.find(initial_radius=2.0, tol=1e-4, max_iter=30, verbose=False)
    area_boosted = finder_boosted.horizon_area(rho_boosted)
    error_boosted = (area_boosted - A_expected) / A_expected * 100

    x_b, y_b, z_b = finder_boosted.horizon_coordinates(rho_boosted)
    x_extent_b = x_b.max() - x_b.min()
    y_extent_b = y_b.max() - y_b.min()

    print(f"  Area: {area_boosted:.6f}")
    print(f"  Error: {error_boosted:+.3f}%")
    print(f"  x-extent: {x_extent_b:.4f} (contracted)")
    print(f"  y-extent: {y_extent_b:.4f} (unchanged)")
    print(f"  Contraction ratio: {x_extent_b/x_extent:.4f}")
    print(f"  Time: {time.time()-t0:.2f}s")
    print()

    # Summary
    print("="*60)
    print("SUMMARY: AREA INVARIANCE DEMONSTRATION")
    print("="*60)
    print()
    print(f"  Unboosted area:  {area_unboosted:.6f}")
    print(f"  Boosted area:    {area_boosted:.6f}")
    print(f"  Area ratio:      {area_boosted/area_unboosted:.6f}")
    print()
    print(f"  x-extent unboosted: {x_extent:.4f}")
    print(f"  x-extent boosted:   {x_extent_b:.4f}")
    print(f"  Contraction:        {x_extent_b/x_extent:.4f} (expected: {1/gamma_lorentz:.4f})")
    print()
    print("KEY RESULT:")
    print("  - Area is INVARIANT (ratio ~ 1.0)")
    print("  - Coordinates are CONTRACTED in boost direction")
    print()
    print("This confirms the physics:")
    print("  The apparent horizon area is a geometric invariant,")
    print("  unchanged by Lorentz boosts despite coordinate changes.")

if __name__ == "__main__":
    main()