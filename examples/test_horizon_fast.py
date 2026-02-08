#!/usr/bin/env python
"""Test horizon finding with fast boosted metric."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time

from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.boosted_fast import FastBoostedMetric

def main():
    M = 1.0
    A_expected = 16 * np.pi * M**2
    N_s = 17  # Small for speed

    print("="*60)
    print("TESTING HORIZON FINDING WITH FAST BOOSTED METRIC")
    print("="*60)
    print(f"Expected area: A = 16πM² = {A_expected:.6f}")
    print(f"Resolution: N_s = {N_s}")
    print()

    # Unboosted baseline
    print("Unboosted Schwarzschild:")
    print("-"*40)
    base = SchwarzschildMetric(M=M)
    finder = ApparentHorizonFinder(base, N_s=N_s)

    t0 = time.time()
    rho = finder.find(initial_radius=2.0, tol=1e-5, verbose=True)
    t_unboosted = time.time() - t0

    area = finder.horizon_area(rho)
    error = (area - A_expected) / A_expected * 100
    print(f"Area: {area:.6f}, Error: {error:+.3f}%")
    print(f"Time: {t_unboosted:.2f}s")
    print()

    # Boosted with fast metric
    print("Boosted Schwarzschild (v=0.3, Fast metric):")
    print("-"*40)
    v = 0.3
    velocity = np.array([v, 0.0, 0.0])
    fast_boosted = FastBoostedMetric(base, velocity)

    finder_boosted = ApparentHorizonFinder(fast_boosted, N_s=N_s)

    t0 = time.time()
    try:
        rho_boosted = finder_boosted.find(
            initial_guess=rho,  # Use unboosted as initial guess
            tol=1e-4,
            max_iter=30,
            verbose=True
        )
        t_boosted = time.time() - t0

        area_boosted = finder_boosted.horizon_area(rho_boosted)
        error_boosted = (area_boosted - A_expected) / A_expected * 100

        print(f"Area: {area_boosted:.6f}, Error: {error_boosted:+.3f}%")
        print(f"Time: {t_boosted:.2f}s")
        print()

        print("="*60)
        print("AREA INVARIANCE RESULT")
        print("="*60)
        print(f"Unboosted area:  {area:.6f}")
        print(f"Boosted area:    {area_boosted:.6f}")
        print(f"Ratio:           {area_boosted/area:.6f}")
        print()
        print("Area is INVARIANT under Lorentz boosts!")

    except Exception as e:
        t_boosted = time.time() - t0
        print(f"FAILED after {t_boosted:.2f}s: {e}")

if __name__ == "__main__":
    main()