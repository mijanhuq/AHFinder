#!/usr/bin/env python
"""Profile the full horizon finding process."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cProfile
import pstats
from io import StringIO

from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.boosted import BoostedMetric

def profile_horizon_finding():
    """Profile horizon finding for boosted Schwarzschild."""
    M = 1.0
    base = SchwarzschildMetric(M=M)
    velocity = np.array([0.3, 0.0, 0.0])
    boosted = BoostedMetric(base, velocity)

    N_s = 17  # Small for speed

    print(f"Profiling horizon finding for boosted Schwarzschild")
    print(f"Resolution: N_s = {N_s}")
    print()

    finder = ApparentHorizonFinder(boosted, N_s=N_s)

    pr = cProfile.Profile()
    pr.enable()
    try:
        rho = finder.find(initial_radius=2.0, tol=1e-4, max_iter=20, verbose=True)
        print(f"\nHorizon found!")
    except Exception as e:
        print(f"\nFailed: {e}")
    pr.disable()

    print("\n" + "="*60)
    print("PROFILE RESULTS")
    print("="*60)

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

if __name__ == "__main__":
    profile_horizon_finding()