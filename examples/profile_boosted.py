#!/usr/bin/env python
"""Profile the boosted metric to find bottlenecks."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cProfile
import pstats
from io import StringIO

from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.boosted import BoostedMetric

def profile_metric_calls():
    """Profile basic metric operations."""
    M = 1.0
    base = SchwarzschildMetric(M=M)
    velocity = np.array([0.3, 0.0, 0.0])
    boosted = BoostedMetric(base, velocity)

    # Test point outside horizon
    x, y, z = 3.0, 1.0, 0.5

    print("Profiling 1000 calls to each method...")
    print()

    # Profile gamma
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(1000):
        boosted.gamma(x, y, z)
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print("gamma() calls:")
    print(s.getvalue())

    # Profile dgamma (this is likely the slow one)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):  # Fewer calls since it's slower
        boosted.dgamma(x, y, z)
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print("dgamma() calls (100x):")
    print(s.getvalue())

    # Profile extrinsic_curvature
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):
        boosted.extrinsic_curvature(x, y, z)
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print("extrinsic_curvature() calls (100x):")
    print(s.getvalue())

if __name__ == "__main__":
    profile_metric_calls()