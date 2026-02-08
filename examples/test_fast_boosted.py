#!/usr/bin/env python
"""Test the fast boosted metric implementation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time

from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.boosted import BoostedMetric
from ahfinder.metrics.boosted_fast import FastBoostedMetric

def test_correctness():
    """Verify fast boosted gives same results as original."""
    print("Testing correctness...")

    M = 1.0
    v = 0.3
    velocity = np.array([v, 0.0, 0.0])

    base = SchwarzschildMetric(M=M)
    original = BoostedMetric(base, velocity)
    fast = FastBoostedMetric(base, velocity)

    # Test at several points
    test_points = [
        (3.0, 0.0, 0.0),
        (2.0, 1.0, 0.5),
        (0.0, 3.0, 1.0),
    ]

    for x, y, z in test_points:
        print(f"\n  Point ({x}, {y}, {z}):")

        # Test gamma
        g_orig = original.gamma(x, y, z)
        g_fast = fast.gamma(x, y, z)
        err = np.max(np.abs(g_orig - g_fast))
        print(f"    gamma: max error = {err:.2e}")

        # Test dgamma
        dg_orig = original.dgamma(x, y, z)
        dg_fast = fast.dgamma(x, y, z)
        err = np.max(np.abs(dg_orig - dg_fast))
        print(f"    dgamma: max error = {err:.2e}")

        # Test extrinsic curvature
        K_orig = original.extrinsic_curvature(x, y, z)
        K_fast = fast.extrinsic_curvature(x, y, z)
        err = np.max(np.abs(K_orig - K_fast))
        print(f"    K: max error = {err:.2e}")


def test_speed():
    """Compare speed of original vs fast implementation."""
    print("\n\nTesting speed...")

    M = 1.0
    v = 0.3
    velocity = np.array([v, 0.0, 0.0])

    base = SchwarzschildMetric(M=M)
    original = BoostedMetric(base, velocity)
    fast = FastBoostedMetric(base, velocity)

    n_calls = 100
    x, y, z = 3.0, 1.0, 0.5

    # Test gamma
    t0 = time.time()
    for _ in range(n_calls):
        original.gamma(x, y, z)
    t_orig_gamma = time.time() - t0

    t0 = time.time()
    for _ in range(n_calls):
        fast.gamma(x, y, z)
    t_fast_gamma = time.time() - t0

    print(f"\n  gamma() x {n_calls}:")
    print(f"    Original: {t_orig_gamma*1000:.2f} ms")
    print(f"    Fast:     {t_fast_gamma*1000:.2f} ms")
    print(f"    Speedup:  {t_orig_gamma/t_fast_gamma:.2f}x")

    # Test dgamma
    t0 = time.time()
    for _ in range(n_calls):
        original.dgamma(x, y, z)
    t_orig_dgamma = time.time() - t0

    t0 = time.time()
    for _ in range(n_calls):
        fast.dgamma(x, y, z)
    t_fast_dgamma = time.time() - t0

    print(f"\n  dgamma() x {n_calls}:")
    print(f"    Original: {t_orig_dgamma*1000:.2f} ms")
    print(f"    Fast:     {t_fast_dgamma*1000:.2f} ms")
    print(f"    Speedup:  {t_orig_dgamma/t_fast_dgamma:.2f}x")

    # Test extrinsic_curvature
    n_K = 50  # Fewer calls since this is slower
    t0 = time.time()
    for _ in range(n_K):
        original.extrinsic_curvature(x, y, z)
    t_orig_K = time.time() - t0

    t0 = time.time()
    for _ in range(n_K):
        fast.extrinsic_curvature(x, y, z)
    t_fast_K = time.time() - t0

    print(f"\n  extrinsic_curvature() x {n_K}:")
    print(f"    Original: {t_orig_K*1000:.2f} ms")
    print(f"    Fast:     {t_fast_K*1000:.2f} ms")
    print(f"    Speedup:  {t_orig_K/t_fast_K:.2f}x")


if __name__ == "__main__":
    test_correctness()
    test_speed()
    print("\nDone!")