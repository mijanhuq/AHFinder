#!/usr/bin/env python3
"""
Benchmark to understand where the overhead is coming from.
"""

import numpy as np
import time
from numba import jit

# Import JIT functions
from ahfinder.metrics.boosted_kerr_fast import (
    _compute_H_l_rest_frame,
    _compute_dH_dl_numerical,
    _compute_dgamma_from_H_l,
    _compute_christoffel_from_dgamma,
    _compute_K_from_H_l_chris,
    _transform_derivatives_to_lab,
)


@jit(nopython=True, cache=True)
def compute_all_in_one(x_rest, M, a, Lambda):
    """Compute everything in a single JIT function."""
    # Base computation
    H, l, r = _compute_H_l_rest_frame(x_rest, M, a)
    dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, M, a)
    dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, Lambda)

    # gamma_inv
    f = 2 * H / (1 + 2 * H)
    gamma_inv = np.eye(3) - f * np.outer(l, l)

    # dgamma
    dgamma = _compute_dgamma_from_H_l(H, l, dH, dl)

    # christoffel
    chris = _compute_christoffel_from_dgamma(gamma_inv, dgamma)

    # K
    K = _compute_K_from_H_l_chris(H, l, dH, dl, chris)

    # K_trace
    K_trace = 0.0
    for i in range(3):
        for j in range(3):
            K_trace += gamma_inv[i, j] * K[i, j]

    return gamma_inv, dgamma, K, K_trace


def main():
    print("=" * 70)
    print("Overhead Analysis")
    print("=" * 70)

    M, a = 1.0, 0.5
    velocity = np.array([0.3, 0.0, 0.0])
    v_mag = np.linalg.norm(velocity)
    lorentz_gamma = 1.0 / np.sqrt(1 - v_mag**2)
    n_hat = velocity / v_mag
    Lambda = np.eye(3) + (lorentz_gamma - 1) * np.outer(n_hat, n_hat)

    # Test points
    n_points = 10000
    np.random.seed(42)
    theta = np.random.uniform(0.1, np.pi - 0.1, n_points)
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    r = np.random.uniform(1.8, 2.2, n_points)
    x_pts = r * np.sin(theta) * np.cos(phi)
    y_pts = r * np.sin(theta) * np.sin(phi)
    z_pts = r * np.cos(theta)

    # Warm up
    print("\nWarming up JIT...")
    for i in range(100):
        x_lab = np.array([x_pts[i], y_pts[i], z_pts[i]])
        x_rest = Lambda @ x_lab
        _ = compute_all_in_one(x_rest, M, a, Lambda)

    # Benchmark all-in-one JIT
    print(f"\nBenchmarking {n_points} points...")

    start = time.perf_counter()
    for i in range(n_points):
        x_lab = np.array([x_pts[i], y_pts[i], z_pts[i]])
        x_rest = Lambda @ x_lab
        gamma_inv, dgamma, K, K_trace = compute_all_in_one(x_rest, M, a, Lambda)
    t_allinone = time.perf_counter() - start

    # Benchmark separate JIT calls (like cached version does internally)
    start = time.perf_counter()
    for i in range(n_points):
        x_lab = np.array([x_pts[i], y_pts[i], z_pts[i]])
        x_rest = Lambda @ x_lab

        H, l, r = _compute_H_l_rest_frame(x_rest, M, a)
        dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, M, a)
        dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, Lambda)

        f = 2 * H / (1 + 2 * H)
        gamma_inv = np.eye(3) - f * np.outer(l, l)

        dgamma = _compute_dgamma_from_H_l(H, l, dH, dl)
        chris = _compute_christoffel_from_dgamma(gamma_inv, dgamma)
        K = _compute_K_from_H_l_chris(H, l, dH, dl, chris)
        K_trace = np.sum(gamma_inv * K)
    t_separate = time.perf_counter() - start

    # Compare with cached class
    from ahfinder.metrics.boosted_kerr_fast_cached import FastBoostedKerrMetricCached
    metric = FastBoostedKerrMetricCached(M=M, a=a, velocity=velocity)

    # warm up
    for i in range(100):
        _ = metric.compute_all_geometric(x_pts[i], y_pts[i], z_pts[i])

    start = time.perf_counter()
    for i in range(n_points):
        gamma_inv, dgamma, K, K_trace = metric.compute_all_geometric(x_pts[i], y_pts[i], z_pts[i])
    t_class = time.perf_counter() - start

    print(f"\n{'Method':<45} {'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'All-in-one JIT (no Python overhead)':<45} {t_allinone*1000:>12.2f} {'1.0x':>10}")
    print(f"{'Separate JIT calls (raw)':<45} {t_separate*1000:>12.2f} {t_allinone/t_separate:>10.2f}x")
    print(f"{'Cached class method':<45} {t_class*1000:>12.2f} {t_allinone/t_class:>10.2f}x")

    # Calculate overheads
    print("\n" + "-" * 70)
    print(f"Overhead from separate JIT calls: {(t_separate - t_allinone)*1000:.2f} ms ({100*(t_separate-t_allinone)/t_allinone:.0f}%)")
    print(f"Overhead from class methods:      {(t_class - t_allinone)*1000:.2f} ms ({100*(t_class-t_allinone)/t_allinone:.0f}%)")

    # What about just the base computation?
    print("\n" + "=" * 70)
    print("Base computation breakdown")
    print("=" * 70)

    # Just H, l, dH, dl
    start = time.perf_counter()
    for i in range(n_points):
        x_lab = np.array([x_pts[i], y_pts[i], z_pts[i]])
        x_rest = Lambda @ x_lab
        H, l, r = _compute_H_l_rest_frame(x_rest, M, a)
        dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, M, a)
        dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, Lambda)
    t_base = time.perf_counter() - start

    # Just derived quantities (given H, l, dH, dl)
    # Pre-compute one set
    x_rest = Lambda @ np.array([x_pts[0], y_pts[0], z_pts[0]])
    H, l, r = _compute_H_l_rest_frame(x_rest, M, a)
    dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, M, a)
    dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, Lambda)

    start = time.perf_counter()
    for i in range(n_points):
        f = 2 * H / (1 + 2 * H)
        gamma_inv = np.eye(3) - f * np.outer(l, l)
        dgamma = _compute_dgamma_from_H_l(H, l, dH, dl)
        chris = _compute_christoffel_from_dgamma(gamma_inv, dgamma)
        K = _compute_K_from_H_l_chris(H, l, dH, dl, chris)
        K_trace = np.sum(gamma_inv * K)
    t_derived = time.perf_counter() - start

    print(f"\n{'Component':<40} {'Time (ms)':>12} {'%':>10}")
    print("-" * 65)
    print(f"{'Base (H, l, dH, dl)':<40} {t_base*1000:>12.2f} {100*t_base/(t_base+t_derived):>10.1f}%")
    print(f"{'Derived (gamma_inv, dgamma, K)':<40} {t_derived*1000:>12.2f} {100*t_derived/(t_base+t_derived):>10.1f}%")
    print(f"{'Total':<40} {(t_base+t_derived)*1000:>12.2f}")

if __name__ == "__main__":
    main()
