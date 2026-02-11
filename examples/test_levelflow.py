"""
Test the Level Flow method for finding apparent horizons.

Demonstrates:
1. Pure Level Flow evolution
2. Hybrid method (Level Flow + Newton)
3. Comparison with Newton alone
4. Application to Kerr black holes
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/mijan/PycharmProjects/AHFinder/src')

from ahfinder import ApparentHorizonFinder, LevelFlowFinder
from ahfinder.metrics import SchwarzschildMetric, KerrMetric


def test_schwarzschild():
    """Test on Schwarzschild (spherical horizon at r=2M)."""
    print("=" * 60)
    print("Test 1: Schwarzschild Black Hole (M=1, horizon at r=2)")
    print("=" * 60)

    M = 1.0
    metric = SchwarzschildMetric(M=M)
    N_s = 17

    # Newton method
    print("\n1.1 Newton method (from r=3):")
    finder_newton = ApparentHorizonFinder(metric, N_s=N_s, use_vectorized_jacobian=True)
    t0 = time.perf_counter()
    rho_newton = finder_newton.find(initial_radius=3.0, tol=1e-8, verbose=False)
    t_newton = time.perf_counter() - t0
    print(f"  Time: {t_newton:.2f}s, Mean radius: {np.mean(rho_newton):.6f}")

    # Hybrid method
    print("\n1.2 Hybrid method (Level Flow + Newton, from r=3):")
    finder_flow = LevelFlowFinder(metric, N_s=N_s)
    t0 = time.perf_counter()
    rho_hybrid, info = finder_flow.find_hybrid(
        initial_radius=3.0,
        flow_tol=0.5,
        newton_tol=1e-8,
        verbose=False
    )
    t_hybrid = time.perf_counter() - t0
    print(f"  Time: {t_hybrid:.2f}s, Mean radius: {np.mean(rho_hybrid):.6f}")
    print(f"  Level Flow steps: {info['flow_steps']}")

    return rho_newton, rho_hybrid


def test_kerr():
    """Test on Kerr (oblate horizon)."""
    print("\n" + "=" * 60)
    print("Test 2: Kerr Black Hole (M=1, a=0.7)")
    print("=" * 60)

    M = 1.0
    a = 0.7
    metric = KerrMetric(M=M, a=a)
    N_s = 17

    # Expected horizon radius (approximate - Kerr horizon is oblate)
    r_plus = M + np.sqrt(M**2 - a**2)
    print(f"Expected equatorial radius: ~{r_plus:.4f}")

    # Newton method
    print("\n2.1 Newton method (from r=2.5):")
    finder_newton = ApparentHorizonFinder(metric, N_s=N_s, use_vectorized_jacobian=True)
    t0 = time.perf_counter()
    try:
        rho_newton = finder_newton.find(initial_radius=2.5, tol=1e-8, verbose=False)
        print(f"  Time: {time.perf_counter()-t0:.2f}s")
        print(f"  ρ range: [{np.min(rho_newton):.4f}, {np.max(rho_newton):.4f}]")
    except Exception as e:
        print(f"  Failed: {e}")
        rho_newton = None

    # Hybrid method
    print("\n2.2 Hybrid method (from r=2.5):")
    finder_flow = LevelFlowFinder(metric, N_s=N_s)
    t0 = time.perf_counter()
    try:
        rho_hybrid, info = finder_flow.find_hybrid(
            initial_radius=2.5,
            flow_tol=0.5,
            newton_tol=1e-8,
            max_flow_steps=500,
            verbose=False
        )
        print(f"  Time: {time.perf_counter()-t0:.2f}s")
        print(f"  ρ range: [{np.min(rho_hybrid):.4f}, {np.max(rho_hybrid):.4f}]")
        print(f"  Level Flow steps: {info['flow_steps']}")
    except Exception as e:
        print(f"  Failed: {e}")
        rho_hybrid = None

    return rho_newton, rho_hybrid


def test_evolution_visualization():
    """Show the Level Flow evolution over time."""
    print("\n" + "=" * 60)
    print("Test 3: Level Flow Evolution Visualization")
    print("=" * 60)

    metric = SchwarzschildMetric(M=1.0)
    finder = LevelFlowFinder(metric, N_s=13)

    print("\nEvolution from r=4 toward horizon at r=2:")
    result = finder.evolve(
        initial_radius=4.0,
        tol=1e-6,
        max_steps=1000,
        save_history=True,
        history_interval=20,
        verbose=True
    )

    if result.history:
        print("\nEvolution history:")
        print(f"{'Step':>6} {'t':>8} {'||Θ||':>12} {'ρ_mean':>10}")
        print("-" * 40)
        for h in result.history[:10]:  # First 10 snapshots
            print(f"{h['step']:>6} {h['t']:>8.2f} {h['residual_norm']:>12.4e} {h['rho_mean']:>10.4f}")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Level Flow Method Tests")
    print("#" * 60)

    test_schwarzschild()
    test_kerr()
    test_evolution_visualization()

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
