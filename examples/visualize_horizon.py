#!/usr/bin/env python3
"""
Example: Visualizing apparent horizons.

This script demonstrates visualization capabilities including
3D surface plots, cross-sections, and convergence plots.

Reference: Huq, Choptuik & Matzner (2000) - arXiv:gr-qc/0002076
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path if running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ahfinder import ApparentHorizonFinder
from ahfinder.metrics import SchwarzschildMetric, KerrMetric, BoostedMetric
from ahfinder.visualization import (
    plot_horizon_3d,
    plot_horizon_cross_section,
    plot_convergence,
    compare_horizons,
    plot_horizon_radius_profile
)


def visualize_schwarzschild():
    """Visualize the Schwarzschild horizon."""
    print("Finding Schwarzschild horizon...")

    metric = SchwarzschildMetric(M=1.0)
    finder = ApparentHorizonFinder(metric, N_s=41)
    rho = finder.find(initial_radius=2.0, tol=1e-8, verbose=False)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    plot_horizon_3d(rho, finder.mesh, ax=ax1, title="Schwarzschild Horizon (3D)")

    # XZ cross-section
    ax2 = fig.add_subplot(132)
    plot_horizon_cross_section(rho, finder.mesh, plane='xz', ax=ax2,
                               label='Horizon')
    ax2.set_title("XZ Cross-Section")

    # Convergence history
    ax3 = fig.add_subplot(133)
    res_hist, delta_hist = finder.convergence_history
    plot_convergence(res_hist, delta_hist, ax=ax3,
                    title="Newton Convergence")

    plt.tight_layout()
    plt.savefig('schwarzschild_horizon.png', dpi=150)
    print("Saved: schwarzschild_horizon.png")
    plt.close()


def visualize_kerr():
    """Visualize Kerr horizons for different spins."""
    print("Finding Kerr horizons...")

    M = 1.0
    spins = [0.0, 0.5, 0.9]
    colors = ['blue', 'green', 'red']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rho_list = []
    mesh = None

    for a, color in zip(spins, colors):
        print(f"  a/M = {a}...")
        metric = KerrMetric(M=M, a=a)
        r_plus = M + np.sqrt(M**2 - a**2)

        finder = ApparentHorizonFinder(metric, N_s=41)
        try:
            rho = finder.find(initial_radius=r_plus, tol=1e-6, verbose=False)
            rho_list.append(rho)
            mesh = finder.mesh
        except:
            print(f"    Failed for a={a}")
            continue

    if len(rho_list) > 0:
        # XZ cross-sections comparison
        ax = axes[0]
        labels = [f'a/M = {a}' for a in spins[:len(rho_list)]]
        compare_horizons(rho_list, mesh, labels, plane='xz', ax=ax,
                        colors=colors[:len(rho_list)],
                        title="Kerr Horizons (XZ plane)")

        # XY cross-sections (equatorial)
        ax = axes[1]
        compare_horizons(rho_list, mesh, labels, plane='xy', ax=ax,
                        colors=colors[:len(rho_list)],
                        title="Kerr Horizons (Equatorial)")

        # Radius profiles
        ax = axes[2]
        for rho, label, color in zip(rho_list, labels, colors):
            plot_horizon_radius_profile(rho, mesh, ax=ax, label=label, color=color)
        ax.set_title("Radius vs θ")
        ax.legend()

    plt.tight_layout()
    plt.savefig('kerr_horizons.png', dpi=150)
    print("Saved: kerr_horizons.png")
    plt.close()


def visualize_boosted():
    """Visualize boosted Schwarzschild horizons."""
    print("Finding boosted horizons...")

    M = 1.0
    base = SchwarzschildMetric(M=M)

    velocities = [0.0, 0.3, 0.6]
    colors = ['blue', 'green', 'red']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rho_list = []
    mesh = None

    for v, color in zip(velocities, colors):
        print(f"  v/c = {v}...")

        if v > 0:
            velocity = np.array([v, 0.0, 0.0])
            metric = BoostedMetric(base, velocity)
        else:
            metric = base

        finder = ApparentHorizonFinder(metric, N_s=41)
        try:
            rho = finder.find(initial_radius=2.0, tol=1e-6, verbose=False)
            rho_list.append(rho)
            mesh = finder.mesh
        except:
            print(f"    Failed for v={v}")
            continue

    if len(rho_list) > 0:
        # XZ cross-sections (shows Lorentz contraction in x)
        ax = axes[0]
        labels = [f'v/c = {v}' for v in velocities[:len(rho_list)]]
        compare_horizons(rho_list, mesh, labels, plane='xz', ax=ax,
                        colors=colors[:len(rho_list)],
                        title="Boosted Horizons (XZ plane)")

        # XY cross-sections (equatorial)
        ax = axes[1]
        compare_horizons(rho_list, mesh, labels, plane='xy', ax=ax,
                        colors=colors[:len(rho_list)],
                        title="Boosted Horizons (Equatorial)")

        # YZ cross-sections (perpendicular to boost)
        ax = axes[2]
        compare_horizons(rho_list, mesh, labels, plane='yz', ax=ax,
                        colors=colors[:len(rho_list)],
                        title="Boosted Horizons (YZ plane)")

    plt.tight_layout()
    plt.savefig('boosted_horizons.png', dpi=150)
    print("Saved: boosted_horizons.png")
    plt.close()


def visualize_3d_comparison():
    """Create 3D comparison plots."""
    print("Creating 3D comparison...")

    fig = plt.figure(figsize=(15, 5))

    # Schwarzschild
    ax1 = fig.add_subplot(131, projection='3d')
    metric = SchwarzschildMetric(M=1.0)
    finder = ApparentHorizonFinder(metric, N_s=33)
    rho = finder.find(initial_radius=2.0, tol=1e-7, verbose=False)
    plot_horizon_3d(rho, finder.mesh, ax=ax1, title="Schwarzschild")

    # Kerr a=0.9
    ax2 = fig.add_subplot(132, projection='3d')
    metric = KerrMetric(M=1.0, a=0.9)
    finder = ApparentHorizonFinder(metric, N_s=33)
    try:
        r_plus = 1.0 + np.sqrt(1 - 0.81)
        rho = finder.find(initial_radius=r_plus, tol=1e-5, verbose=False)
        plot_horizon_3d(rho, finder.mesh, ax=ax2, color='green',
                       title="Kerr (a/M=0.9)")
    except:
        ax2.set_title("Kerr (failed)")

    # Boosted Schwarzschild
    ax3 = fig.add_subplot(133, projection='3d')
    base = SchwarzschildMetric(M=1.0)
    velocity = np.array([0.5, 0.0, 0.0])
    metric = BoostedMetric(base, velocity)
    finder = ApparentHorizonFinder(metric, N_s=33)
    try:
        rho = finder.find(initial_radius=2.0, tol=1e-5, verbose=False)
        plot_horizon_3d(rho, finder.mesh, ax=ax3, color='red',
                       title="Boosted (v=0.5c)")
    except:
        ax3.set_title("Boosted (failed)")

    plt.tight_layout()
    plt.savefig('horizons_3d_comparison.png', dpi=150)
    print("Saved: horizons_3d_comparison.png")
    plt.close()


def convergence_study_plot():
    """Create convergence study visualization."""
    print("Running convergence study...")

    metric = SchwarzschildMetric(M=1.0)
    resolutions = [17, 25, 33, 41, 49]

    errors = []
    h_values = []

    for N_s in resolutions:
        finder = ApparentHorizonFinder(metric, N_s=N_s)
        rho = finder.find(initial_radius=2.0, tol=1e-10, verbose=False)
        r_mean = finder.horizon_radius_average(rho)
        error = abs(r_mean - 2.0)
        errors.append(error)
        h_values.append(np.pi / (N_s - 1))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(h_values, errors, 'bo-', markersize=8, linewidth=2, label='Numerical')

    # Reference lines
    h_ref = np.array(h_values)
    ax.loglog(h_ref, 0.1 * h_ref**2, 'g--', label='O(h²)')
    ax.loglog(h_ref, 0.01 * h_ref**4, 'r--', label='O(h⁴)')

    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('|r - 2M|', fontsize=12)
    ax.set_title('Schwarzschild Horizon: Convergence Study', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150)
    print("Saved: convergence_study.png")
    plt.close()


def main():
    """Run all visualizations."""
    print("\n" + "=" * 60)
    print("Apparent Horizon Visualization")
    print("=" * 60 + "\n")

    visualize_schwarzschild()
    visualize_kerr()
    visualize_boosted()
    visualize_3d_comparison()
    convergence_study_plot()

    print("\nAll visualizations complete!")
    print("Generated files:")
    print("  - schwarzschild_horizon.png")
    print("  - kerr_horizons.png")
    print("  - boosted_horizons.png")
    print("  - horizons_3d_comparison.png")
    print("  - convergence_study.png")


if __name__ == "__main__":
    main()
