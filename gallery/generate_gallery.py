"""
Generate a gallery of 3D horizon plots for boosted Kerr black holes.

Varies:
- Boost velocity v = 0, 0.3, 0.6, 0.9
- Spin parameter a = 0, 0.25, 0.5, 0.75, 0.99
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ahfinder import ApparentHorizonFinder
from ahfinder.metrics import SchwarzschildMetricFast, KerrMetric
from ahfinder.metrics.boosted_kerr_fast import FastBoostedKerrMetric


def find_horizon(a, velocity, N_s=25, max_iter=30, tol=1e-5, initial_rho=None):
    """Find horizon for given spin and velocity vector.

    Args:
        a: Spin parameter
        velocity: 3-vector velocity or scalar (scalar means x-direction boost)
        N_s: Mesh resolution
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance
        initial_rho: Initial guess (for continuation)

    If initial_rho is provided, it will be interpolated to the new mesh resolution
    and used as the starting guess (continuation approach).
    """
    from scipy.interpolate import RectBivariateSpline

    # Convert scalar velocity to x-direction vector
    if np.isscalar(velocity):
        velocity = np.array([velocity, 0.0, 0.0])
    else:
        velocity = np.asarray(velocity)

    v_mag = np.linalg.norm(velocity)

    # Create metric using FastBoostedKerrMetric which handles both
    # Schwarzschild (a=0) and Kerr (a>0) cases correctly with boosts
    metric = FastBoostedKerrMetric(M=1.0, a=a, velocity=velocity)

    # Find horizon
    finder = ApparentHorizonFinder(metric, N_s=N_s)

    # Use continuation from previous solution if available
    if initial_rho is not None:
        # Interpolate initial_rho to new mesh resolution if needed
        old_shape = initial_rho.shape
        new_shape = (N_s, N_s)  # Mesh uses N_s for both theta and phi

        if old_shape != new_shape:
            # Create coordinate grids for interpolation
            # Note: phi uses endpoint=False to match SurfaceMesh
            old_theta = np.linspace(0, np.pi, old_shape[0])
            old_phi = np.linspace(0, 2*np.pi, old_shape[1], endpoint=False)
            new_theta = np.linspace(0, np.pi, new_shape[0])
            new_phi = np.linspace(0, 2*np.pi, new_shape[1], endpoint=False)

            # Interpolate
            interp = RectBivariateSpline(old_theta, old_phi, initial_rho)
            rho_init = interp(new_theta, new_phi)
        else:
            rho_init = initial_rho.copy()

        rho = finder.find(initial_guess=rho_init, tol=tol, max_iter=max_iter, verbose=False)
    else:
        # Initial guess based on expected Lorentz contraction
        if a == 0:
            r_base = 2.0
        else:
            r_plus = 1.0 + np.sqrt(1.0 - a**2)
            r_base = r_plus

        # At high velocity, the horizon is Lorentz contracted
        gamma = 1.0 / np.sqrt(1.0 - v_mag**2) if v_mag < 1 else 10
        # Use average of contracted and uncontracted
        r_init = r_base * (1.0 + 1.0/gamma) / 2 * 1.05

        rho = finder.find(initial_radius=r_init, tol=tol, max_iter=max_iter, verbose=False)

    return rho, finder.mesh, finder


def plot_horizon_3d(rho, mesh, center=(0, 0, 0), title="", filename=None,
                    elev=20, azim=45, figsize=(8, 8),
                    xlim=None, ylim=None, zlim=None):
    """Create a 3D surface plot of the horizon with fixed axis limits."""
    # Get coordinates
    x, y, z = mesh.xyz_from_rho(rho, center)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.9,
                    linewidth=0.1, edgecolor='k', antialiased=True)

    # Set fixed axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title, fontsize=14)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")

    plt.close(fig)
    return fig


def main():
    gallery_dir = os.path.dirname(os.path.abspath(__file__))

    # Parameter ranges
    spins = [0.0, 0.25, 0.5, 0.75, 0.99]

    # Build list of cases: (spin, velocity_vector, v_mag, boost_type, filename_suffix)
    cases = []

    # X-direction boosts (standard cases)
    for v_mag in [0.0, 0.3, 0.6]:
        for a in spins:
            velocity = np.array([v_mag, 0.0, 0.0])
            v_str = f"{v_mag:.1f}".replace(".", "p")
            cases.append((a, velocity, v_mag, 'x', f"v{v_str}"))

    # Diagonal boosts (xy-plane, 45 degrees)
    for v_mag in [0.3, 0.6]:
        for a in [0.0, 0.5]:  # Just a few spin values for diagonal
            # Velocity along (1,1,0)/sqrt(2) direction
            velocity = np.array([v_mag/np.sqrt(2), v_mag/np.sqrt(2), 0.0])
            v_str = f"{v_mag:.1f}".replace(".", "p")
            cases.append((a, velocity, v_mag, 'diag_xy', f"v{v_str}_diag"))

    print("=" * 60)
    print("Generating Boosted Kerr Horizon Gallery")
    print("=" * 60)
    print(f"Spins: {spins}")
    print(f"Total cases: {len(cases)}")
    print("=" * 60)
    sys.stdout.flush()

    # First pass: find all horizons and determine global axis limits
    print("\nPass 1: Finding all horizons...")
    results = []
    all_x, all_y, all_z = [], [], []

    for a, velocity, v_mag, boost_type, suffix in cases:
        if boost_type == 'x':
            print(f"  Processing: a={a}, v={v_mag}", end="", flush=True)
        else:
            print(f"  Processing: a={a}, v={v_mag} ({boost_type})", end="", flush=True)

        # Skip extremely difficult cases (near-extremal spin with high boost)
        if a >= 0.99 and v_mag >= 0.6:
            print(" - SKIPPED (extreme case)")
            results.append({
                'a': a, 'velocity': velocity, 'v_mag': v_mag,
                'boost_type': boost_type, 'suffix': suffix,
                'success': False, 'error': 'Skipped: extreme case'
            })
            continue

        # Adjust parameters based on difficulty
        if v_mag >= 0.6 and a >= 0.75:
            N_s = 25
            max_iter = 50
            tol = 1e-4
        elif v_mag >= 0.6:
            N_s = 25
            max_iter = 40
            tol = 1e-4
        else:
            N_s = 25
            max_iter = 30
            tol = 1e-5

        try:
            rho, mesh, finder = find_horizon(a, velocity, N_s=N_s, max_iter=max_iter, tol=tol)
            x, y, z = mesh.xyz_from_rho(rho, (0, 0, 0))

            # Track global extents
            all_x.extend([x.min(), x.max()])
            all_y.extend([y.min(), y.max()])
            all_z.extend([z.min(), z.max()])

            # Store result
            r_mean = np.mean(rho)
            r_eq = finder.horizon_radius_equatorial(rho)
            r_pol = finder.horizon_radius_polar(rho)
            area = finder.horizon_area(rho)

            results.append({
                'a': a, 'velocity': velocity, 'v_mag': v_mag,
                'boost_type': boost_type, 'suffix': suffix,
                'rho': rho, 'mesh': mesh,
                'r_mean': r_mean, 'r_eq': r_eq, 'r_pol': r_pol,
                'area': area, 'success': True
            })
            print(f" - OK (r_eq={r_eq:.3f}, r_pol={r_pol:.3f})")

        except Exception as e:
            print(f" - FAILED: {e}")
            results.append({
                'a': a, 'velocity': velocity, 'v_mag': v_mag,
                'boost_type': boost_type, 'suffix': suffix,
                'success': False, 'error': str(e)
            })

        sys.stdout.flush()

    # Compute global axis limits (symmetric, same for all axes)
    if all_x:
        max_extent = max(
            max(abs(min(all_x)), abs(max(all_x))),
            max(abs(min(all_y)), abs(max(all_y))),
            max(abs(min(all_z)), abs(max(all_z)))
        ) * 1.1  # 10% margin
    else:
        max_extent = 2.5

    xlim = (-max_extent, max_extent)
    ylim = (-max_extent, max_extent)
    zlim = (-max_extent, max_extent)

    print(f"\nGlobal axis limits: [{-max_extent:.2f}, {max_extent:.2f}]")

    # Second pass: create plots with consistent axis limits
    print("\nPass 2: Creating plots...")
    for result in results:
        if not result['success']:
            continue

        a = result['a']
        v_mag = result['v_mag']
        boost_type = result['boost_type']
        suffix = result['suffix']
        rho, mesh = result['rho'], result['mesh']

        # Create title
        if a == 0:
            metric_name = "Schwarzschild"
        else:
            metric_name = f"Kerr (a={a})"

        if v_mag > 0:
            if boost_type == 'diag_xy':
                title = f"{metric_name}, v={v_mag}c (diagonal)"
            else:
                title = f"{metric_name}, v={v_mag}c"
        else:
            title = f"{metric_name}"

        # Create filename
        a_str = f"{a:.2f}".replace(".", "p")
        filename = os.path.join(gallery_dir, f"horizon_a{a_str}_{suffix}.png")

        # Plot with fixed axis limits
        plot_horizon_3d(rho, mesh, title=title, filename=filename,
                       xlim=xlim, ylim=ylim, zlim=zlim)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'a':>6} {'|v|':>6} {'type':>8} {'r_mean':>10} {'r_eq':>10} {'r_pol':>10} {'area':>10}")
    print("-" * 70)

    for r in results:
        boost_label = r['boost_type'] if r['boost_type'] != 'x' else 'x-dir'
        if r['success']:
            print(f"{r['a']:>6.2f} {r['v_mag']:>6.2f} {boost_label:>8} {r['r_mean']:>10.4f} {r['r_eq']:>10.4f} {r['r_pol']:>10.4f} {r['area']:>10.4f}")
        else:
            print(f"{r['a']:>6.2f} {r['v_mag']:>6.2f} {boost_label:>8} FAILED")

    print("=" * 70)

    # Count successes
    n_success = sum(1 for r in results if r['success'])
    print(f"\nSuccessfully generated: {n_success}/{len(results)} horizons")
    print(f"Images saved to: {gallery_dir}")


if __name__ == "__main__":
    main()
