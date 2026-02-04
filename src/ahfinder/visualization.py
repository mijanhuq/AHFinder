"""
Visualization utilities for apparent horizons.

Provides functions for 3D surface plots, cross-sections, and
convergence plots.
"""

import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .surface import SurfaceMesh
from .finder import ApparentHorizonFinder


def plot_horizon_3d(
    rho: np.ndarray,
    mesh: SurfaceMesh,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    alpha: float = 0.7,
    title: str = "Apparent Horizon"
) -> plt.Axes:
    """
    Create a 3D surface plot of the apparent horizon.

    Args:
        rho: Surface values ρ(θ, φ), shape (N_s, N_s)
        mesh: SurfaceMesh instance
        center: Center of coordinate system
        ax: Matplotlib 3D axes (created if None)
        color: Surface color
        alpha: Surface transparency
        title: Plot title

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    x, y, z = mesh.xyz_from_rho(rho, center)

    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

    # Set equal aspect ratio
    max_range = np.max([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]) / 2.0

    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)

    return ax


def plot_horizon_cross_section(
    rho: np.ndarray,
    mesh: SurfaceMesh,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    plane: str = 'xz',
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    label: Optional[str] = None,
    linestyle: str = '-',
    linewidth: float = 2.0
) -> plt.Axes:
    """
    Plot a 2D cross-section of the horizon through a coordinate plane.

    Args:
        rho: Surface values ρ(θ, φ), shape (N_s, N_s)
        mesh: SurfaceMesh instance
        center: Center of coordinate system
        plane: Which plane to slice ('xy', 'xz', or 'yz')
        ax: Matplotlib axes (created if None)
        color: Line color
        label: Label for legend
        linestyle: Line style
        linewidth: Line width

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    x, y, z = mesh.xyz_from_rho(rho, center)

    if plane == 'xz':
        # φ = 0 slice (i_phi = 0) and φ = π slice (i_phi = N_s // 2)
        i_phi_0 = 0
        i_phi_pi = mesh.N_s // 2

        x_pos = x[:, i_phi_0]
        z_pos = z[:, i_phi_0]
        x_neg = x[:, i_phi_pi]
        z_neg = z[:, i_phi_pi]

        # Combine for full cross-section
        x_cross = np.concatenate([x_pos, x_neg[::-1]])
        z_cross = np.concatenate([z_pos, z_neg[::-1]])

        ax.plot(x_cross, z_cross, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

    elif plane == 'xy':
        # θ = π/2 slice (equatorial)
        i_theta_eq = mesh.N_s // 2

        x_eq = x[i_theta_eq, :]
        y_eq = y[i_theta_eq, :]

        # Close the curve
        x_closed = np.append(x_eq, x_eq[0])
        y_closed = np.append(y_eq, y_eq[0])

        ax.plot(x_closed, y_closed, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    elif plane == 'yz':
        # φ = π/2 slice (i_phi = N_s // 4) and φ = 3π/2 slice
        i_phi_half = mesh.N_s // 4
        i_phi_three_half = 3 * mesh.N_s // 4

        y_pos = y[:, i_phi_half]
        z_pos = z[:, i_phi_half]
        y_neg = y[:, i_phi_three_half]
        z_neg = z[:, i_phi_three_half]

        y_cross = np.concatenate([y_pos, y_neg[::-1]])
        z_cross = np.concatenate([z_pos, z_neg[::-1]])

        ax.plot(y_cross, z_cross, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label)
        ax.set_xlabel('y')
        ax.set_ylabel('z')

    else:
        raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'.")

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    return ax


def plot_convergence(
    residual_history: List[float],
    delta_rho_history: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Newton Convergence"
) -> plt.Axes:
    """
    Plot the convergence history of Newton iteration.

    Args:
        residual_history: List of ||F|| values per iteration
        delta_rho_history: List of ||δρ|| values per iteration
        ax: Matplotlib axes (created if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    iterations = range(len(residual_history))

    ax.semilogy(iterations, residual_history, 'b-o', label='||F||', markersize=6)

    if len(delta_rho_history) > 0:
        # δρ history is one shorter
        ax.semilogy(range(len(delta_rho_history)), delta_rho_history,
                    'r-s', label='||δρ||', markersize=6)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Norm')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def compare_horizons(
    rho_list: List[np.ndarray],
    mesh: SurfaceMesh,
    labels: List[str],
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    plane: str = 'xz',
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    title: str = "Horizon Comparison"
) -> plt.Axes:
    """
    Compare multiple horizons in a cross-section plot.

    Args:
        rho_list: List of surface values
        mesh: SurfaceMesh instance (assumed same for all)
        labels: Labels for each horizon
        center: Center of coordinate system
        plane: Which plane to slice
        ax: Matplotlib axes (created if None)
        colors: Colors for each horizon
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(rho_list)))

    for rho, label, color in zip(rho_list, labels, colors):
        plot_horizon_cross_section(
            rho, mesh, center, plane, ax,
            color=color, label=label
        )

    ax.set_title(title)
    ax.legend()

    return ax


def plot_horizon_radius_profile(
    rho: np.ndarray,
    mesh: SurfaceMesh,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: str = 'blue'
) -> plt.Axes:
    """
    Plot the radial profile ρ(θ) averaged over φ.

    Args:
        rho: Surface values, shape (N_s, N_s)
        mesh: SurfaceMesh instance
        ax: Matplotlib axes (created if None)
        label: Label for legend
        color: Line color

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Average over φ
    rho_avg = np.mean(rho, axis=1)
    theta_deg = np.degrees(mesh.theta)

    ax.plot(theta_deg, rho_avg, color=color, label=label, linewidth=2)
    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('ρ')
    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    return ax


def plot_residual_map(
    residual: np.ndarray,
    mesh: SurfaceMesh,
    ax: Optional[plt.Axes] = None,
    title: str = "Residual Distribution"
) -> plt.Axes:
    """
    Plot the residual F[ρ] as a function of (θ, φ).

    Args:
        residual: Flat array of residual values
        mesh: SurfaceMesh instance
        ax: Matplotlib axes (created if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Convert flat residual to grid
    residual_grid = mesh.flat_to_grid(residual)

    theta_deg = np.degrees(mesh.theta)
    phi_deg = np.degrees(mesh.phi)

    im = ax.pcolormesh(phi_deg, theta_deg, residual_grid,
                       shading='auto', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='F[ρ]')

    ax.set_xlabel('φ (degrees)')
    ax.set_ylabel('θ (degrees)')
    ax.set_title(title)

    return ax


def create_horizon_animation(
    rho_sequence: List[np.ndarray],
    mesh: SurfaceMesh,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    interval: int = 200,
    title: str = "Horizon Evolution"
):
    """
    Create an animation of horizon evolution (e.g., during Newton iteration).

    Args:
        rho_sequence: List of surface values at each step
        mesh: SurfaceMesh instance
        center: Center of coordinate system
        interval: Milliseconds between frames
        title: Animation title

    Returns:
        Matplotlib animation object
    """
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Initialize plots
    x0, y0, z0 = mesh.xyz_from_rho(rho_sequence[0], center)

    # 3D view
    ax1.remove()
    ax1 = fig.add_subplot(121, projection='3d')
    surf = [ax1.plot_surface(x0, y0, z0, color='blue', alpha=0.7)]

    # Cross-section
    line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)

    def update(frame):
        rho = rho_sequence[frame]
        x, y, z = mesh.xyz_from_rho(rho, center)

        # Update 3D
        ax1.clear()
        ax1.plot_surface(x, y, z, color='blue', alpha=0.7)

        max_r = np.max(np.abs(rho)) * 1.2
        ax1.set_xlim(-max_r, max_r)
        ax1.set_ylim(-max_r, max_r)
        ax1.set_zlim(-max_r, max_r)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        # Update cross-section
        x_cross = np.concatenate([x[:, 0], x[::-1, mesh.N_s // 2]])
        z_cross = np.concatenate([z[:, 0], z[::-1, mesh.N_s // 2]])
        line.set_data(x_cross, z_cross)

        ax2.set_title(f'Frame {frame + 1}/{len(rho_sequence)}')

        return surf

    anim = FuncAnimation(fig, update, frames=len(rho_sequence),
                         interval=interval, blit=False)

    return anim
