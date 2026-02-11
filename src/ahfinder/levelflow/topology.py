"""
Topology detection for apparent horizon finding.

Builds a 3D expansion field Θ(r, θ, φ) and uses marching cubes
to extract Θ = 0 isosurfaces, then identifies connected components
corresponding to distinct apparent horizons.

This enables:
- Finding multiple horizons (e.g., in binary black hole mergers)
- Detecting topology changes (horizon mergers)
- Providing initial guesses for Level Flow or Newton methods

Reference: Thornburg (2007) - Living Reviews in Relativity
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings

from ..surface import SurfaceMesh
from ..metrics.base import Metric
from ..residual_vectorized import create_vectorized_residual_evaluator


@dataclass
class HorizonCandidate:
    """A candidate horizon surface detected from topology analysis."""
    rho: np.ndarray              # Surface shape ρ(θ, φ)
    center: Tuple[float, float, float]  # Center of this component
    mean_radius: float           # Mean radius
    component_id: int            # ID for tracking
    volume: float                # Approximate enclosed volume
    is_valid: bool               # Whether this is a valid horizon candidate


class TopologyDetector:
    """
    Detect apparent horizon topology using 3D expansion field analysis.

    Builds a 3D grid of expansion values Θ(r, θ, φ), then extracts
    the zero-level set using marching cubes. Connected components
    of the isosurface correspond to distinct apparent horizons.

    Args:
        mesh: SurfaceMesh instance (defines θ, φ grid)
        metric: Metric providing geometric data
        center: Center of coordinate system
        r_range: (r_min, r_max) range for radial sampling
        n_r: Number of radial grid points
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        r_range: Tuple[float, float] = (0.5, 5.0),
        n_r: int = 50
    ):
        self.mesh = mesh
        self.metric = metric
        self.center = center
        self.r_range = r_range
        self.n_r = n_r
        self.N_s = mesh.N_s

        # Radial grid
        self.r_vals = np.linspace(r_range[0], r_range[1], n_r)
        self.dr = self.r_vals[1] - self.r_vals[0]

        # Create residual evaluator
        self.residual_evaluator = create_vectorized_residual_evaluator(
            mesh, metric, center
        )

        # Index mapping
        self._indices = mesh.independent_indices()
        self._n_independent = len(self._indices)

        # Cached theta field
        self._theta_field = None

    def _residual_to_grid(self, residual_flat: np.ndarray) -> np.ndarray:
        """Convert flat residual to (N_s, N_s) grid."""
        grid = np.zeros((self.N_s, self.N_s))
        for k, (i, j) in enumerate(self._indices):
            grid[i, j] = residual_flat[k]
        grid[0, :] = grid[0, 0]
        grid[-1, :] = grid[-1, 0]
        return grid

    def build_theta_field(self, verbose: bool = True) -> np.ndarray:
        """
        Build 3D expansion field Θ(r, θ, φ).

        Evaluates the expansion scalar at every point on a 3D grid
        covering the radial range and angular mesh.

        Returns:
            Array of shape (n_r, N_s, N_s) with Θ values
        """
        theta_field = np.zeros((self.n_r, self.N_s, self.N_s))

        if verbose:
            print(f"Building Θ field: {self.n_r} × {self.N_s} × {self.N_s}")

        for ir, r in enumerate(self.r_vals):
            # Constant radius surface
            rho = np.full((self.N_s, self.N_s), r)

            # Evaluate Θ at this radius
            theta_flat = self.residual_evaluator.evaluate(rho)
            theta_field[ir] = self._residual_to_grid(theta_flat)

            if verbose and (ir + 1) % 10 == 0:
                print(f"  r = {r:.2f}: Θ range = [{theta_field[ir].min():.3f}, {theta_field[ir].max():.3f}]")

        self._theta_field = theta_field
        return theta_field

    def find_sign_changes(self) -> List[Tuple[int, int, int]]:
        """
        Find grid cells where Θ changes sign (indicating horizon crossing).

        Returns:
            List of (ir, i_theta, i_phi) tuples where sign changes occur
        """
        if self._theta_field is None:
            self.build_theta_field(verbose=False)

        sign_changes = []
        theta = self._theta_field

        # Check radial direction
        for ir in range(self.n_r - 1):
            for i_th in range(self.N_s):
                for i_ph in range(self.N_s):
                    if theta[ir, i_th, i_ph] * theta[ir + 1, i_th, i_ph] < 0:
                        sign_changes.append((ir, i_th, i_ph))

        return sign_changes

    def _find_zero_crossing_radius(
        self,
        ir: int,
        i_theta: int,
        i_phi: int
    ) -> float:
        """Find radius where Θ = 0 between grid points using linear interpolation."""
        theta = self._theta_field
        th0 = theta[ir, i_theta, i_phi]
        th1 = theta[ir + 1, i_theta, i_phi]

        # Linear interpolation
        t = th0 / (th0 - th1)
        return self.r_vals[ir] + t * self.dr

    def find_horizons_simple(self, verbose: bool = True) -> List[np.ndarray]:
        """
        Find horizons using simple sign-change detection.

        For each (θ, φ) direction, finds where Θ changes sign radially
        and interpolates to find the horizon radius.

        This is a simpler alternative to marching cubes that works
        for single, simply-connected horizons.

        Returns:
            List of surface shapes ρ(θ, φ) for each detected horizon
        """
        if self._theta_field is None:
            self.build_theta_field(verbose=verbose)

        theta = self._theta_field

        # For each (θ, φ), find all radii where sign changes
        # Group these into connected surfaces
        horizons_data = {}  # (horizon_id) -> list of (i_th, i_ph, r)

        # First pass: find all zero crossings
        crossings = np.full((self.N_s, self.N_s), np.nan)

        for i_th in range(self.N_s):
            for i_ph in range(self.N_s):
                # Find first sign change from outside (large r) to inside
                # Horizons typically have Θ < 0 inside, Θ > 0 outside
                for ir in range(self.n_r - 1, 0, -1):
                    if theta[ir, i_th, i_ph] * theta[ir - 1, i_th, i_ph] < 0:
                        # Found crossing - interpolate
                        crossings[i_th, i_ph] = self._find_zero_crossing_radius(
                            ir - 1, i_th, i_ph
                        )
                        break

        # Check if we found crossings everywhere
        valid = ~np.isnan(crossings)
        fraction_valid = np.sum(valid) / valid.size

        if verbose:
            print(f"Found crossings at {fraction_valid * 100:.1f}% of grid points")

        if fraction_valid < 0.5:
            if verbose:
                print("  Warning: Less than 50% of directions have crossings")
            return []

        # Fill in missing values with interpolation
        if fraction_valid < 1.0:
            crossings = self._fill_missing_values(crossings, valid)

        # Enforce pole conditions
        crossings[0, :] = crossings[0, 0] if not np.isnan(crossings[0, 0]) else np.nanmean(crossings[1, :])
        crossings[-1, :] = crossings[-1, 0] if not np.isnan(crossings[-1, 0]) else np.nanmean(crossings[-2, :])

        return [crossings]

    def _fill_missing_values(
        self,
        data: np.ndarray,
        valid: np.ndarray
    ) -> np.ndarray:
        """Fill missing values using neighbor averaging."""
        filled = data.copy()
        N_s = self.N_s

        # Iterative filling
        for _ in range(N_s):
            still_missing = np.isnan(filled)
            if not np.any(still_missing):
                break

            for i in range(N_s):
                for j in range(N_s):
                    if not np.isnan(filled[i, j]):
                        continue

                    # Collect valid neighbors
                    neighbors = []
                    if i > 0 and not np.isnan(filled[i - 1, j]):
                        neighbors.append(filled[i - 1, j])
                    if i < N_s - 1 and not np.isnan(filled[i + 1, j]):
                        neighbors.append(filled[i + 1, j])
                    jp = (j + 1) % N_s
                    jm = (j - 1) % N_s
                    if not np.isnan(filled[i, jp]):
                        neighbors.append(filled[i, jp])
                    if not np.isnan(filled[i, jm]):
                        neighbors.append(filled[i, jm])

                    if neighbors:
                        filled[i, j] = np.mean(neighbors)

        return filled

    def find_horizons(self, verbose: bool = True) -> List[HorizonCandidate]:
        """
        Find apparent horizons using marching cubes and connected components.

        This is the full topology detection algorithm:
        1. Build 3D Θ field
        2. Extract Θ = 0 isosurface with marching cubes
        3. Find connected components
        4. Convert each component to ρ(θ, φ) surface

        Returns:
            List of HorizonCandidate objects
        """
        try:
            from skimage.measure import marching_cubes
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            warnings.warn(
                "scikit-image not available. Using simple sign-change detection."
            )
            simple_horizons = self.find_horizons_simple(verbose=verbose)
            return [
                HorizonCandidate(
                    rho=h,
                    center=self.center,
                    mean_radius=np.mean(h),
                    component_id=i,
                    volume=4/3 * np.pi * np.mean(h)**3,
                    is_valid=True
                )
                for i, h in enumerate(simple_horizons)
            ]

        # Build Θ field
        if self._theta_field is None:
            self.build_theta_field(verbose=verbose)

        theta = self._theta_field

        # Check if there's a zero crossing at all
        if theta.min() > 0 or theta.max() < 0:
            if verbose:
                print("No Θ = 0 surface found in sampled region")
            return []

        # Extract isosurface at Θ = 0
        try:
            verts, faces, normals, values = marching_cubes(theta, level=0.0)
        except ValueError:
            if verbose:
                print("Marching cubes failed - no valid surface")
            return []

        if len(verts) == 0:
            if verbose:
                print("No vertices found in isosurface")
            return []

        if verbose:
            print(f"Marching cubes found {len(verts)} vertices, {len(faces)} faces")

        # Build adjacency matrix from faces
        n_verts = len(verts)
        row_ind = []
        col_ind = []
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                row_ind.extend([v1, v2])
                col_ind.extend([v2, v1])

        adj = csr_matrix(
            (np.ones(len(row_ind)), (row_ind, col_ind)),
            shape=(n_verts, n_verts)
        )

        # Find connected components
        n_components, labels = connected_components(adj, directed=False)

        if verbose:
            print(f"Found {n_components} connected component(s)")

        # Convert each component to ρ(θ, φ) surface
        candidates = []
        for comp_id in range(n_components):
            comp_verts = verts[labels == comp_id]

            if len(comp_verts) < 10:
                # Too few vertices - skip
                continue

            # Convert marching cubes indices to physical coordinates
            # verts are in index space: (ir, i_theta, i_phi)
            rho_surface = self._vertices_to_surface(comp_verts)

            if rho_surface is None:
                continue

            mean_r = np.mean(rho_surface)
            volume = 4/3 * np.pi * mean_r**3

            candidates.append(HorizonCandidate(
                rho=rho_surface,
                center=self.center,
                mean_radius=mean_r,
                component_id=comp_id,
                volume=volume,
                is_valid=True
            ))

        return candidates

    def _vertices_to_surface(self, verts: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert marching cubes vertices to ρ(θ, φ) surface.

        The vertices are in index space (ir, i_theta, i_phi).
        We interpolate to get radius at each (θ, φ) grid point.

        Args:
            verts: Array of (ir, i_theta, i_phi) vertex positions

        Returns:
            Surface shape ρ(θ, φ) or None if conversion fails
        """
        # Convert index coordinates to physical
        r_verts = self.r_vals[0] + verts[:, 0] * self.dr
        theta_verts = verts[:, 1] * self.mesh.d_theta
        phi_verts = verts[:, 2] * self.mesh.d_phi

        # Create surface by averaging radii in each angular bin
        rho = np.full((self.N_s, self.N_s), np.nan)
        counts = np.zeros((self.N_s, self.N_s))

        for r, th, ph in zip(r_verts, theta_verts, phi_verts):
            # Find nearest grid point
            i_th = int(th / self.mesh.d_theta + 0.5)
            i_ph = int(ph / self.mesh.d_phi + 0.5) % self.N_s

            i_th = max(0, min(self.N_s - 1, i_th))

            if np.isnan(rho[i_th, i_ph]):
                rho[i_th, i_ph] = r
            else:
                rho[i_th, i_ph] += r
            counts[i_th, i_ph] += 1

        # Average where we have multiple samples
        mask = counts > 0
        rho[mask] /= counts[mask]

        # Fill missing values
        valid = ~np.isnan(rho)
        if np.sum(valid) < self.N_s * self.N_s * 0.3:
            # Too sparse
            return None

        rho = self._fill_missing_values(rho, valid)

        # Enforce pole conditions
        rho[0, :] = np.nanmean(rho[0, :])
        rho[-1, :] = np.nanmean(rho[-1, :])

        return rho

    def visualize_theta_slice(
        self,
        phi_index: int = 0,
        ax=None,
        cmap: str = 'RdBu_r'
    ):
        """
        Visualize a φ-slice of the Θ field.

        Args:
            phi_index: Index of φ slice to show
            ax: Matplotlib axes (creates new if None)
            cmap: Colormap name
        """
        if self._theta_field is None:
            self.build_theta_field(verbose=False)

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Extract slice
        theta_slice = self._theta_field[:, :, phi_index]

        # Create coordinate arrays
        R, Theta = np.meshgrid(self.r_vals, self.mesh.theta)
        X = R * np.sin(Theta)
        Z = R * np.cos(Theta)

        # Plot
        vmax = np.max(np.abs(theta_slice))
        im = ax.pcolormesh(X, Z, theta_slice.T, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.contour(X, Z, theta_slice.T, levels=[0], colors='black', linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Θ')

        return ax


def detect_horizon_topology(
    metric: Metric,
    N_s: int = 21,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    r_range: Tuple[float, float] = (0.5, 5.0),
    n_r: int = 50,
    verbose: bool = True
) -> List[HorizonCandidate]:
    """
    Convenience function to detect apparent horizon topology.

    Args:
        metric: Metric providing geometric data
        N_s: Angular grid resolution
        center: Center of coordinate system
        r_range: Radial range to search
        n_r: Number of radial grid points
        verbose: Print progress

    Returns:
        List of HorizonCandidate objects

    Example:
        >>> from ahfinder.metrics import SchwarzschildMetric
        >>> metric = SchwarzschildMetric(M=1.0)
        >>> candidates = detect_horizon_topology(metric, r_range=(1.0, 4.0))
        >>> print(f"Found {len(candidates)} horizon(s)")
    """
    mesh = SurfaceMesh(N_s)
    detector = TopologyDetector(mesh, metric, center, r_range, n_r)
    return detector.find_horizons(verbose=verbose)
