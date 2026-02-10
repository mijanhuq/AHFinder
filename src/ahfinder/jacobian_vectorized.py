"""
Vectorized sparse Jacobian computation.

Batches multiple residual evaluations together to maximize
vectorization benefits from interpolation.
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Set, List
from .surface import SurfaceMesh
from .interpolation_lagrange import LagrangeInterpolator
from .metrics.base import Metric
from .residual_vectorized import (
    compute_all_coords,
    compute_all_stencil_points,
    cartesian_to_spherical_batch,
    compute_derivatives_batch,
    compute_expansion_batch
)


class VectorizedSparseJacobianComputer:
    """
    Computes sparse Jacobian with vectorized residual evaluation.

    Key optimization: Batches all affected residuals for each column,
    performing one large interpolation call per column instead of
    many small ones.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        interpolator: LagrangeInterpolator,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing_factor: float = 0.5,
        epsilon: float = 1e-5
    ):
        self.mesh = mesh
        self.interpolator = interpolator
        self.metric = metric
        self.center = center
        self.h = spacing_factor * mesh.d_theta
        self.epsilon = epsilon

        # Index mappings
        indices = mesh.independent_indices()
        self._indices = indices
        self.n = len(indices)
        self._grid_to_flat = {(i, j): k for k, (i, j) in enumerate(indices)}

        # Precompute grid coordinates
        self._theta_grid = np.array([mesh.theta[i] for i, _ in indices])
        self._phi_grid = np.array([mesh.phi[j] for _, j in indices])

        # Dependencies cache
        self._dependencies: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        self._reverse_deps: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

    def _compute_dependencies(self, rho: np.ndarray):
        """Compute dependency structure for current surface."""
        cx, cy, cz = self.center
        h = self.h

        self._dependencies.clear()
        self._reverse_deps.clear()

        for i_th, i_ph in self._indices:
            theta = self.mesh.theta[i_th]
            phi = self.mesh.phi[i_ph]
            r = rho[i_th, i_ph]

            x0 = cx + r * np.sin(theta) * np.cos(phi)
            y0 = cy + r * np.sin(theta) * np.sin(phi)
            z0 = cz + r * np.cos(theta)

            # Get 27 stencil points
            stencil_xyz = np.array([x0, y0, z0]) + h * np.array([
                [di, dj, dk]
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                for dk in [-1, 0, 1]
            ])

            # Convert to spherical
            dx = stencil_xyz[:, 0] - cx
            dy = stencil_xyz[:, 1] - cy
            dz = stencil_xyz[:, 2] - cz
            r_sph = np.sqrt(dx**2 + dy**2 + dz**2)
            r_safe = np.maximum(r_sph, 1e-14)
            theta_sph = np.arccos(np.clip(dz / r_safe, -1, 1))
            phi_sph = np.arctan2(dy, dx)
            phi_sph = np.where(phi_sph < 0, phi_sph + 2*np.pi, phi_sph)

            # Get all grid dependencies
            deps = set()
            for th, ph in zip(theta_sph, phi_sph):
                if th <= 0.01:
                    deps.add((0, 0))
                elif th >= np.pi - 0.01:
                    deps.add((self.mesh.N_s - 1, 0))
                else:
                    stencil_idx = self.interpolator.get_stencil_indices(th, ph)
                    deps.update(stencil_idx)

            self._dependencies[(i_th, i_ph)] = deps

            for dep in deps:
                if dep not in self._reverse_deps:
                    self._reverse_deps[dep] = set()
                self._reverse_deps[dep].add((i_th, i_ph))

    def _get_affected_residuals(self, i_theta: int, i_phi: int) -> Set[Tuple[int, int]]:
        """Get residuals affected by perturbing this grid point."""
        if i_theta == 0:
            affected = set()
            for i_ph in range(self.mesh.N_s):
                if (0, i_ph) in self._reverse_deps:
                    affected.update(self._reverse_deps[(0, i_ph)])
            return affected
        elif i_theta == self.mesh.N_s - 1:
            affected = set()
            for i_ph in range(self.mesh.N_s):
                if (self.mesh.N_s - 1, i_ph) in self._reverse_deps:
                    affected.update(self._reverse_deps[(self.mesh.N_s - 1, i_ph)])
            return affected

        return self._reverse_deps.get((i_theta, i_phi), set())

    def _evaluate_residuals_batch(
        self,
        rho: np.ndarray,
        points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Evaluate residuals at multiple points in a vectorized manner.
        """
        if not points:
            return np.array([])

        n_pts = len(points)
        cx, cy, cz = self.center
        h = self.h

        # Get coordinates for all points
        theta_arr = np.array([self.mesh.theta[i] for i, _ in points])
        phi_arr = np.array([self.mesh.phi[j] for _, j in points])
        rho_arr = np.array([rho[i, j] for i, j in points])

        # Compute surface point coordinates
        x0, y0, z0 = compute_all_coords(theta_arr, phi_arr, rho_arr, self.center)

        # Compute all stencil points (n_pts × 27)
        x_stencil, y_stencil, z_stencil = compute_all_stencil_points(x0, y0, z0, h)

        # Flatten for batch processing
        x_flat = x_stencil.ravel()
        y_flat = y_stencil.ravel()
        z_flat = z_stencil.ravel()

        # Convert to spherical
        r_sph, theta_sph, phi_sph = cartesian_to_spherical_batch(
            x_flat, y_flat, z_flat, cx, cy, cz
        )

        # Single batch interpolation for ALL stencil points
        rho_interp = self.interpolator.interpolate_batch(rho, theta_sph, phi_sph)

        # Compute φ = r - ρ
        phi_flat = r_sph - rho_interp
        phi_values = phi_flat.reshape(n_pts, 27)

        # Compute derivatives
        grad_phi, hess_phi = compute_derivatives_batch(phi_values, h)

        # Get metric quantities (still point-by-point for now)
        gamma_inv = np.empty((n_pts, 3, 3))
        dgamma = np.empty((n_pts, 3, 3, 3))
        K_tensor = np.empty((n_pts, 3, 3))
        K_trace = np.empty(n_pts)

        for p in range(n_pts):
            gamma_inv[p] = self.metric.gamma_inv(x0[p], y0[p], z0[p])
            dgamma[p] = self.metric.dgamma(x0[p], y0[p], z0[p])
            K_tensor[p] = self.metric.extrinsic_curvature(x0[p], y0[p], z0[p])
            K_trace[p] = self.metric.K_trace(x0[p], y0[p], z0[p])

        # Compute expansion
        theta_expansion = compute_expansion_batch(
            grad_phi, hess_phi, gamma_inv, dgamma, K_tensor, K_trace
        )

        return theta_expansion

    def compute_sparse(self, rho: np.ndarray, verbose: bool = True) -> sparse.csr_matrix:
        """
        Compute sparse Jacobian with vectorized residual evaluation.
        """
        eps = self.epsilon
        n = self.n

        # Compute dependency structure
        self._compute_dependencies(rho)

        # Compute reference residual (vectorized)
        all_points = list(self._indices)
        F0 = self._evaluate_residuals_batch(rho, all_points)

        # Build sparse matrix
        rows = []
        cols = []
        data = []
        n_evals = 0

        for nu in range(n):
            i_th, i_ph = self._indices[nu]

            # Get affected residuals
            affected = self._get_affected_residuals(i_th, i_ph)

            if not affected:
                continue

            # Filter to independent points only
            affected_list = [(i, j) for i, j in affected if (i, j) in self._grid_to_flat]

            if not affected_list:
                continue

            # Perturb grid point
            rho_pert = rho.copy()
            rho_pert[i_th, i_ph] += eps

            # Handle pole replication
            if i_th == 0:
                rho_pert[0, :] = rho_pert[0, 0]
            elif i_th == self.mesh.N_s - 1:
                rho_pert[-1, :] = rho_pert[-1, 0]

            # Batch evaluate all affected residuals
            F_pert = self._evaluate_residuals_batch(rho_pert, affected_list)
            n_evals += len(affected_list)

            # Extract Jacobian entries
            for idx, (i_th_aff, i_ph_aff) in enumerate(affected_list):
                mu = self._grid_to_flat[(i_th_aff, i_ph_aff)]
                dF = (F_pert[idx] - F0[mu]) / eps

                if abs(dF) > 1e-14:
                    rows.append(mu)
                    cols.append(nu)
                    data.append(dF)

        if verbose:
            density = len(data) / (n * n) * 100
            print(f"  Vectorized Sparse Jacobian: {len(data)} nonzeros ({density:.1f}%), "
                  f"{n_evals} evals")

        J = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        return J


def create_vectorized_jacobian_computer(
    mesh: SurfaceMesh,
    metric: Metric,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_factor: float = 0.5,
    epsilon: float = 1e-5
) -> VectorizedSparseJacobianComputer:
    """Create a vectorized sparse Jacobian computer."""
    interpolator = LagrangeInterpolator(mesh)
    return VectorizedSparseJacobianComputer(
        mesh, interpolator, metric, center, spacing_factor, epsilon
    )
