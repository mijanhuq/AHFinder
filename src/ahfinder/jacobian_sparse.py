"""
Sparse Jacobian computation using Lagrange interpolation.

The Jacobian J_μν = ∂F_μ/∂ρ_ν is sparse when using local Lagrange interpolation
because each residual F_μ only depends on grid points within its stencil.

Key insight:
- Each residual evaluation uses 27 Cartesian stencil points
- Each stencil point is interpolated using 16 Lagrange grid points
- Total: up to 27×16 = 432 grid point dependencies per residual
- In practice, many overlap → ~25 unique dependencies per residual

This enables O(n) Jacobian computation instead of O(n²).
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Set, List
from .surface import SurfaceMesh
from .interpolation_lagrange import LagrangeInterpolator, interpolate_batch_lagrange
from .residual import ResidualEvaluator
from .stencil import CartesianStencil
from .metrics.base import Metric


class LagrangeStencil:
    """
    Cartesian stencil that uses Lagrange interpolation and tracks dependencies.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        interpolator: LagrangeInterpolator,
        spacing_factor: float = 0.5
    ):
        """
        Initialize stencil with Lagrange interpolation.

        Args:
            mesh: SurfaceMesh instance
            interpolator: LagrangeInterpolator instance
            spacing_factor: Factor c such that h = c × d_theta
        """
        self.mesh = mesh
        self.interpolator = interpolator
        self.spacing_factor = spacing_factor
        self.h = spacing_factor * mesh.d_theta

        # Precompute offset vectors for 27-point stencil
        offsets = np.array([-1, 0, 1])
        di, dj, dk = np.meshgrid(offsets, offsets, offsets, indexing='ij')
        self._offset_vectors = np.stack([di.ravel(), dj.ravel(), dk.ravel()], axis=1)

    def compute_phi_stencil(
        self,
        rho: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Evaluate φ = r - ρ on the 27-point stencil.
        """
        h = self.h
        cx, cy, cz = center

        # Compute all 27 stencil points
        stencil_points = np.array([x0, y0, z0]) + h * self._offset_vectors

        x_pts = stencil_points[:, 0]
        y_pts = stencil_points[:, 1]
        z_pts = stencil_points[:, 2]

        # Convert to spherical coordinates
        dx = x_pts - cx
        dy = y_pts - cy
        dz = z_pts - cz

        r_pts = np.sqrt(dx**2 + dy**2 + dz**2)
        r_safe = np.maximum(r_pts, 1e-14)

        theta_pts = np.arccos(np.clip(dz / r_safe, -1, 1))
        phi_pts = np.arctan2(dy, dx)
        phi_pts = np.where(phi_pts < 0, phi_pts + 2*np.pi, phi_pts)

        # Interpolate using Lagrange
        rho_interp = self.interpolator.interpolate_batch(rho, theta_pts, phi_pts)

        phi_values = r_pts - rho_interp
        return phi_values.reshape(3, 3, 3)

    def get_stencil_dependencies(
        self,
        x0: float,
        y0: float,
        z0: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Set[Tuple[int, int]]:
        """
        Get all grid points that affect interpolation at the 27 stencil points.

        Returns:
            Set of (i_theta, i_phi) tuples
        """
        h = self.h
        cx, cy, cz = center

        # Compute all 27 stencil points
        stencil_points = np.array([x0, y0, z0]) + h * self._offset_vectors

        x_pts = stencil_points[:, 0]
        y_pts = stencil_points[:, 1]
        z_pts = stencil_points[:, 2]

        # Convert to spherical coordinates
        dx = x_pts - cx
        dy = y_pts - cy
        dz = z_pts - cz

        r_pts = np.sqrt(dx**2 + dy**2 + dz**2)
        r_safe = np.maximum(r_pts, 1e-14)

        theta_pts = np.arccos(np.clip(dz / r_safe, -1, 1))
        phi_pts = np.arctan2(dy, dx)
        phi_pts = np.where(phi_pts < 0, phi_pts + 2*np.pi, phi_pts)

        # Collect all grid dependencies
        all_deps = set()
        for theta, phi in zip(theta_pts, phi_pts):
            # Handle poles
            if theta <= 0.01:
                all_deps.add((0, 0))  # North pole
            elif theta >= np.pi - 0.01:
                all_deps.add((self.mesh.N_s - 1, 0))  # South pole
            else:
                stencil_indices = self.interpolator.get_stencil_indices(theta, phi)
                all_deps.update(stencil_indices)

        return all_deps

    def compute_derivatives(self, phi_stencil: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute first and second derivatives from stencil values."""
        h = self.h
        h2 = h * h
        phi_0 = phi_stencil[1, 1, 1]

        grad_phi = np.array([
            (phi_stencil[2, 1, 1] - phi_stencil[0, 1, 1]) / (2 * h),
            (phi_stencil[1, 2, 1] - phi_stencil[1, 0, 1]) / (2 * h),
            (phi_stencil[1, 1, 2] - phi_stencil[1, 1, 0]) / (2 * h),
        ])

        hess_phi = np.zeros((3, 3))
        hess_phi[0, 0] = (phi_stencil[2, 1, 1] - 2 * phi_0 + phi_stencil[0, 1, 1]) / h2
        hess_phi[1, 1] = (phi_stencil[1, 2, 1] - 2 * phi_0 + phi_stencil[1, 0, 1]) / h2
        hess_phi[2, 2] = (phi_stencil[1, 1, 2] - 2 * phi_0 + phi_stencil[1, 1, 0]) / h2

        hess_phi[0, 1] = (phi_stencil[2, 2, 1] - phi_stencil[2, 0, 1]
                          - phi_stencil[0, 2, 1] + phi_stencil[0, 0, 1]) / (4 * h2)
        hess_phi[1, 0] = hess_phi[0, 1]

        hess_phi[0, 2] = (phi_stencil[2, 1, 2] - phi_stencil[2, 1, 0]
                          - phi_stencil[0, 1, 2] + phi_stencil[0, 1, 0]) / (4 * h2)
        hess_phi[2, 0] = hess_phi[0, 2]

        hess_phi[1, 2] = (phi_stencil[1, 2, 2] - phi_stencil[1, 2, 0]
                          - phi_stencil[1, 0, 2] + phi_stencil[1, 0, 0]) / (4 * h2)
        hess_phi[2, 1] = hess_phi[1, 2]

        return grad_phi, hess_phi

    def compute_all_derivatives(
        self,
        rho: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute derivatives of φ at a surface point."""
        phi_stencil = self.compute_phi_stencil(rho, x0, y0, z0, center)
        return self.compute_derivatives(phi_stencil)


class SparseResidualEvaluator:
    """
    Residual evaluator using Lagrange interpolation with dependency tracking.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        stencil: LagrangeStencil,
        metric: Metric,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.mesh = mesh
        self.stencil = stencil
        self.metric = metric
        self.center = center

        # Cache dependencies for each independent point
        self._dependencies: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

        # Reverse mapping: which residuals depend on each grid point
        self._reverse_deps: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

        # Try to import fast Numba version
        try:
            from .residual_fast import compute_expansion_fast
            self._use_numba = True
            self._compute_expansion = compute_expansion_fast
        except ImportError:
            from .residual import compute_expansion
            self._use_numba = False
            self._compute_expansion = compute_expansion

    def _compute_dependencies(self, rho: np.ndarray):
        """Precompute dependency structure for current surface shape."""
        indices = self.mesh.independent_indices()
        cx, cy, cz = self.center

        self._dependencies.clear()
        self._reverse_deps.clear()

        for i_th, i_ph in indices:
            theta = self.mesh.theta[i_th]
            phi = self.mesh.phi[i_ph]
            r = rho[i_th, i_ph]

            x0 = cx + r * np.sin(theta) * np.cos(phi)
            y0 = cy + r * np.sin(theta) * np.sin(phi)
            z0 = cz + r * np.cos(theta)

            deps = self.stencil.get_stencil_dependencies(x0, y0, z0, self.center)
            self._dependencies[(i_th, i_ph)] = deps

            # Build reverse mapping
            for dep in deps:
                if dep not in self._reverse_deps:
                    self._reverse_deps[dep] = set()
                self._reverse_deps[dep].add((i_th, i_ph))

    def get_affected_residuals(
        self,
        i_theta: int,
        i_phi: int
    ) -> Set[Tuple[int, int]]:
        """
        Get residual points that would be affected by perturbing (i_theta, i_phi).
        """
        # Handle poles: all phi values at poles are the same point
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

    def evaluate_at_point(
        self,
        rho: np.ndarray,
        i_theta: int,
        i_phi: int
    ) -> float:
        """Evaluate F[ρ] at a single grid point."""
        cx, cy, cz = self.center

        theta = self.mesh.theta[i_theta]
        phi = self.mesh.phi[i_phi]
        r = rho[i_theta, i_phi]

        x0 = cx + r * np.sin(theta) * np.cos(phi)
        y0 = cy + r * np.sin(theta) * np.sin(phi)
        z0 = cz + r * np.cos(theta)

        grad_phi, hess_phi = self.stencil.compute_all_derivatives(
            rho, x0, y0, z0, self.center
        )

        if hasattr(self.metric, 'compute_all_geometric'):
            gamma_inv, dgamma, K_tensor, K_trace = self.metric.compute_all_geometric(x0, y0, z0)
        else:
            gamma_inv = self.metric.gamma_inv(x0, y0, z0)
            dgamma = self.metric.dgamma(x0, y0, z0)
            K_tensor = self.metric.extrinsic_curvature(x0, y0, z0)
            K_trace = self.metric.K_trace(x0, y0, z0)

        return self._compute_expansion(
            grad_phi, hess_phi,
            gamma_inv, dgamma,
            K_tensor, K_trace
        )

    def evaluate(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate F[ρ] at all independent grid points."""
        indices = self.mesh.independent_indices()
        residual = np.zeros(len(indices))

        for k, (i_theta, i_phi) in enumerate(indices):
            residual[k] = self.evaluate_at_point(rho, i_theta, i_phi)

        return residual


class SparseJacobianComputer:
    """
    Computes sparse Jacobian using Lagrange interpolation locality.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        residual_evaluator: SparseResidualEvaluator,
        epsilon: float = 1e-5
    ):
        self.mesh = mesh
        self.residual = residual_evaluator
        self.epsilon = epsilon

        # Index mappings
        indices = mesh.independent_indices()
        self._indices = indices
        self._grid_to_flat = {}
        for k, (i_th, i_ph) in enumerate(indices):
            self._grid_to_flat[(i_th, i_ph)] = k

    def compute_sparse(self, rho: np.ndarray) -> sparse.csr_matrix:
        """
        Compute sparse Jacobian exploiting Lagrange interpolation locality.
        """
        n = self.mesh.n_independent
        eps = self.epsilon

        # Precompute dependency structure
        self.residual._compute_dependencies(rho)

        # Compute reference residual
        F0 = self.residual.evaluate(rho)

        # Build sparse matrix
        rows = []
        cols = []
        data = []

        # Count residual evaluations for statistics
        n_evals = 0

        for nu in range(n):
            i_th, i_ph = self._indices[nu]

            # Get residuals affected by perturbing this point
            affected = self.residual.get_affected_residuals(i_th, i_ph)

            if not affected:
                continue

            # Perturb grid point
            rho_pert = rho.copy()
            rho_pert[i_th, i_ph] += eps

            # Handle pole replication
            if i_th == 0:
                rho_pert[0, :] = rho_pert[0, 0]
            elif i_th == self.mesh.N_s - 1:
                rho_pert[-1, :] = rho_pert[-1, 0]

            # Only evaluate at affected points
            for (i_th_aff, i_ph_aff) in affected:
                if (i_th_aff, i_ph_aff) not in self._grid_to_flat:
                    continue

                mu = self._grid_to_flat[(i_th_aff, i_ph_aff)]
                F_pert_mu = self.residual.evaluate_at_point(rho_pert, i_th_aff, i_ph_aff)
                n_evals += 1

                dF = (F_pert_mu - F0[mu]) / eps

                if abs(dF) > 1e-14:
                    rows.append(mu)
                    cols.append(nu)
                    data.append(dF)

        # Statistics
        density = len(data) / (n * n) * 100
        evals_saved = n * n - n_evals
        print(f"  Sparse Jacobian: {len(data)} nonzeros ({density:.1f}%), "
              f"{n_evals} evals (saved {evals_saved})")

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def compute_dense(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute dense Jacobian using sparse structure.

        Returns dense matrix for compatibility with existing solver.
        """
        J_sparse = self.compute_sparse(rho)
        return J_sparse.toarray()


def create_sparse_residual_evaluator(
    mesh: SurfaceMesh,
    metric: Metric,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_factor: float = 0.5
) -> SparseResidualEvaluator:
    """
    Create a sparse residual evaluator with Lagrange interpolation.
    """
    lagrange_interp = LagrangeInterpolator(mesh)
    stencil = LagrangeStencil(mesh, lagrange_interp, spacing_factor)
    return SparseResidualEvaluator(mesh, stencil, metric, center)
