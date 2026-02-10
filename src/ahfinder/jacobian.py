"""
Numerical Jacobian computation for Newton iteration.

The Jacobian J_μν = ∂F_μ/∂ρ_ν is computed numerically using finite differences:

    J_μν = (1/ε) [F_μ[ρ + ε e_ν] - F_μ[ρ]]

where e_ν is a unit perturbation at grid point ν.

The Jacobian is sparse due to the local nature of the finite difference stencils.

Reference: Huq, Choptuik & Matzner (2000), Section II.E, Eq. 9
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Set
from .surface import SurfaceMesh
from .residual import ResidualEvaluator


class JacobianComputer:
    """
    Computes the Jacobian matrix for the Newton iteration.

    The Jacobian is computed column by column, perturbing each independent
    grid point and measuring the change in the residual.

    Due to the local stencil, each perturbation only affects a small number
    of residual values, making the Jacobian sparse.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        residual_evaluator: ResidualEvaluator,
        epsilon: float = 1e-5
    ):
        """
        Initialize Jacobian computer.

        Args:
            mesh: SurfaceMesh instance
            residual_evaluator: ResidualEvaluator for computing F[ρ]
            epsilon: Perturbation size for finite differences
        """
        self.mesh = mesh
        self.residual = residual_evaluator
        self.epsilon = epsilon

        # Precompute index mappings
        self._setup_index_maps()

    def _setup_index_maps(self):
        """
        Set up mappings between grid indices and flat indices.
        """
        mesh = self.mesh
        indices = mesh.independent_indices()

        # Map from (i_theta, i_phi) to flat index
        self._grid_to_flat = {}
        for k, (i_th, i_ph) in enumerate(indices):
            self._grid_to_flat[(i_th, i_ph)] = k

        # Store indices array
        self._indices = indices

    def _affected_points(self, i_theta: int, i_phi: int) -> Set[Tuple[int, int]]:
        """
        Determine which residual points are affected by perturbing (i_theta, i_phi).

        Due to the interpolation stencil (4×4) and the Cartesian stencil (3×3×3),
        perturbations propagate to nearby points.

        Args:
            i_theta, i_phi: Perturbed grid point

        Returns:
            Set of (i_theta, i_phi) tuples for affected points
        """
        mesh = self.mesh
        N = mesh.N_s

        # Conservative estimate: stencil radius of ~3 in each direction
        radius = 3
        affected = set()

        for di in range(-radius, radius + 1):
            i_th = i_theta + di
            if i_th < 0 or i_th >= N:
                continue

            for dj in range(-radius, radius + 1):
                i_ph = (i_phi + dj) % N

                # Skip redundant pole points
                if i_th == 0 or i_th == N - 1:
                    if i_ph != 0:
                        continue

                affected.add((i_th, i_ph))

        return affected

    def compute_dense(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute the full dense Jacobian matrix.

        Note: Despite the local Cartesian stencil, the Jacobian is essentially
        fully dense (98-100% fill) due to the spherical coordinate mapping
        and interpolation spreading the coupling globally.

        Args:
            rho: Current surface values, shape (N_s, N_s)

        Returns:
            Dense Jacobian matrix, shape (n_independent, n_independent)
        """
        n = self.mesh.n_independent
        J = np.zeros((n, n))

        # Compute reference residual
        F0 = self.residual.evaluate(rho)

        # Perturb each independent point
        for nu in range(n):
            i_th, i_ph = self._indices[nu]

            # Create perturbed rho
            rho_pert = rho.copy()
            rho_pert[i_th, i_ph] += self.epsilon

            # Handle pole replication
            if i_th == 0:
                rho_pert[0, :] = rho_pert[0, 0]
            elif i_th == self.mesh.N_s - 1:
                rho_pert[-1, :] = rho_pert[-1, 0]

            # Compute perturbed residual
            F_pert = self.residual.evaluate(rho_pert)

            # Jacobian column
            J[:, nu] = (F_pert - F0) / self.epsilon

        return J

    def compute_sparse(self, rho: np.ndarray) -> sparse.csr_matrix:
        """
        Compute the sparse Jacobian matrix.

        Only computes entries that are expected to be non-zero based on
        the stencil locality.

        Args:
            rho: Current surface values, shape (N_s, N_s)

        Returns:
            Sparse CSR Jacobian matrix
        """
        n = self.mesh.n_independent

        # Compute reference residual
        F0 = self.residual.evaluate(rho)

        # Lists for sparse matrix construction
        rows = []
        cols = []
        data = []

        # Perturb each independent point
        for nu in range(n):
            i_th, i_ph = self._indices[nu]

            # Create perturbed rho
            rho_pert = rho.copy()
            rho_pert[i_th, i_ph] += self.epsilon

            # Handle pole replication
            if i_th == 0:
                rho_pert[0, :] = rho_pert[0, 0]
            elif i_th == self.mesh.N_s - 1:
                rho_pert[-1, :] = rho_pert[-1, 0]

            # Determine affected points
            affected = self._affected_points(i_th, i_ph)

            # Compute residual only at affected points
            for (i_th_aff, i_ph_aff) in affected:
                if (i_th_aff, i_ph_aff) not in self._grid_to_flat:
                    continue

                mu = self._grid_to_flat[(i_th_aff, i_ph_aff)]
                F_pert_mu = self.residual.evaluate_at_point(rho_pert, i_th_aff, i_ph_aff)

                dF = (F_pert_mu - F0[mu]) / self.epsilon

                if abs(dF) > 1e-14:
                    rows.append(mu)
                    cols.append(nu)
                    data.append(dF)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def compute(
        self,
        rho: np.ndarray,
        sparse_output: bool = True
    ):
        """
        Compute the Jacobian matrix.

        Args:
            rho: Current surface values
            sparse_output: If True, return sparse matrix; otherwise dense

        Returns:
            Jacobian matrix (sparse or dense)
        """
        if sparse_output:
            return self.compute_sparse(rho)
        else:
            return self.compute_dense(rho)


def compute_jacobian(
    rho: np.ndarray,
    mesh: SurfaceMesh,
    residual_evaluator: ResidualEvaluator,
    epsilon: float = 1e-5,
    sparse_output: bool = True
):
    """
    Compute the Jacobian matrix for the Newton iteration.

    Args:
        rho: Current surface values
        mesh: SurfaceMesh instance
        residual_evaluator: ResidualEvaluator instance
        epsilon: Perturbation size
        sparse_output: If True, return sparse matrix

    Returns:
        Jacobian matrix
    """
    computer = JacobianComputer(mesh, residual_evaluator, epsilon)
    return computer.compute(rho, sparse_output)
