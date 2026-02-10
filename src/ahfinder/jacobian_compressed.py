"""
Compressed Jacobian computation using graph coloring.

When the Jacobian is sparse, many columns are structurally independent
(don't share any non-zero rows). By identifying these groups via graph
coloring, we can compute multiple columns simultaneously with a single
perturbed residual evaluation.

For N_s=25: 577 columns compress to 77 groups → 7.5x fewer evaluations.
"""

import numpy as np
from scipy import sparse
from typing import List, Set, Dict, Tuple
from collections import defaultdict

from .surface import SurfaceMesh
from .jacobian_sparse import SparseResidualEvaluator


def compute_column_colors(
    mesh: SurfaceMesh,
    residual: SparseResidualEvaluator,
    rho: np.ndarray
) -> Tuple[List[int], int]:
    """
    Compute graph coloring for Jacobian columns.

    Two columns can have the same color if they don't share any rows
    (i.e., perturbing both grid points doesn't affect any common residuals).

    Returns:
        colors: List of color assignments for each column
        num_colors: Total number of colors needed
    """
    n = mesh.n_independent
    indices = mesh.independent_indices()

    # Ensure dependencies are computed
    residual._compute_dependencies(rho)

    # Build column -> rows mapping
    grid_to_flat = {(i, j): k for k, (i, j) in enumerate(indices)}

    col_to_rows = []
    for col_idx in range(n):
        i_th, i_ph = indices[col_idx]
        affected = residual.get_affected_residuals(i_th, i_ph)
        rows = set()
        for (i, j) in affected:
            if (i, j) in grid_to_flat:
                rows.add(grid_to_flat[(i, j)])
        col_to_rows.append(rows)

    # Greedy graph coloring
    colors = [-1] * n
    num_colors = 0

    for col in range(n):
        # Find colors used by conflicting columns
        used_colors = set()
        for other_col in range(col):
            if colors[other_col] >= 0 and col_to_rows[col] & col_to_rows[other_col]:
                used_colors.add(colors[other_col])

        # Assign smallest available color
        color = 0
        while color in used_colors:
            color += 1
        colors[col] = color
        num_colors = max(num_colors, color + 1)

    return colors, num_colors


def compute_compressed_jacobian(
    mesh: SurfaceMesh,
    residual: SparseResidualEvaluator,
    rho: np.ndarray,
    epsilon: float = 1e-5,
    verbose: bool = False
) -> sparse.csr_matrix:
    """
    Compute sparse Jacobian using graph coloring compression.

    Instead of one residual evaluation per column, we group columns
    by color and evaluate all columns in a group simultaneously.

    Args:
        mesh: Surface mesh
        residual: Sparse residual evaluator
        rho: Current surface values
        epsilon: Finite difference perturbation
        verbose: Print compression statistics

    Returns:
        Sparse Jacobian matrix
    """
    n = mesh.n_independent
    indices = mesh.independent_indices()

    # Compute dependencies
    residual._compute_dependencies(rho)

    # Compute graph coloring
    colors, num_colors = compute_column_colors(mesh, residual, rho)

    if verbose:
        print(f"  Compressed Jacobian: {n} columns → {num_colors} groups "
              f"({n/num_colors:.1f}x compression)")

    # Group columns by color
    color_groups: Dict[int, List[int]] = defaultdict(list)
    for col, color in enumerate(colors):
        color_groups[color].append(col)

    # Build column -> rows mapping for extracting results
    grid_to_flat = {(i, j): k for k, (i, j) in enumerate(indices)}

    col_to_rows = []
    for col_idx in range(n):
        i_th, i_ph = indices[col_idx]
        affected = residual.get_affected_residuals(i_th, i_ph)
        rows = []
        for (i, j) in affected:
            if (i, j) in grid_to_flat:
                rows.append(grid_to_flat[(i, j)])
        col_to_rows.append(rows)

    # Compute reference residual
    F0 = residual.evaluate(rho)

    # Build sparse matrix
    data = []
    row_indices = []
    col_indices = []

    # Process each color group
    for color in range(num_colors):
        cols_in_group = color_groups[color]

        # Create perturbed rho with all columns in this group perturbed
        rho_pert = rho.copy()
        for col in cols_in_group:
            i_th, i_ph = indices[col]
            rho_pert[i_th, i_ph] += epsilon

            # Handle pole replication
            if i_th == 0:
                rho_pert[0, :] = rho_pert[0, 0]
            elif i_th == mesh.N_s - 1:
                rho_pert[-1, :] = rho_pert[-1, 0]

        # Evaluate perturbed residual once for all columns in group
        F_pert = residual.evaluate(rho_pert)

        # Extract Jacobian entries for each column
        for col in cols_in_group:
            rows = col_to_rows[col]
            for row in rows:
                dF = (F_pert[row] - F0[row]) / epsilon
                if abs(dF) > 1e-14:
                    data.append(dF)
                    row_indices.append(row)
                    col_indices.append(col)

    J = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    return J


class CompressedJacobianComputer:
    """
    Computes sparse Jacobian using graph coloring compression.
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

        # Cache coloring (recompute if rho changes significantly)
        self._colors = None
        self._num_colors = None

    def compute_sparse(self, rho: np.ndarray, verbose: bool = True) -> sparse.csr_matrix:
        """Compute compressed sparse Jacobian."""
        return compute_compressed_jacobian(
            self.mesh, self.residual, rho, self.epsilon, verbose
        )

    def compute_dense(self, rho: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Compute compressed Jacobian and return as dense array."""
        return self.compute_sparse(rho, verbose).toarray()
