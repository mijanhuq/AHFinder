"""
Visualize Jacobian sparsity: Spline vs Lagrange interpolation.

Creates side-by-side spy plots showing the non-zero structure of the Jacobian
matrix for both interpolation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from ahfinder.surface import SurfaceMesh, create_sphere
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.interpolation import FastInterpolator
from ahfinder.residual import create_residual_evaluator
from ahfinder.jacobian import JacobianComputer
from ahfinder.jacobian_sparse import (
    create_sparse_residual_evaluator,
    SparseJacobianComputer
)


def compute_jacobians(N_s=17):
    """Compute both dense and sparse Jacobians."""
    mesh = SurfaceMesh(N_s)
    metric = SchwarzschildMetric(M=1.0)
    center = (0.0, 0.0, 0.0)
    rho = create_sphere(mesh, 2.0)

    # Dense Jacobian (spline interpolation)
    print(f"Computing dense Jacobian (N_s={N_s})...")
    interpolator = FastInterpolator(mesh, spline_order=3)
    dense_residual = create_residual_evaluator(mesh, interpolator, metric, center)
    dense_jacobian = JacobianComputer(mesh, dense_residual)
    J_dense = dense_jacobian.compute_dense(rho)

    # Sparse Jacobian (Lagrange interpolation)
    print("Computing sparse Jacobian...")
    sparse_residual = create_sparse_residual_evaluator(mesh, metric, center)
    sparse_jacobian = SparseJacobianComputer(mesh, sparse_residual)
    J_sparse = sparse_jacobian.compute_sparse(rho)

    return J_dense, J_sparse, mesh.n_independent


def plot_sparsity(J_dense, J_sparse, n_independent, threshold=1e-10, save_path=None):
    """Create side-by-side spy plots of Jacobian sparsity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count non-zeros
    nnz_dense = np.sum(np.abs(J_dense) > threshold)
    nnz_sparse = J_sparse.nnz
    total = n_independent * n_independent

    density_dense = 100 * nnz_dense / total
    density_sparse = 100 * nnz_sparse / total

    # Dense Jacobian (spline)
    ax1 = axes[0]
    ax1.spy(np.abs(J_dense) > threshold, markersize=0.5, color='navy')
    ax1.set_title(f'Spline Interpolation\n{nnz_dense:,} nonzeros ({density_dense:.1f}%)',
                  fontsize=12)
    ax1.set_xlabel('Column (perturbed grid point)', fontsize=10)
    ax1.set_ylabel('Row (residual point)', fontsize=10)

    # Sparse Jacobian (Lagrange)
    ax2 = axes[1]
    ax2.spy(J_sparse, markersize=0.5, color='darkred')
    ax2.set_title(f'Lagrange Interpolation\n{nnz_sparse:,} nonzeros ({density_sparse:.1f}%)',
                  fontsize=12)
    ax2.set_xlabel('Column (perturbed grid point)', fontsize=10)
    ax2.set_ylabel('Row (residual point)', fontsize=10)

    # Overall title
    fig.suptitle(f'Jacobian Sparsity Pattern (N_s=17, {n_independent}×{n_independent} matrix)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")

    plt.show()


def main():
    N_s = 17  # Use smaller grid for clearer visualization
    J_dense, J_sparse, n_indep = compute_jacobians(N_s)

    print(f"\nMatrix size: {n_indep} × {n_indep} = {n_indep**2:,} entries")
    print(f"Dense nonzeros: {np.sum(np.abs(J_dense) > 1e-10):,}")
    print(f"Sparse nonzeros: {J_sparse.nnz:,}")

    plot_sparsity(J_dense, J_sparse, n_indep,
                  save_path='jacobian_sparsity_comparison.png')


if __name__ == "__main__":
    main()
