"""Tests for Jacobian computation.

These tests verify that the Jacobian is computed correctly, including
the critical row-sum test that catches missing couplings.
"""

import numpy as np
import pytest
from ahfinder.surface import SurfaceMesh, create_sphere
from ahfinder.interpolation import BiquarticInterpolator, FastInterpolator
from ahfinder.residual import create_residual_evaluator
from ahfinder.jacobian import JacobianComputer
from ahfinder.metrics.schwarzschild import SchwarzschildMetric


class TestJacobianBasic:
    """Basic tests for Jacobian computation."""

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.0)
        J = jacobian.compute_dense(rho)

        n = mesh.n_independent
        assert J.shape == (n, n)

    def test_jacobian_finite(self):
        """Test that Jacobian values are finite."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)
        J = jacobian.compute_dense(rho)

        assert np.all(np.isfinite(J))

    def test_jacobian_nonzero_diagonal(self):
        """Test that Jacobian diagonal entries are non-zero."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)
        J = jacobian.compute_dense(rho)

        diag = np.diag(J)
        assert np.all(np.abs(diag) > 1e-10)


class TestJacobianTaylorExpansion:
    """Tests for Jacobian accuracy via Taylor expansion."""

    def test_first_order_taylor(self):
        """Test that F(ρ + δρ) ≈ F(ρ) + J @ δρ."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)
        F0 = evaluator.evaluate(rho)
        J = jacobian.compute_dense(rho)

        # Small perturbation
        np.random.seed(42)
        delta_rho_flat = 1e-4 * np.random.randn(mesh.n_independent)
        delta_rho = mesh.flat_to_grid(delta_rho_flat)

        # Actual change
        F1 = evaluator.evaluate(rho + delta_rho)
        actual_dF = F1 - F0

        # Taylor prediction
        predicted_dF = J @ delta_rho_flat

        # Should match to O(||δρ||²)
        error = np.linalg.norm(actual_dF - predicted_dF)
        delta_norm = np.linalg.norm(delta_rho_flat)

        # Error should be O(δρ²), so error/δρ² should be bounded
        assert error < 10 * delta_norm**2

    def test_taylor_convergence_order(self):
        """Test that Taylor error converges as O(||δρ||²)."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)
        F0 = evaluator.evaluate(rho)
        J = jacobian.compute_dense(rho)

        np.random.seed(42)
        direction = np.random.randn(mesh.n_independent)
        direction /= np.linalg.norm(direction)

        errors = []
        epsilons = [1e-3, 1e-4, 1e-5]

        for eps in epsilons:
            delta_rho_flat = eps * direction
            delta_rho = mesh.flat_to_grid(delta_rho_flat)

            F1 = evaluator.evaluate(rho + delta_rho)
            actual_dF = F1 - F0
            predicted_dF = J @ delta_rho_flat

            error = np.linalg.norm(actual_dF - predicted_dF)
            errors.append(error)

        # Compute convergence order
        order1 = np.log(errors[0] / errors[1]) / np.log(epsilons[0] / epsilons[1])
        order2 = np.log(errors[1] / errors[2]) / np.log(epsilons[1] / epsilons[2])

        # Should be approximately 2 (second order)
        # Allow some tolerance due to numerical precision at small perturbations
        assert 1.2 < order1 < 2.8
        assert 1.0 < order2 < 2.8  # Can degrade at smallest epsilon


class TestJacobianRowSum:
    """Critical tests for Jacobian row sums matching dF/dr.

    This test catches the bug where sparse Jacobian computation
    missed important couplings (especially to poles).
    """

    def test_row_sum_matches_dF_dr(self):
        """Test that Jacobian row sums equal dF/dr for uniform perturbation.

        For a uniform change δρ = ε (same at all points):
            F(ρ + ε) - F(ρ) ≈ J @ (ε·ones) = ε · row_sums(J)

        Therefore: row_sums(J) ≈ dF/dr
        """
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        r = 2.5
        eps = 0.01

        rho1 = create_sphere(mesh, r)
        rho2 = create_sphere(mesh, r + eps)

        F1 = evaluator.evaluate(rho1)
        F2 = evaluator.evaluate(rho2)

        # Actual dF/dr
        dF_dr = (F2 - F1) / eps

        # Jacobian row sums
        J = jacobian.compute_dense(rho1)
        row_sums = np.sum(J, axis=1)

        # They should match
        np.testing.assert_allclose(row_sums, dF_dr, rtol=0.1)

    def test_row_sum_sign_positive_outside_horizon(self):
        """Test that row sums are positive outside the horizon.

        For Schwarzschild, Θ increases with r outside the horizon,
        so ∂F/∂ρ > 0, meaning row sums should be positive.
        """
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        # Outside horizon (r > 2M)
        rho = create_sphere(mesh, 2.5)
        J = jacobian.compute_dense(rho)
        row_sums = np.sum(J, axis=1)

        # Mean should be positive
        assert np.mean(row_sums) > 0

    def test_row_sum_multiple_radii(self):
        """Test row sum consistency at multiple radii."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        for r in [1.8, 2.0, 2.2, 2.5, 3.0]:
            eps = 0.01

            rho1 = create_sphere(mesh, r)
            rho2 = create_sphere(mesh, r + eps)

            F1 = evaluator.evaluate(rho1)
            F2 = evaluator.evaluate(rho2)
            dF_dr = (F2 - F1) / eps

            J = jacobian.compute_dense(rho1)
            row_sums = np.sum(J, axis=1)

            # Check that signs match
            assert np.sign(np.mean(row_sums)) == np.sign(np.mean(dF_dr)), \
                f"Sign mismatch at r={r}"


class TestJacobianDenseVsSparse:
    """Tests comparing dense and sparse Jacobian computation."""

    def test_dense_sparse_diagonal_match(self):
        """Test that diagonal entries match between dense and sparse."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)

        J_dense = jacobian.compute_dense(rho)
        J_sparse = jacobian.compute_sparse(rho).toarray()

        diag_dense = np.diag(J_dense)
        diag_sparse = np.diag(J_sparse)

        # Diagonals should match
        np.testing.assert_allclose(diag_dense, diag_sparse, rtol=1e-10)

    def test_dense_has_more_nonzeros(self):
        """Test that dense Jacobian has more non-zero entries than sparse.

        The sparse approximation misses some couplings.
        """
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)

        J_dense = jacobian.compute_dense(rho)
        J_sparse = jacobian.compute_sparse(rho).toarray()

        # Count non-zeros (with tolerance)
        nnz_dense = np.sum(np.abs(J_dense) > 1e-10)
        nnz_sparse = np.sum(np.abs(J_sparse) > 1e-10)

        # Dense should have at least as many (likely more)
        assert nnz_dense >= nnz_sparse


class TestNewtonConvergence:
    """Tests for Newton solver convergence with correct Jacobian."""

    def test_convergence_from_below(self):
        """Test Newton convergence starting below the horizon."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 1.5)  # Below horizon at r=2

        for _ in range(20):
            F = evaluator.evaluate(rho)
            if np.linalg.norm(F) < 1e-4:
                break

            J = jacobian.compute_dense(rho)
            delta_rho_flat = np.linalg.solve(J, -F)
            delta_rho = mesh.flat_to_grid(delta_rho_flat)
            rho = rho + delta_rho

        # Should converge to r ≈ 2
        assert abs(np.mean(rho) - 2.0) < 0.01

    def test_convergence_from_above(self):
        """Test Newton convergence starting above the horizon."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 2.5)  # Above horizon at r=2

        for _ in range(20):
            F = evaluator.evaluate(rho)
            if np.linalg.norm(F) < 1e-4:
                break

            J = jacobian.compute_dense(rho)
            delta_rho_flat = np.linalg.solve(J, -F)
            delta_rho = mesh.flat_to_grid(delta_rho_flat)
            rho = rho + delta_rho

        # Should converge to r ≈ 2
        assert abs(np.mean(rho) - 2.0) < 0.01

    def test_convergence_from_r_equals_3(self):
        """Test Newton convergence starting far from the horizon."""
        mesh = SurfaceMesh(N_s=9)
        interpolator = FastInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        jacobian = JacobianComputer(mesh, evaluator)

        rho = create_sphere(mesh, 3.0)  # Far above horizon

        for _ in range(30):
            F = evaluator.evaluate(rho)
            if np.linalg.norm(F) < 1e-4:
                break

            J = jacobian.compute_dense(rho)
            delta_rho_flat = np.linalg.solve(J, -F)
            delta_rho = mesh.flat_to_grid(delta_rho_flat)
            rho = rho + delta_rho

        # Should converge to r ≈ 2
        assert abs(np.mean(rho) - 2.0) < 0.01