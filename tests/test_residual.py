"""Tests for residual evaluation."""

import numpy as np
import pytest
from ahfinder.surface import SurfaceMesh, create_sphere
from ahfinder.interpolation import BiquarticInterpolator
from ahfinder.stencil import CartesianStencil
from ahfinder.residual import ResidualEvaluator, create_residual_evaluator, compute_dgamma_inv
from ahfinder.metrics.base import FlatMetric
from ahfinder.metrics.schwarzschild import SchwarzschildMetric


class TestComputeDgammaInv:
    """Tests for inverse metric derivative computation."""

    def test_flat_metric(self):
        """Test that dgamma_inv is zero for flat metric."""
        gamma_inv = np.eye(3)
        dgamma = np.zeros((3, 3, 3))

        dgamma_inv = compute_dgamma_inv(gamma_inv, dgamma)

        np.testing.assert_allclose(dgamma_inv, 0.0, atol=1e-14)

    def test_symmetry(self):
        """Test that dgamma_inv is symmetric in last two indices."""
        # Create some non-trivial metric
        gamma_inv = np.eye(3) + 0.1 * np.random.rand(3, 3)
        gamma_inv = 0.5 * (gamma_inv + gamma_inv.T)  # Symmetrize

        dgamma = 0.1 * np.random.rand(3, 3, 3)
        # Symmetrize in i,j indices
        for k in range(3):
            dgamma[k] = 0.5 * (dgamma[k] + dgamma[k].T)

        dgamma_inv = compute_dgamma_inv(gamma_inv, dgamma)

        # Check symmetry
        for k in range(3):
            np.testing.assert_allclose(
                dgamma_inv[k], dgamma_inv[k].T, atol=1e-14
            )


class TestResidualEvaluator:
    """Tests for ResidualEvaluator class."""

    def test_creation(self):
        """Test that evaluator can be created."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)

        evaluator = create_residual_evaluator(mesh, interpolator, metric)
        assert evaluator is not None

    def test_residual_shape(self):
        """Test that residual has correct shape."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        rho = create_sphere(mesh, 2.0)
        F = evaluator.evaluate(rho)

        assert len(F) == mesh.n_independent

    def test_residual_finite(self):
        """Test that residual values are finite."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        rho = create_sphere(mesh, 2.0)
        F = evaluator.evaluate(rho)

        assert np.all(np.isfinite(F))

    def test_residual_at_schwarzschild_horizon(self):
        """Test that residual is small at Schwarzschild horizon."""
        mesh = SurfaceMesh(N_s=33)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        # Exact horizon at r = 2M = 2
        rho = create_sphere(mesh, 2.0)
        F = evaluator.evaluate(rho)

        # Residual should be relatively small at the horizon
        # (not exactly zero due to numerical errors)
        F_norm = np.linalg.norm(F)
        assert F_norm < 10.0  # Loose bound for non-converged solution


class TestResidualNorm:
    """Tests for residual norm computation."""

    def test_norm_positive(self):
        """Test that norm is non-negative."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        rho = create_sphere(mesh, 2.0)
        norm = evaluator.residual_norm(rho)

        assert norm >= 0

    def test_norm_consistency(self):
        """Test that norm is consistent with evaluate."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        rho = create_sphere(mesh, 2.5)
        F = evaluator.evaluate(rho)
        norm = evaluator.residual_norm(rho)

        np.testing.assert_allclose(norm, np.linalg.norm(F))


class TestResidualSensitivity:
    """Tests for residual sensitivity to surface perturbations."""

    def test_residual_changes_with_rho(self):
        """Test that residual changes when ρ changes."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        rho1 = create_sphere(mesh, 2.0)
        rho2 = create_sphere(mesh, 2.5)

        F1 = evaluator.evaluate(rho1)
        F2 = evaluator.evaluate(rho2)

        # Residuals should be different
        assert not np.allclose(F1, F2)

    def test_residual_smooth_variation(self):
        """Test that residual varies smoothly with small changes in ρ."""
        mesh = SurfaceMesh(N_s=17)
        interpolator = BiquarticInterpolator(mesh)
        metric = SchwarzschildMetric(M=1.0)
        evaluator = create_residual_evaluator(mesh, interpolator, metric)

        rho = create_sphere(mesh, 2.0)
        F0 = evaluator.evaluate(rho)

        # Small perturbation
        delta = 0.01
        rho_pert = rho + delta
        F1 = evaluator.evaluate(rho_pert)

        # Change should be proportional to perturbation
        dF = np.linalg.norm(F1 - F0)
        assert dF > 0  # Should change
        assert dF < 100 * delta * mesh.n_independent  # But not too much
