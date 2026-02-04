"""Tests for Schwarzschild metric and horizon finding."""

import numpy as np
import pytest
from ahfinder import ApparentHorizonFinder, SurfaceMesh
from ahfinder.surface import create_sphere
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.solver import find_horizon, ConvergenceError


class TestSchwarzschildMetric:
    """Tests for SchwarzschildMetric class."""

    def test_creation(self):
        """Test metric creation."""
        metric = SchwarzschildMetric(M=1.0)
        assert metric.M == 1.0

    def test_invalid_mass(self):
        """Test that negative mass raises error."""
        with pytest.raises(ValueError):
            SchwarzschildMetric(M=-1.0)

    def test_gamma_at_infinity(self):
        """Test that metric approaches flat at large r."""
        metric = SchwarzschildMetric(M=1.0)
        gamma = metric.gamma(100.0, 0.0, 0.0)

        # Should be close to identity
        np.testing.assert_allclose(gamma, np.eye(3), atol=0.1)

    def test_gamma_positive_definite(self):
        """Test that metric is positive definite."""
        metric = SchwarzschildMetric(M=1.0)

        # Test at several points outside horizon
        test_points = [
            (3.0, 0.0, 0.0),
            (2.5, 2.5, 0.0),
            (0.0, 0.0, 4.0),
            (2.0, 2.0, 2.0),
        ]

        for x, y, z in test_points:
            gamma = metric.gamma(x, y, z)
            eigenvalues = np.linalg.eigvalsh(gamma)
            assert np.all(eigenvalues > 0), f"Metric not positive definite at ({x}, {y}, {z})"

    def test_gamma_symmetric(self):
        """Test that metric is symmetric."""
        metric = SchwarzschildMetric(M=1.0)
        gamma = metric.gamma(3.0, 2.0, 1.0)

        np.testing.assert_allclose(gamma, gamma.T)

    def test_gamma_inverse(self):
        """Test that gamma_inv is inverse of gamma."""
        metric = SchwarzschildMetric(M=1.0)

        for r in [2.5, 4.0, 10.0]:
            gamma = metric.gamma(r, 0.0, 0.0)
            gamma_inv = metric.gamma_inv(r, 0.0, 0.0)

            product = gamma @ gamma_inv
            np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_horizon_radius(self):
        """Test horizon radius is 2M."""
        metric = SchwarzschildMetric(M=1.5)
        assert metric.horizon_radius() == 3.0

    def test_lapse_at_horizon(self):
        """Test lapse at horizon."""
        metric = SchwarzschildMetric(M=1.0)

        # On horizon (r=2M), H = M/r = 1/2, so α = 1/√(1+1) = 1/√2
        alpha = metric.lapse(2.0, 0.0, 0.0)
        expected = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(alpha, expected, rtol=1e-10)

    def test_extrinsic_curvature_symmetric(self):
        """Test that extrinsic curvature is symmetric."""
        metric = SchwarzschildMetric(M=1.0)
        K = metric.extrinsic_curvature(3.0, 2.0, 1.0)

        np.testing.assert_allclose(K, K.T, atol=1e-10)


class TestSchwarzschildHorizonFinding:
    """Tests for finding the Schwarzschild horizon."""

    def test_horizon_finder_converges(self):
        """Test that horizon finder converges."""
        metric = SchwarzschildMetric(M=1.0)

        try:
            rho, mesh = find_horizon(
                metric,
                N_s=17,
                initial_radius=2.0,
                tol=1e-6,
                max_iter=30,
                verbose=False
            )
            converged = True
        except ConvergenceError:
            converged = False

        assert converged, "Newton iteration should converge for Schwarzschild"

    def test_horizon_radius_accuracy(self):
        """Test accuracy of found horizon radius."""
        metric = SchwarzschildMetric(M=1.0)

        rho, mesh = find_horizon(
            metric,
            N_s=33,
            initial_radius=2.0,
            tol=1e-8,
            max_iter=30,
            verbose=False
        )

        # Horizon should be at r = 2M = 2
        mean_radius = np.mean(rho)
        expected_radius = 2.0

        # Allow 1% error
        np.testing.assert_allclose(mean_radius, expected_radius, rtol=0.01)

    def test_horizon_is_spherical(self):
        """Test that Schwarzschild horizon is spherical."""
        metric = SchwarzschildMetric(M=1.0)

        rho, mesh = find_horizon(
            metric,
            N_s=33,
            initial_radius=2.0,
            tol=1e-8,
            max_iter=30,
            verbose=False
        )

        # All radii should be approximately equal
        std_radius = np.std(rho)
        mean_radius = np.mean(rho)

        # Standard deviation should be small relative to mean
        assert std_radius / mean_radius < 0.01

    def test_horizon_area(self):
        """Test horizon area matches analytical value."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)

        finder = ApparentHorizonFinder(metric, N_s=33)
        rho = finder.find(initial_radius=2.0, tol=1e-8, verbose=False)

        # Schwarzschild horizon area = 16πM² = 4π(2M)²
        computed_area = finder.horizon_area(rho)
        expected_area = 16 * np.pi * M**2

        # Allow 5% error due to numerical integration
        np.testing.assert_allclose(computed_area, expected_area, rtol=0.05)

    def test_different_masses(self):
        """Test horizon finding for different masses."""
        for M in [0.5, 1.0, 2.0]:
            metric = SchwarzschildMetric(M=M)

            rho, mesh = find_horizon(
                metric,
                N_s=25,
                initial_radius=2.0 * M,
                tol=1e-6,
                max_iter=30,
                verbose=False
            )

            mean_radius = np.mean(rho)
            expected_radius = 2.0 * M

            np.testing.assert_allclose(
                mean_radius, expected_radius, rtol=0.02,
                err_msg=f"Failed for M={M}"
            )


class TestSchwarzschildConvergence:
    """Tests for convergence properties."""

    @pytest.mark.slow
    def test_mesh_convergence(self):
        """Test that solution converges with mesh refinement."""
        metric = SchwarzschildMetric(M=1.0)

        resolutions = [17, 25, 33]
        mean_radii = []

        for N_s in resolutions:
            rho, mesh = find_horizon(
                metric,
                N_s=N_s,
                initial_radius=2.0,
                tol=1e-8,
                max_iter=30,
                verbose=False
            )
            mean_radii.append(np.mean(rho))

        # Higher resolution should give more accurate result
        # Check that results are converging to 2.0
        errors = [abs(r - 2.0) for r in mean_radii]

        # Error should decrease (or at least not increase significantly)
        # with increasing resolution
        assert errors[-1] <= errors[0] + 0.001

    def test_initial_guess_robustness(self):
        """Test that solver converges from different initial guesses."""
        metric = SchwarzschildMetric(M=1.0)

        # Try different initial radii
        initial_radii = [1.5, 2.0, 2.5, 3.0]

        for r0 in initial_radii:
            rho, mesh = find_horizon(
                metric,
                N_s=25,
                initial_radius=r0,
                tol=1e-6,
                max_iter=30,
                verbose=False
            )

            mean_radius = np.mean(rho)
            np.testing.assert_allclose(
                mean_radius, 2.0, rtol=0.02,
                err_msg=f"Failed to converge from initial radius {r0}"
            )
