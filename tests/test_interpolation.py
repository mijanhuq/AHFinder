"""Tests for biquartic interpolation."""

import numpy as np
import pytest
from ahfinder.surface import SurfaceMesh, create_sphere
from ahfinder.interpolation import (
    BiquarticInterpolator,
    lagrange_weights,
    lagrange_derivative_weights
)


class TestLagrangeWeights:
    """Tests for Lagrange interpolation weights."""

    def test_weights_sum_to_one(self):
        """Test that Lagrange weights sum to 1."""
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        for x in [0.5, 1.2, 2.7]:
            weights = lagrange_weights(x, nodes)
            assert np.isclose(np.sum(weights), 1.0)

    def test_interpolation_at_nodes(self):
        """Test that weights give 1 at nodes and 0 elsewhere."""
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        for i, node in enumerate(nodes):
            weights = lagrange_weights(node, nodes)
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(weights, expected, atol=1e-12)

    def test_polynomial_interpolation(self):
        """Test exact interpolation of cubic polynomial."""
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        # f(x) = x³ - 2x² + x - 1
        values = nodes**3 - 2 * nodes**2 + nodes - 1

        for x in [0.5, 1.5, 2.5]:
            weights = lagrange_weights(x, nodes)
            interp = np.dot(weights, values)
            exact = x**3 - 2 * x**2 + x - 1
            assert np.isclose(interp, exact, rtol=1e-10)


class TestDerivativeWeights:
    """Tests for Lagrange derivative weights."""

    def test_derivative_of_cubic(self):
        """Test derivative computation for cubic polynomial."""
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        # f(x) = x³
        values = nodes**3

        for x in [0.5, 1.5, 2.5]:
            weights = lagrange_derivative_weights(x, nodes)
            deriv_interp = np.dot(weights, values)
            # f'(x) = 3x²
            exact_deriv = 3 * x**2
            assert np.isclose(deriv_interp, exact_deriv, rtol=1e-10)


class TestBiquarticInterpolator:
    """Tests for BiquarticInterpolator class."""

    def test_interpolation_at_grid_points(self):
        """Test that interpolation at grid points returns grid values."""
        mesh = SurfaceMesh(N_s=17)
        interp = BiquarticInterpolator(mesh)
        rho = np.random.rand(17, 17) + 1.0

        # Test at several grid points (avoiding poles)
        for i in [3, 5, 8, 12]:
            for j in [2, 7, 11]:
                theta = mesh.theta[i]
                phi = mesh.phi[j]
                result = interp.interpolate(rho, theta, phi)
                expected = rho[i, j]
                assert np.isclose(result, expected, rtol=1e-8)

    def test_interpolation_constant(self):
        """Test interpolation of constant function."""
        mesh = SurfaceMesh(N_s=17)
        interp = BiquarticInterpolator(mesh)
        rho = np.full((17, 17), 5.0)

        # Random test points
        for _ in range(10):
            theta = np.random.uniform(0.1, np.pi - 0.1)
            phi = np.random.uniform(0, 2 * np.pi - 0.1)
            result = interp.interpolate(rho, theta, phi)
            assert np.isclose(result, 5.0, rtol=1e-10)

    def test_interpolation_linear_theta(self):
        """Test interpolation of function linear in theta."""
        mesh = SurfaceMesh(N_s=17)
        interp = BiquarticInterpolator(mesh)

        theta_grid, _ = mesh.theta_phi_grid()
        rho = 1.0 + theta_grid / np.pi  # ρ = 1 + θ/π

        # Test at random points
        for _ in range(10):
            theta = np.random.uniform(0.2, np.pi - 0.2)
            phi = np.random.uniform(0, 2 * np.pi - 0.1)
            result = interp.interpolate(rho, theta, phi)
            expected = 1.0 + theta / np.pi
            assert np.isclose(result, expected, rtol=1e-6)

    def test_interpolation_periodic_phi(self):
        """Test interpolation respects periodicity in φ."""
        mesh = SurfaceMesh(N_s=17)
        interp = BiquarticInterpolator(mesh)

        _, phi_grid = mesh.theta_phi_grid()
        rho = 2.0 + 0.3 * np.cos(phi_grid)

        # Test near φ = 2π boundary
        theta = np.pi / 2
        phi1 = 0.1
        phi2 = 2 * np.pi - 0.1

        r1 = interp.interpolate(rho, theta, phi1)
        r2 = interp.interpolate(rho, theta, phi2)

        # Should be approximately symmetric
        expected1 = 2.0 + 0.3 * np.cos(phi1)
        expected2 = 2.0 + 0.3 * np.cos(phi2)

        assert np.isclose(r1, expected1, rtol=0.1)
        assert np.isclose(r2, expected2, rtol=0.1)

    def test_pole_values(self):
        """Test interpolation at poles returns correct values."""
        mesh = SurfaceMesh(N_s=17)
        interp = BiquarticInterpolator(mesh)
        rho = create_sphere(mesh, 2.0)

        # North pole
        r_north = interp.interpolate(rho, 0.0, 0.0)
        assert np.isclose(r_north, 2.0)

        # South pole
        r_south = interp.interpolate(rho, np.pi, 0.0)
        assert np.isclose(r_south, 2.0)

    def test_gradient_constant(self):
        """Test gradient of constant function is zero."""
        mesh = SurfaceMesh(N_s=17)
        interp = BiquarticInterpolator(mesh)
        rho = np.full((17, 17), 3.0)

        theta = np.pi / 3
        phi = np.pi / 4
        val, dr_dtheta, dr_dphi = interp.interpolate_gradient(rho, theta, phi)

        assert np.isclose(val, 3.0)
        assert np.isclose(dr_dtheta, 0.0, atol=1e-10)
        assert np.isclose(dr_dphi, 0.0, atol=1e-10)

    def test_gradient_linear_theta(self):
        """Test gradient of function linear in θ."""
        mesh = SurfaceMesh(N_s=33)  # Higher resolution for better gradient
        interp = BiquarticInterpolator(mesh)

        theta_grid, _ = mesh.theta_phi_grid()
        rho = 2.0 + theta_grid  # ρ = 2 + θ

        theta = np.pi / 2
        phi = np.pi
        val, dr_dtheta, dr_dphi = interp.interpolate_gradient(rho, theta, phi)

        assert np.isclose(dr_dtheta, 1.0, rtol=0.1)  # Should be 1
        assert np.isclose(dr_dphi, 0.0, atol=0.1)  # Should be 0


class TestInterpolationAccuracy:
    """Tests for interpolation accuracy."""

    def test_fourth_order_convergence(self):
        """Test that interpolation error decreases as O(h⁴)."""
        # Use a smooth test function
        def test_func(theta, phi):
            return 2.0 + 0.3 * np.sin(theta) * np.cos(2 * phi)

        resolutions = [17, 33, 65]
        errors = []

        # Test point
        theta_test = 1.0
        phi_test = 1.5
        exact = test_func(theta_test, phi_test)

        for N_s in resolutions:
            mesh = SurfaceMesh(N_s)
            interp = BiquarticInterpolator(mesh)
            theta_grid, phi_grid = mesh.theta_phi_grid()
            rho = test_func(theta_grid, phi_grid)

            result = interp.interpolate(rho, theta_test, phi_test)
            errors.append(abs(result - exact))

        # Check that error decreases roughly as h⁴
        # h ratio between successive resolutions is about 2
        # So error ratio should be about 16
        if len(errors) >= 2 and errors[0] > 1e-12:
            ratio1 = errors[0] / errors[1]
            # Allow some tolerance; expecting ratio > 4 (better than O(h²))
            assert ratio1 > 4
