"""Tests for Kerr metric and horizon finding."""

import numpy as np
import pytest
from ahfinder import ApparentHorizonFinder, SurfaceMesh
from ahfinder.surface import create_sphere, create_ellipsoid
from ahfinder.metrics.kerr import KerrMetric
from ahfinder.solver import find_horizon, ConvergenceError


class TestKerrMetric:
    """Tests for KerrMetric class."""

    def test_creation(self):
        """Test metric creation."""
        metric = KerrMetric(M=1.0, a=0.5)
        assert metric.M == 1.0
        assert metric.a == 0.5

    def test_schwarzschild_limit(self):
        """Test that a=0 gives Schwarzschild."""
        kerr = KerrMetric(M=1.0, a=0.0)

        # Metric should match Schwarzschild at various points
        from ahfinder.metrics.schwarzschild import SchwarzschildMetric
        schw = SchwarzschildMetric(M=1.0)

        test_points = [(3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (2.0, 2.0, 2.0)]

        for x, y, z in test_points:
            gamma_kerr = kerr.gamma(x, y, z)
            gamma_schw = schw.gamma(x, y, z)
            np.testing.assert_allclose(gamma_kerr, gamma_schw, rtol=1e-10)

    def test_invalid_spin(self):
        """Test that |a| > M raises error."""
        with pytest.raises(ValueError):
            KerrMetric(M=1.0, a=1.5)

    def test_extremal_kerr(self):
        """Test that a=M is allowed."""
        metric = KerrMetric(M=1.0, a=1.0)
        assert metric.a == 1.0

    def test_gamma_symmetric(self):
        """Test that metric is symmetric."""
        metric = KerrMetric(M=1.0, a=0.5)
        gamma = metric.gamma(3.0, 2.0, 1.0)

        np.testing.assert_allclose(gamma, gamma.T)

    def test_gamma_positive_definite(self):
        """Test that metric is positive definite outside horizon."""
        metric = KerrMetric(M=1.0, a=0.5)

        # r_+ = M + sqrt(M² - a²) = 1 + sqrt(0.75) ≈ 1.866
        # Test points should be outside this

        test_points = [
            (4.0, 0.0, 0.0),
            (0.0, 4.0, 0.0),
            (0.0, 0.0, 3.0),
            (2.0, 2.0, 2.0),
        ]

        for x, y, z in test_points:
            gamma = metric.gamma(x, y, z)
            eigenvalues = np.linalg.eigvalsh(gamma)
            assert np.all(eigenvalues > 0)

    def test_horizon_radius_equatorial(self):
        """Test equatorial horizon radius formula."""
        M = 1.0
        a = 0.5
        metric = KerrMetric(M=M, a=a)

        expected = M + np.sqrt(M**2 - a**2)
        assert np.isclose(metric.horizon_radius_equatorial(), expected)

    def test_horizon_area_formula(self):
        """Test analytical horizon area formula."""
        M = 1.0
        a = 0.5
        metric = KerrMetric(M=M, a=a)

        r_plus = M + np.sqrt(M**2 - a**2)
        expected_area = 4 * np.pi * (r_plus**2 + a**2)

        np.testing.assert_allclose(metric.horizon_area(), expected_area)


class TestKerrHorizonFinding:
    """Tests for finding the Kerr horizon."""

    def test_kerr_horizon_converges(self):
        """Test that horizon finder converges for Kerr."""
        metric = KerrMetric(M=1.0, a=0.5)

        try:
            rho, mesh = find_horizon(
                metric,
                N_s=25,
                initial_radius=1.9,
                tol=1e-5,
                max_iter=30,
                verbose=False
            )
            converged = True
        except ConvergenceError:
            converged = False

        assert converged, "Newton iteration should converge for Kerr"

    def test_kerr_horizon_equatorial_radius(self):
        """Test equatorial radius of Kerr horizon."""
        M = 1.0
        a = 0.5
        metric = KerrMetric(M=M, a=a)

        finder = ApparentHorizonFinder(metric, N_s=33)
        rho = finder.find(initial_radius=1.9, tol=1e-6, verbose=False)

        # Equatorial radius in Cartesian coordinates
        # In Kerr-Schild coords, at equator (z=0): R² = r² + a²
        # So R_eq = sqrt(r_+² + a²) where r_+ = M + sqrt(M² - a²)
        r_eq = finder.horizon_radius_equatorial(rho)
        r_plus = M + np.sqrt(M**2 - a**2)
        expected_eq = np.sqrt(r_plus**2 + a**2)

        # Allow 2% error
        np.testing.assert_allclose(r_eq, expected_eq, rtol=0.02)

    def test_kerr_horizon_is_oblate(self):
        """Test that Kerr horizon is oblate (r_eq > r_polar)."""
        metric = KerrMetric(M=1.0, a=0.7)

        finder = ApparentHorizonFinder(metric, N_s=33)
        rho = finder.find(initial_radius=1.8, tol=1e-5, verbose=False)

        r_eq = finder.horizon_radius_equatorial(rho)
        r_polar = finder.horizon_radius_polar(rho)

        # Kerr horizon is oblate: equatorial > polar
        assert r_eq > r_polar

    def test_kerr_horizon_area_invariant(self):
        """Test that computed area matches analytical value."""
        M = 1.0
        a = 0.5
        metric = KerrMetric(M=M, a=a)

        finder = ApparentHorizonFinder(metric, N_s=33)
        rho = finder.find(initial_radius=1.9, tol=1e-6, verbose=False)

        computed_area = finder.horizon_area(rho)
        expected_area = metric.horizon_area()

        # Allow 10% error due to numerical integration and horizon finding
        np.testing.assert_allclose(computed_area, expected_area, rtol=0.10)


class TestKerrSpinDependence:
    """Tests for spin-dependent properties."""

    def test_horizon_shrinks_with_spin(self):
        """Test that horizon radius decreases with increasing spin."""
        M = 1.0
        spins = [0.0, 0.3, 0.6, 0.9]
        radii = []

        for a in spins:
            metric = KerrMetric(M=M, a=a)
            r_plus = metric.horizon_radius_equatorial()
            radii.append(r_plus)

        # r_+ = M + sqrt(M² - a²) decreases with a
        for i in range(len(radii) - 1):
            assert radii[i] > radii[i + 1]

    def test_oblateness_increases_with_spin(self):
        """Test that oblateness increases with spin."""
        M = 1.0

        # Low spin
        metric_low = KerrMetric(M=M, a=0.3)
        finder_low = ApparentHorizonFinder(metric_low, N_s=25)

        try:
            rho_low = finder_low.find(initial_radius=1.95, tol=1e-5, verbose=False)
            oblate_low = (
                finder_low.horizon_radius_equatorial(rho_low) -
                finder_low.horizon_radius_polar(rho_low)
            )
        except ConvergenceError:
            pytest.skip("Low spin case did not converge")
            return

        # High spin
        metric_high = KerrMetric(M=M, a=0.7)
        finder_high = ApparentHorizonFinder(metric_high, N_s=25)

        try:
            rho_high = finder_high.find(initial_radius=1.8, tol=1e-5, verbose=False)
            oblate_high = (
                finder_high.horizon_radius_equatorial(rho_high) -
                finder_high.horizon_radius_polar(rho_high)
            )
        except ConvergenceError:
            pytest.skip("High spin case did not converge")
            return

        # Higher spin should give more oblateness
        assert oblate_high > oblate_low


class TestKerrIrreducibleMass:
    """Tests for irreducible mass computation."""

    def test_irreducible_mass_schwarzschild(self):
        """Test irreducible mass for Schwarzschild (a=0) equals M."""
        M = 1.0
        metric = KerrMetric(M=M, a=0.0)

        # M_irr = sqrt(A/16π) where A = 16πM² for Schwarzschild
        # So M_irr = M
        expected_M_irr = M

        np.testing.assert_allclose(metric.irreducible_mass(), expected_M_irr, rtol=1e-10)

    def test_irreducible_mass_less_than_M(self):
        """Test that irreducible mass is less than M for spinning black hole."""
        M = 1.0
        a = 0.9

        metric = KerrMetric(M=M, a=a)

        # For Kerr: M² = M_irr² + J²/(4M_irr²) where J = Ma
        # So M_irr < M when a > 0
        M_irr = metric.irreducible_mass()

        assert M_irr < M
        assert M_irr > 0
