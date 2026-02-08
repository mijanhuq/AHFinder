"""Tests for boosted metrics."""

import numpy as np
import pytest
import time
from ahfinder import ApparentHorizonFinder
from ahfinder.metrics.schwarzschild import SchwarzschildMetric
from ahfinder.metrics.kerr import KerrMetric
from ahfinder.metrics.boosted import BoostedMetric, boost_metric
from ahfinder.metrics.boosted_fast import FastBoostedMetric, CachedBoostedMetric, fast_boost_metric
from ahfinder.solver import find_horizon, ConvergenceError


class TestBoostedMetric:
    """Tests for BoostedMetric class."""

    def test_creation(self):
        """Test boosted metric creation."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.5, 0.0, 0.0])

        boosted = BoostedMetric(base, velocity)

        assert boosted.v_mag == 0.5
        np.testing.assert_allclose(boosted.n_hat, [1.0, 0.0, 0.0])

    def test_zero_velocity(self):
        """Test that zero velocity gives original metric."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.0, 0.0, 0.0])

        boosted = BoostedMetric(base, velocity)

        # Should match original at several points
        test_points = [(3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (2.0, 2.0, 2.0)]

        for x, y, z in test_points:
            gamma_base = base.gamma(x, y, z)
            gamma_boosted = boosted.gamma(x, y, z)
            np.testing.assert_allclose(gamma_boosted, gamma_base, rtol=1e-6)

    def test_invalid_velocity(self):
        """Test that superluminal velocity raises error."""
        base = SchwarzschildMetric(M=1.0)

        with pytest.raises(ValueError):
            BoostedMetric(base, np.array([0.8, 0.8, 0.0]))  # |v| > 1

    def test_lorentz_factor(self):
        """Test Lorentz factor computation."""
        base = SchwarzschildMetric(M=1.0)
        v = 0.6
        velocity = np.array([v, 0.0, 0.0])

        boosted = BoostedMetric(base, velocity)

        expected_gamma = 1.0 / np.sqrt(1 - v**2)
        assert np.isclose(boosted.lorentz_gamma, expected_gamma)

    def test_gamma_symmetric(self):
        """Test that boosted metric is symmetric."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.5, 0.0, 0.0])
        boosted = BoostedMetric(base, velocity)

        gamma = boosted.gamma(3.0, 2.0, 1.0)
        np.testing.assert_allclose(gamma, gamma.T, atol=1e-10)

    def test_gamma_positive_definite(self):
        """Test that boosted metric is positive definite."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.5, 0.0, 0.0])
        boosted = BoostedMetric(base, velocity)

        # Test at point far from origin
        gamma = boosted.gamma(5.0, 3.0, 2.0)
        eigenvalues = np.linalg.eigvalsh(gamma)
        assert np.all(eigenvalues > 0)


class TestBoostedSchwarzschildHorizon:
    """Tests for boosted Schwarzschild horizon finding."""

    def test_boosted_horizon_converges(self):
        """Test that horizon finder converges for boosted Schwarzschild."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])
        boosted = BoostedMetric(base, velocity)

        try:
            rho, mesh = find_horizon(
                boosted,
                N_s=25,
                initial_radius=2.0,
                tol=1e-5,
                max_iter=30,
                verbose=False
            )
            converged = True
        except ConvergenceError:
            converged = False

        assert converged, "Newton iteration should converge for boosted Schwarzschild"

    def test_horizon_lorentz_contracted(self):
        """Test that boosted horizon is Lorentz contracted."""
        base = SchwarzschildMetric(M=1.0)
        v = 0.5
        velocity = np.array([v, 0.0, 0.0])
        boosted = BoostedMetric(base, velocity)

        finder = ApparentHorizonFinder(boosted, N_s=33)

        try:
            rho = finder.find(initial_radius=2.0, tol=1e-5, verbose=False)
        except ConvergenceError:
            pytest.skip("Boosted horizon did not converge")
            return

        # Get coordinates
        x, y, z = finder.horizon_coordinates(rho)

        # Extent in x direction (boost direction) should be contracted
        x_extent = x.max() - x.min()

        # Extent in y direction (perpendicular) should be approximately unchanged
        y_extent = y.max() - y.min()

        # Expected contraction factor is 1/gamma
        gamma = 1.0 / np.sqrt(1 - v**2)

        # x_extent should be less than y_extent
        # For Schwarzschild, unboosted diameter is 4M = 4
        # Boosted: x_extent ~ 4/gamma, y_extent ~ 4
        expected_ratio = 1.0 / gamma

        actual_ratio = x_extent / y_extent

        # Allow 20% tolerance due to numerical effects
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.2)


class TestBoostedAreaInvariance:
    """Tests for area invariance under boosts."""

    def test_area_invariant_low_velocity(self):
        """Test that horizon area is approximately invariant under low-velocity boost."""
        M = 1.0
        base = SchwarzschildMetric(M=M)

        # Unboosted horizon
        finder_unboosted = ApparentHorizonFinder(base, N_s=33)
        try:
            rho_unboosted = finder_unboosted.find(initial_radius=2.0, tol=1e-6, verbose=False)
            area_unboosted = finder_unboosted.horizon_area(rho_unboosted)
        except ConvergenceError:
            pytest.skip("Unboosted horizon did not converge")
            return

        # Boosted horizon
        velocity = np.array([0.3, 0.0, 0.0])
        boosted = BoostedMetric(base, velocity)
        finder_boosted = ApparentHorizonFinder(boosted, N_s=33)

        try:
            rho_boosted = finder_boosted.find(initial_radius=2.0, tol=1e-5, verbose=False)
            area_boosted = finder_boosted.horizon_area(rho_boosted)
        except ConvergenceError:
            pytest.skip("Boosted horizon did not converge")
            return

        # Areas should be approximately equal (within 15%)
        np.testing.assert_allclose(area_boosted, area_unboosted, rtol=0.15)


class TestBoostFunction:
    """Tests for the boost_metric convenience function."""

    def test_boost_metric_function(self):
        """Test that boost_metric function works correctly."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.4, 0.0, 0.0])

        boosted = boost_metric(base, velocity)

        assert isinstance(boosted, BoostedMetric)
        assert np.isclose(boosted.v_mag, 0.4)

    def test_boost_kerr(self):
        """Test boosting a Kerr metric."""
        base = KerrMetric(M=1.0, a=0.5)
        velocity = np.array([0.3, 0.0, 0.0])

        boosted = boost_metric(base, velocity)

        # Should create valid boosted metric
        gamma = boosted.gamma(5.0, 0.0, 0.0)
        assert gamma.shape == (3, 3)
        np.testing.assert_allclose(gamma, gamma.T)


class TestBoostDirections:
    """Tests for boosts in different directions."""

    def test_boost_y_direction(self):
        """Test boost in y direction."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.0, 0.4, 0.0])
        boosted = BoostedMetric(base, velocity)

        np.testing.assert_allclose(boosted.n_hat, [0.0, 1.0, 0.0])

    def test_boost_z_direction(self):
        """Test boost in z direction."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.0, 0.0, 0.5])
        boosted = BoostedMetric(base, velocity)

        np.testing.assert_allclose(boosted.n_hat, [0.0, 0.0, 1.0])

    def test_boost_diagonal_direction(self):
        """Test boost in diagonal direction."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.3, 0.0])
        boosted = BoostedMetric(base, velocity)

        expected_v_mag = np.sqrt(0.3**2 + 0.3**2)
        assert np.isclose(boosted.v_mag, expected_v_mag)

        expected_n = np.array([0.3, 0.3, 0.0]) / expected_v_mag
        np.testing.assert_allclose(boosted.n_hat, expected_n)


class TestFastBoostedMetric:
    """Tests for the fast (analytical) boosted metric implementation."""

    def test_fast_matches_numerical_gamma(self):
        """Test that fast gamma matches numerical implementation."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        numerical = BoostedMetric(base, velocity)
        fast = FastBoostedMetric(base, velocity)

        x, y, z = 2.5, 0.5, 0.3
        gamma_num = numerical.gamma(x, y, z)
        gamma_fast = fast.gamma(x, y, z)

        np.testing.assert_allclose(gamma_fast, gamma_num, rtol=1e-8)

    def test_fast_matches_numerical_dgamma(self):
        """Test that fast dgamma matches numerical implementation."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        numerical = BoostedMetric(base, velocity)
        fast = FastBoostedMetric(base, velocity)

        x, y, z = 2.5, 0.5, 0.3
        dgamma_num = numerical.dgamma(x, y, z)
        dgamma_fast = fast.dgamma(x, y, z)

        np.testing.assert_allclose(dgamma_fast, dgamma_num, rtol=1e-4)

    def test_fast_matches_numerical_K(self):
        """Test that fast extrinsic curvature matches numerical implementation."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        numerical = BoostedMetric(base, velocity)
        fast = FastBoostedMetric(base, velocity)

        x, y, z = 2.5, 0.5, 0.3
        K_num = numerical.extrinsic_curvature(x, y, z)
        K_fast = fast.extrinsic_curvature(x, y, z)

        np.testing.assert_allclose(K_fast, K_num, rtol=1e-4)

    def test_fast_matches_numerical_lapse_shift(self):
        """Test that fast lapse and shift match numerical implementation."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        numerical = BoostedMetric(base, velocity)
        fast = FastBoostedMetric(base, velocity)

        x, y, z = 2.5, 0.5, 0.3

        assert np.isclose(fast.lapse(x, y, z), numerical.lapse(x, y, z), rtol=1e-8)
        np.testing.assert_allclose(fast.shift(x, y, z), numerical.shift(x, y, z), rtol=1e-8)

    def test_cached_metric(self):
        """Test that cached metric returns correct values."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        cached = CachedBoostedMetric(base, velocity)
        fast = FastBoostedMetric(base, velocity)

        x, y, z = 2.5, 0.5, 0.3

        # Test that cached returns same values as non-cached
        np.testing.assert_allclose(cached.gamma(x, y, z), fast.gamma(x, y, z))
        np.testing.assert_allclose(cached.dgamma(x, y, z), fast.dgamma(x, y, z))
        np.testing.assert_allclose(cached.extrinsic_curvature(x, y, z),
                                   fast.extrinsic_curvature(x, y, z))

    def test_fast_boost_metric_function(self):
        """Test the fast_boost_metric convenience function."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.4, 0.0, 0.0])

        boosted = fast_boost_metric(base, velocity, use_cache=True)
        assert isinstance(boosted, CachedBoostedMetric)

        boosted_no_cache = fast_boost_metric(base, velocity, use_cache=False)
        assert isinstance(boosted_no_cache, FastBoostedMetric)


class TestFastBoostedHorizon:
    """Tests for horizon finding with fast boosted metric."""

    def test_fast_horizon_converges(self):
        """Test that horizon finder converges with fast boosted metric."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])
        boosted = CachedBoostedMetric(base, velocity)

        finder = ApparentHorizonFinder(boosted, N_s=13)

        try:
            rho = finder.find(initial_radius=2.0, tol=1e-5, verbose=False)
            converged = True
        except ConvergenceError:
            converged = False

        assert converged, "Newton iteration should converge with fast boosted metric"

    def test_fast_area_matches_numerical(self):
        """Test that fast metric gives same horizon area as numerical."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        numerical = BoostedMetric(base, velocity)
        fast = CachedBoostedMetric(base, velocity)

        # Find horizons with both
        finder_num = ApparentHorizonFinder(numerical, N_s=9)
        finder_fast = ApparentHorizonFinder(fast, N_s=9)

        try:
            rho_num = finder_num.find(initial_radius=2.0, tol=1e-5, verbose=False)
            area_num = finder_num.horizon_area(rho_num)

            rho_fast = finder_fast.find(initial_radius=2.0, tol=1e-5, verbose=False)
            area_fast = finder_fast.horizon_area(rho_fast)

            # Areas should match closely
            np.testing.assert_allclose(area_fast, area_num, rtol=0.01)
        except ConvergenceError:
            pytest.skip("One of the metrics did not converge")

    def test_fast_area_invariance(self):
        """Test that fast boosted metric preserves area invariance."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])
        boosted = CachedBoostedMetric(base, velocity)

        # Unboosted
        finder1 = ApparentHorizonFinder(base, N_s=13)
        rho1 = finder1.find(initial_radius=2.0, tol=1e-6, verbose=False)
        area1 = finder1.horizon_area(rho1)

        # Boosted
        finder2 = ApparentHorizonFinder(boosted, N_s=13)
        rho2 = finder2.find(initial_radius=2.0, tol=1e-5, verbose=False)
        area2 = finder2.horizon_area(rho2)

        # Areas should be approximately equal (within 5%)
        ratio = area2 / area1
        assert 0.95 < ratio < 1.05, f"Area ratio {ratio} should be close to 1.0"

    def test_fast_is_faster(self):
        """Test that fast metric is significantly faster than numerical."""
        base = SchwarzschildMetric(M=1.0)
        velocity = np.array([0.3, 0.0, 0.0])

        numerical = BoostedMetric(base, velocity)
        fast = CachedBoostedMetric(base, velocity)

        # Time extrinsic curvature calls
        n_calls = 50
        points = [(2.0 + i*0.05, 0.3, 0.2) for i in range(n_calls)]

        t0 = time.time()
        for p in points:
            numerical.extrinsic_curvature(*p)
        t_num = time.time() - t0

        t0 = time.time()
        for p in points:
            fast.extrinsic_curvature(*p)
        t_fast = time.time() - t0

        speedup = t_num / t_fast
        assert speedup > 2.0, f"Fast metric should be at least 2x faster, got {speedup:.1f}x"
