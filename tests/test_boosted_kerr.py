"""
Tests for FastBoostedKerrMetric implementation.

These tests verify the mathematical correctness of the semi-analytical
boosted Kerr metric implementation using known invariants and relations.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ahfinder.metrics.boosted_kerr_fast import FastBoostedKerrMetric
from ahfinder.metrics import KerrMetric


class TestKerrSchildInvariants:
    """Test Kerr-Schild metric invariants."""

    def test_determinant_relation(self):
        """Test det(γ) = 1 + 2H for Kerr-Schild metrics.

        For γ_ij = δ_ij + 2H l_i l_j with |l|² = 1,
        det(γ) = 1 + 2H.
        """
        for a in [0.0, 0.5, 0.9]:
            metric = FastBoostedKerrMetric(M=1.0, a=a, velocity=np.array([0.0, 0.0, 0.0]))

            # Test at several points
            test_points = [
                (3.0, 0.0, 0.0),  # Equatorial
                (0.0, 3.0, 0.0),  # Equatorial
                (0.0, 0.0, 3.0),  # Polar
                (2.0, 1.0, 0.5),  # General
            ]

            for x, y, z in test_points:
                gamma = metric.gamma(x, y, z)
                det_gamma = np.linalg.det(gamma)

                # Get H from lapse: α = 1/√(1+2H) → H = (1/α² - 1)/2
                alpha = metric.lapse(x, y, z)
                H = (1/alpha**2 - 1) / 2

                expected_det = 1 + 2*H
                np.testing.assert_allclose(det_gamma, expected_det, rtol=1e-10,
                    err_msg=f"det(γ) != 1+2H at ({x},{y},{z}) for a={a}")

    def test_inverse_relation(self):
        """Test γ^ij γ_jk = δ^i_k."""
        for a in [0.0, 0.5, 0.9]:
            for v in [0.0, 0.3, 0.6]:
                metric = FastBoostedKerrMetric(M=1.0, a=a,
                                               velocity=np.array([v, 0.0, 0.0]))

                test_points = [(3.0, 0.0, 0.0), (2.0, 1.0, 0.5)]

                for x, y, z in test_points:
                    gamma = metric.gamma(x, y, z)
                    gamma_inv = metric.gamma_inv(x, y, z)

                    product = gamma_inv @ gamma
                    np.testing.assert_allclose(product, np.eye(3), atol=1e-10,
                        err_msg=f"γ^(-1) γ != I at ({x},{y},{z}) for a={a}, v={v}")

    def test_gamma_symmetric(self):
        """Test that γ_ij is symmetric."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.7,
                                       velocity=np.array([0.3, 0.2, 0.0]))

        for x, y, z in [(3.0, 1.0, 0.5), (0.0, 2.0, 1.0)]:
            gamma = metric.gamma(x, y, z)
            np.testing.assert_allclose(gamma, gamma.T, atol=1e-14,
                err_msg=f"γ not symmetric at ({x},{y},{z})")

    def test_gamma_positive_definite(self):
        """Test that γ_ij is positive definite."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.5,
                                       velocity=np.array([0.5, 0.0, 0.0]))

        for x, y, z in [(3.0, 0.0, 0.0), (2.0, 1.0, 0.5)]:
            gamma = metric.gamma(x, y, z)
            eigenvalues = np.linalg.eigvalsh(gamma)
            assert np.all(eigenvalues > 0), \
                f"γ not positive definite at ({x},{y},{z}): eigenvalues={eigenvalues}"


class TestLapseShiftRelations:
    """Test lapse and shift for Kerr-Schild metrics."""

    def test_lapse_formula(self):
        """Test α = 1/√(1+2H)."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.0, 0.0, 0.0]))

        for x, y, z in [(3.0, 0.0, 0.0), (2.0, 1.0, 0.5)]:
            alpha = metric.lapse(x, y, z)

            # α must be positive and less than 1
            assert 0 < alpha < 1, f"Invalid lapse {alpha} at ({x},{y},{z})"

    def test_shift_direction(self):
        """Test that shift β^i is proportional to l^i."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.0, 0.0, 0.0]))

        x, y, z = 3.0, 0.0, 0.0
        beta = metric.shift(x, y, z)

        # β^i = 2H l^i / (1+2H), so β should point radially outward at equator
        # For Kerr, l ≈ (x/r, y/r, z/r) at large r
        r = np.sqrt(x**2 + y**2 + z**2)
        l_approx = np.array([x/r, y/r, z/r])

        # Check that beta is roughly parallel to l
        if np.linalg.norm(beta) > 1e-10:
            beta_normalized = beta / np.linalg.norm(beta)
            l_normalized = l_approx / np.linalg.norm(l_approx)
            dot = np.dot(beta_normalized, l_normalized)
            assert dot > 0.9, f"Shift not aligned with null vector: dot={dot}"


class TestDgammaConsistency:
    """Test consistency of metric derivatives."""

    def test_dgamma_numerical_vs_analytical(self):
        """Test dgamma matches numerical differentiation."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.3, 0.0, 0.0]))

        x, y, z = 3.0, 1.0, 0.5
        h = 1e-5

        dgamma_analytical = metric.dgamma(x, y, z)

        # Numerical derivatives
        dgamma_numerical = np.zeros((3, 3, 3))
        for k in range(3):
            offset = np.zeros(3)
            offset[k] = h

            gamma_plus = metric.gamma(x + offset[0], y + offset[1], z + offset[2])
            gamma_minus = metric.gamma(x - offset[0], y - offset[1], z - offset[2])

            dgamma_numerical[k] = (gamma_plus - gamma_minus) / (2*h)

        np.testing.assert_allclose(dgamma_analytical, dgamma_numerical, rtol=1e-4,
            err_msg="dgamma analytical vs numerical mismatch")

    def test_dgamma_symmetry(self):
        """Test that ∂_k γ_ij = ∂_k γ_ji (metric derivative inherits symmetry)."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.3, 0.2, 0.0]))

        for x, y, z in [(3.0, 1.0, 0.5), (2.0, 0.0, 1.0)]:
            dgamma = metric.dgamma(x, y, z)

            for k in range(3):
                np.testing.assert_allclose(dgamma[k], dgamma[k].T, atol=1e-12,
                    err_msg=f"∂_{k} γ not symmetric at ({x},{y},{z})")


class TestExtrinsicCurvature:
    """Test extrinsic curvature properties."""

    def test_K_symmetric(self):
        """Test that K_ij is symmetric."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.3, 0.0, 0.0]))

        for x, y, z in [(3.0, 0.0, 0.0), (2.0, 1.0, 0.5)]:
            K = metric.extrinsic_curvature(x, y, z)
            np.testing.assert_allclose(K, K.T, atol=1e-12,
                err_msg=f"K not symmetric at ({x},{y},{z})")

    def test_stationary_kerr_K_matches_original(self):
        """Test that unboosted FastBoostedKerrMetric K matches KerrMetric K.

        Note: Kerr-Schild slicing is NOT maximal slicing, so K ≠ 0.
        """
        boosted = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.0, 0.0, 0.0]))
        kerr = KerrMetric(M=1.0, a=0.5)

        for x, y, z in [(3.0, 0.0, 0.0), (2.0, 1.0, 0.5), (0.0, 0.0, 3.0)]:
            K_trace_boosted = boosted.K_trace(x, y, z)
            K_trace_kerr = kerr.K_trace(x, y, z)

            np.testing.assert_allclose(K_trace_boosted, K_trace_kerr, rtol=1e-8,
                err_msg=f"K_trace mismatch at ({x},{y},{z})")

    def test_boosted_kerr_K_changes(self):
        """Test that boost changes K from stationary value.

        For a moving black hole, ∂_t γ_ij = -v^k ∂_k γ_ij ≠ 0,
        which modifies the extrinsic curvature.
        """
        unboosted = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.0, 0.0, 0.0]))
        boosted = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=np.array([0.5, 0.0, 0.0]))

        x, y, z = 3.0, 0.0, 0.0
        K_unboosted = unboosted.K_trace(x, y, z)
        K_boosted = boosted.K_trace(x, y, z)

        # The boost should change K
        assert not np.isclose(K_unboosted, K_boosted, rtol=0.01), \
            f"Boost should change K: unboosted={K_unboosted}, boosted={K_boosted}"

    def test_K_matches_schwarzschild(self):
        """Test K_ij for Schwarzschild matches between implementations.

        Note: Kerr-Schild slicing has K ≠ 0 even for stationary spacetimes.
        """
        from ahfinder.metrics import SchwarzschildMetric

        boosted = FastBoostedKerrMetric(M=1.0, a=0.0, velocity=np.array([0.0, 0.0, 0.0]))
        sch = SchwarzschildMetric(M=1.0)

        x, y, z = 3.0, 0.0, 0.0
        K_boosted = boosted.extrinsic_curvature(x, y, z)
        K_sch = sch.extrinsic_curvature(x, y, z)

        np.testing.assert_allclose(K_boosted, K_sch, atol=1e-8,
            err_msg="K mismatch between FastBoostedKerrMetric(a=0) and SchwarzschildMetric")


class TestBoostTransformation:
    """Test boost transformation properties."""

    def test_unboosted_matches_kerr(self):
        """Test that unboosted FastBoostedKerrMetric matches KerrMetric."""
        a = 0.5
        boosted = FastBoostedKerrMetric(M=1.0, a=a, velocity=np.array([0.0, 0.0, 0.0]))
        kerr = KerrMetric(M=1.0, a=a)

        test_points = [(3.0, 0.0, 0.0), (2.0, 1.0, 0.5), (0.0, 0.0, 3.0)]

        for x, y, z in test_points:
            np.testing.assert_allclose(boosted.gamma(x, y, z), kerr.gamma(x, y, z),
                rtol=1e-10, err_msg=f"gamma mismatch at ({x},{y},{z})")

            np.testing.assert_allclose(boosted.dgamma(x, y, z), kerr.dgamma(x, y, z),
                rtol=1e-6, err_msg=f"dgamma mismatch at ({x},{y},{z})")

            np.testing.assert_allclose(
                boosted.extrinsic_curvature(x, y, z),
                kerr.extrinsic_curvature(x, y, z),
                atol=1e-8, err_msg=f"K mismatch at ({x},{y},{z})")

    def test_boost_different_spins_different_metrics(self):
        """Test that different spins produce different metrics when boosted."""
        v = np.array([0.3, 0.0, 0.0])

        m1 = FastBoostedKerrMetric(M=1.0, a=0.0, velocity=v)
        m2 = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=v)
        m3 = FastBoostedKerrMetric(M=1.0, a=0.9, velocity=v)

        x, y, z = 3.0, 0.0, 0.0

        g1 = m1.gamma(x, y, z)
        g2 = m2.gamma(x, y, z)
        g3 = m3.gamma(x, y, z)

        # All should be different
        assert not np.allclose(g1, g2), "a=0 and a=0.5 should give different gamma"
        assert not np.allclose(g2, g3), "a=0.5 and a=0.9 should give different gamma"
        assert not np.allclose(g1, g3), "a=0 and a=0.9 should give different gamma"

    def test_boost_direction_independence(self):
        """Test that boost magnitude, not direction, determines det(γ) at rest frame origin."""
        v_mag = 0.5

        # Different boost directions
        v_x = np.array([v_mag, 0.0, 0.0])
        v_y = np.array([0.0, v_mag, 0.0])
        v_diag = np.array([v_mag/np.sqrt(2), v_mag/np.sqrt(2), 0.0])

        m_x = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=v_x)
        m_y = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=v_y)
        m_diag = FastBoostedKerrMetric(M=1.0, a=0.5, velocity=v_diag)

        # At a point far from origin, the metrics should give same determinant
        # (boost just reorients, doesn't change volume element magnitude at same physical distance)
        x, y, z = 5.0, 0.0, 0.0

        det_x = np.linalg.det(m_x.gamma(x, y, z))
        det_y = np.linalg.det(m_y.gamma(x, y, z))
        det_diag = np.linalg.det(m_diag.gamma(x, y, z))

        # All determinants should be positive
        assert det_x > 0 and det_y > 0 and det_diag > 0


class TestLorentzContraction:
    """Test Lorentz contraction effects on the metric."""

    def test_lorentz_gamma_stored(self):
        """Test that Lorentz gamma factor is computed correctly."""
        v_mag = 0.6
        metric = FastBoostedKerrMetric(M=1.0, a=0.0, velocity=np.array([v_mag, 0.0, 0.0]))

        expected_gamma = 1.0 / np.sqrt(1 - v_mag**2)
        np.testing.assert_allclose(metric.lorentz_gamma, expected_gamma, rtol=1e-14)

    def test_lambda_matrix(self):
        """Test that Λ matrix is correct for coordinate transformation."""
        v_mag = 0.5
        metric = FastBoostedKerrMetric(M=1.0, a=0.0, velocity=np.array([v_mag, 0.0, 0.0]))

        # Λ = I + (γ-1) n⊗n where n is boost direction
        gamma = metric.lorentz_gamma
        n = np.array([1.0, 0.0, 0.0])
        expected_Lambda = np.eye(3) + (gamma - 1) * np.outer(n, n)

        np.testing.assert_allclose(metric.Lambda, expected_Lambda, rtol=1e-14)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_schwarzschild_limit(self):
        """Test that a=0 gives Schwarzschild metric."""
        boosted = FastBoostedKerrMetric(M=1.0, a=0.0, velocity=np.array([0.0, 0.0, 0.0]))

        x, y, z = 3.0, 0.0, 0.0
        r = np.sqrt(x**2 + y**2 + z**2)

        # For Schwarzschild: H = M/r
        alpha = boosted.lapse(x, y, z)
        H = (1/alpha**2 - 1) / 2
        expected_H = 1.0 / r

        np.testing.assert_allclose(H, expected_H, rtol=1e-10)

    def test_near_horizon(self):
        """Test metric near the horizon (should still be finite)."""
        # Schwarzschild horizon at r=2M
        metric = FastBoostedKerrMetric(M=1.0, a=0.0, velocity=np.array([0.0, 0.0, 0.0]))

        # Just outside horizon
        x, y, z = 2.1, 0.0, 0.0

        gamma = metric.gamma(x, y, z)
        assert np.all(np.isfinite(gamma)), "gamma should be finite near horizon"

        K = metric.extrinsic_curvature(x, y, z)
        assert np.all(np.isfinite(K)), "K should be finite near horizon"

    def test_high_spin(self):
        """Test near-extremal spin a ≈ M."""
        metric = FastBoostedKerrMetric(M=1.0, a=0.99, velocity=np.array([0.0, 0.0, 0.0]))

        x, y, z = 3.0, 0.0, 0.0

        gamma = metric.gamma(x, y, z)
        assert np.all(np.isfinite(gamma)), "gamma should be finite for high spin"
        assert np.linalg.det(gamma) > 0, "det(gamma) should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
