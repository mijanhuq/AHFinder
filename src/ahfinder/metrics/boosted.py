"""
Lorentz-boosted metrics.

Transforms any Kerr-Schild metric under a Lorentz boost. The Kerr-Schild
form g_μν = η_μν + 2H l_μ l_ν is preserved under boosts.

Reference: Huq, Choptuik & Matzner (2000), Section III.C
"""

import numpy as np
from .base import Metric


class BoostedMetric(Metric):
    """
    Lorentz-boosted version of any Kerr-Schild metric.

    Under a boost with velocity v in direction n̂, the Kerr-Schild
    quantities transform as:
        H' = γ²(1 - v·l)² H
        l'_μ = (l_μ + terms involving boost)

    where γ = 1/√(1 - v²) is the Lorentz factor.

    The 3+1 quantities in the boosted frame are computed from
    the transformed H and l.

    Attributes:
        base_metric: The underlying (unboosted) metric
        velocity: 3-vector velocity of the boost
        lorentz_gamma: Lorentz factor γ = 1/√(1-v²)
    """

    def __init__(self, base_metric: Metric, velocity: np.ndarray):
        """
        Initialize boosted metric.

        Args:
            base_metric: The metric to boost (must support Kerr-Schild form)
            velocity: 3-vector boost velocity (|v| < 1)
        """
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,):
            raise ValueError("velocity must be a 3-vector")

        v_mag = np.linalg.norm(velocity)
        if v_mag >= 1.0:
            raise ValueError("Boost velocity must have magnitude < 1")

        self.base_metric = base_metric
        self.velocity = velocity.copy()
        self.v_mag = v_mag

        if v_mag > 1e-14:
            self.lorentz_gamma = 1.0 / np.sqrt(1 - v_mag**2)
            self.n_hat = velocity / v_mag
        else:
            self.lorentz_gamma = 1.0
            self.n_hat = np.array([1.0, 0.0, 0.0])

    def _transform_coordinates(self, x: float, y: float, z: float):
        """
        Transform from boosted frame coordinates to rest frame coordinates.

        For a boost with velocity v, the inverse transformation (from
        boosted lab frame to rest frame of black hole) involves:
            t' = γ(t - v·x)
            x' = x + (γ-1)(x·n̂)n̂ - γvt

        At t = 0 in the lab frame:
            x'_rest = x_lab + (γ-1)(x_lab·n̂)n̂

        Returns:
            Tuple of (x', y', z') in rest frame
        """
        if self.v_mag < 1e-14:
            return x, y, z

        x_vec = np.array([x, y, z])

        # Project onto boost direction
        x_parallel = np.dot(x_vec, self.n_hat) * self.n_hat
        x_perp = x_vec - x_parallel

        # Transform: parallel component is Lorentz contracted in lab frame
        # so expand it to get rest frame coordinate
        x_rest = x_perp + self.lorentz_gamma * x_parallel

        return x_rest[0], x_rest[1], x_rest[2]

    def _get_rest_frame_H_and_l(self, x: float, y: float, z: float):
        """
        Get H and l in the rest frame at the transformed coordinates.

        This requires accessing internal methods of the base metric.
        We assume the base metric has _r_and_l and _H methods.
        """
        # Transform to rest frame coordinates
        xr, yr, zr = self._transform_coordinates(x, y, z)

        # Get H and l from base metric
        if hasattr(self.base_metric, '_r_and_l'):
            r, l_rest = self.base_metric._r_and_l(xr, yr, zr)
            if hasattr(self.base_metric, '_H'):
                if hasattr(self.base_metric, 'a'):  # Kerr
                    H_rest = self.base_metric._H(r, zr)
                else:  # Schwarzschild
                    H_rest = self.base_metric._H(r)
            else:
                H_rest = self.base_metric.M / r
        else:
            # Fallback: compute from metric
            r = np.sqrt(xr**2 + yr**2 + zr**2)
            l_rest = np.array([xr, yr, zr]) / max(r, 1e-14)
            H_rest = self.base_metric.M / max(r, 1e-14)

        return H_rest, l_rest, (xr, yr, zr)

    def _boost_4vector(self, l_0: float, l_spatial: np.ndarray):
        """
        Boost a 4-vector l_μ = (l_0, l_i) from rest frame to lab frame.

        For a boost with velocity v in direction n̂:
            l'_0 = γ(l_0 - v·l_spatial)
            l'_parallel = γ(l_parallel - v*l_0)
            l'_perp = l_perp

        Returns:
            Tuple of (l'_0, l'_spatial)
        """
        if self.v_mag < 1e-14:
            return l_0, l_spatial

        v = self.velocity
        gamma = self.lorentz_gamma

        # Transform temporal component
        v_dot_l = np.dot(v, l_spatial)
        l_0_boosted = gamma * (l_0 - v_dot_l)

        # Transform spatial components
        l_parallel = np.dot(l_spatial, self.n_hat) * self.n_hat
        l_perp = l_spatial - l_parallel
        l_spatial_boosted = l_perp + gamma * (l_parallel - v * l_0)

        return l_0_boosted, l_spatial_boosted

    def _get_4metric_components(self, x: float, y: float, z: float):
        """
        Compute the full 4-metric components g_μν at lab frame point.

        The Kerr-Schild form g_μν = η_μν + 2H l_μ l_ν is preserved,
        but we need to compute it properly in the lab frame.

        Returns:
            Tuple of (g_00, g_0i, g_ij, H, l_0, l_spatial)
        """
        # Get H and l in rest frame
        H_rest, l_rest, _ = self._get_rest_frame_H_and_l(x, y, z)

        # In rest frame, l_μ = (1, l_i) with l_i unit vector
        l_0_rest = 1.0

        # Boost to lab frame
        l_0, l_spatial = self._boost_4vector(l_0_rest, l_rest)

        # H is a scalar at the spacetime point - it doesn't transform
        # The Kerr-Schild form is g_μν = η_μν + 2H l_μ l_ν
        H = H_rest

        # Compute 4-metric components
        g_00 = -1.0 + 2 * H * l_0 * l_0
        g_0i = 2 * H * l_0 * l_spatial  # This is a 3-vector
        g_ij = np.eye(3) + 2 * H * np.outer(l_spatial, l_spatial)

        return g_00, g_0i, g_ij, H, l_0, l_spatial

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute boosted 3-metric γ_ij from full 4-metric decomposition.

        The 3-metric is just g_ij for this coordinate choice.
        """
        _, _, g_ij, _, _, _ = self._get_4metric_components(x, y, z)
        return g_ij

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute inverse boosted 3-metric.

        For γ_ij = δ_ij + 2H l_i l_j, the inverse is:
        γ^ij = δ^ij - 2H/(1 + 2H|l|²) l^i l^j
        """
        _, _, g_ij, H, _, l_spatial = self._get_4metric_components(x, y, z)

        # |l|² = l_i l^i (using flat space to raise)
        l_sq = np.dot(l_spatial, l_spatial)

        return np.eye(3) - (2 * H / (1 + 2 * H * l_sq)) * np.outer(l_spatial, l_spatial)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute ∂_k γ_ij numerically.
        """
        h = 1e-6

        dgamma = np.zeros((3, 3, 3))

        coords = [
            (x + h, y, z), (x - h, y, z),
            (x, y + h, z), (x, y - h, z),
            (x, y, z + h), (x, y, z - h)
        ]

        gammas = [self.gamma(*c) for c in coords]

        dgamma[0] = (gammas[0] - gammas[1]) / (2 * h)
        dgamma[1] = (gammas[2] - gammas[3]) / (2 * h)
        dgamma[2] = (gammas[4] - gammas[5]) / (2 * h)

        return dgamma

    def _dgamma_dt(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute ∂_t γ_ij for the boosted metric.

        The boosted black hole is moving, so the metric is time-dependent.
        At time t, the black hole center is at position (v*t, 0, 0).
        We compute the time derivative numerically.
        """
        dt = 1e-6
        v = self.velocity

        def gamma_at_time(t):
            """Compute 3-metric as if black hole center is at (v*t, 0, 0)."""
            # Position relative to black hole at time t
            x_rel = x - v[0] * t
            y_rel = y - v[1] * t
            z_rel = z - v[2] * t

            # Transform to rest frame
            if self.v_mag < 1e-14:
                x_rest, y_rest, z_rest = x_rel, y_rel, z_rel
            else:
                pos_rel = np.array([x_rel, y_rel, z_rel])
                pos_parallel = np.dot(pos_rel, self.n_hat) * self.n_hat
                pos_perp = pos_rel - pos_parallel
                pos_rest = pos_perp + self.lorentz_gamma * pos_parallel
                x_rest, y_rest, z_rest = pos_rest

            # Get H and l in rest frame
            r_rest = np.sqrt(x_rest**2 + y_rest**2 + z_rest**2)
            if r_rest < 1e-14:
                r_rest = 1e-14

            if hasattr(self.base_metric, '_r_and_l'):
                r, l_rest = self.base_metric._r_and_l(x_rest, y_rest, z_rest)
                if hasattr(self.base_metric, '_H'):
                    if hasattr(self.base_metric, 'a'):  # Kerr
                        H = self.base_metric._H(r, z_rest)
                    else:  # Schwarzschild
                        H = self.base_metric._H(r)
                else:
                    H = self.base_metric.M / r
            else:
                H = self.base_metric.M / r_rest
                l_rest = np.array([x_rest, y_rest, z_rest]) / r_rest

            # Boost l to lab frame
            l_0, l_spatial = self._boost_4vector(1.0, l_rest)

            # 3-metric
            return np.eye(3) + 2 * H * np.outer(l_spatial, l_spatial)

        # Numerical time derivative
        gamma_minus = gamma_at_time(-dt)
        gamma_plus = gamma_at_time(dt)
        return (gamma_plus - gamma_minus) / (2 * dt)

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute extrinsic curvature K_ij in the boosted frame.

        For a non-stationary spacetime (boosted black hole is moving):
        K_ij = (1/2α)(D_i β_j + D_j β_i - ∂_t γ_ij)
        """
        h = 1e-6
        alpha = self.lapse(x, y, z)

        # Get 3-metric and Christoffel at this point
        gamma_down = self.gamma(x, y, z)
        chris = self.christoffel(x, y, z)

        # Compute shift and its lowered version
        beta = self.shift(x, y, z)
        beta_down = gamma_down @ beta

        # Spatial derivatives of β_j (lowered)
        d_beta_down = np.zeros((3, 3))
        for i_dir in range(3):
            dx = [0.0, 0.0, 0.0]
            dx[i_dir] = h

            # Plus direction
            xp, yp, zp = x + dx[0], y + dx[1], z + dx[2]
            gamma_p = self.gamma(xp, yp, zp)
            beta_p = self.shift(xp, yp, zp)
            beta_down_p = gamma_p @ beta_p

            # Minus direction
            xm, ym, zm = x - dx[0], y - dx[1], z - dx[2]
            gamma_m = self.gamma(xm, ym, zm)
            beta_m = self.shift(xm, ym, zm)
            beta_down_m = gamma_m @ beta_m

            d_beta_down[i_dir, :] = (beta_down_p - beta_down_m) / (2 * h)

        # D_i β_j = ∂_i β_j - Γ^k_ij β_k
        D_beta = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                D_beta[i, j] = d_beta_down[i, j]
                for k in range(3):
                    D_beta[i, j] -= chris[k, i, j] * beta_down[k]

        # Time derivative of 3-metric (non-zero for boosted/moving black hole)
        dgamma_dt = self._dgamma_dt(x, y, z)

        # K_ij = (1/2α)(D_i β_j + D_j β_i - ∂_t γ_ij)
        K = 0.5 / alpha * (D_beta + D_beta.T - dgamma_dt)

        return K

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute lapse α from 4-metric decomposition.

        For Kerr-Schild: α² = 1/(1 + 2H l_0²)
        """
        _, _, _, H, l_0, _ = self._get_4metric_components(x, y, z)
        return 1.0 / np.sqrt(1 + 2 * H * l_0 * l_0)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute shift β^i from 4-metric decomposition.

        β_i = g_{0i} = 2H l_0 l_i
        β^i = γ^{ij} β_j
        """
        _, g_0i, _, H, l_0, l_spatial = self._get_4metric_components(x, y, z)

        # β_i = g_0i = 2H l_0 l_i
        beta_down = g_0i

        # Raise with inverse 3-metric
        gamma_inv = self.gamma_inv(x, y, z)
        beta_up = gamma_inv @ beta_down

        return beta_up


def boost_metric(base_metric: Metric, velocity: np.ndarray) -> BoostedMetric:
    """
    Create a boosted version of a metric.

    Args:
        base_metric: The metric to boost
        velocity: 3-vector boost velocity

    Returns:
        BoostedMetric instance
    """
    return BoostedMetric(base_metric, velocity)
