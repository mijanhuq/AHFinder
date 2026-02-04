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
        gamma: Lorentz factor
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
            self.gamma = 1.0 / np.sqrt(1 - v_mag**2)
            self.n_hat = velocity / v_mag
        else:
            self.gamma = 1.0
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
        x_rest = x_perp + self.gamma * x_parallel

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

    def _boost_H_and_l(self, H_rest: float, l_rest: np.ndarray):
        """
        Transform H and l under a Lorentz boost.

        The null vector l_μ = (l_0, l_i) transforms as:
            l'_0 = γ(l_0 - v·l)
            l'_i = l_i + [(γ-1)l_∥ - γl_0 v]_i

        where l_∥ = (l·n̂)n̂ is the component parallel to boost.

        H transforms to keep g_μν = η_μν + 2H l_μ l_ν covariant:
            H' = H (in coordinate-invariant sense, but l changes)

        Actually, for Kerr-Schild: H' l'_μ l'_ν = H l_μ l_ν under boost,
        so H' = H / (l'_0)² when l_0 = 1.

        Returns:
            Tuple of (H_boosted, l_boosted_spatial)
        """
        if self.v_mag < 1e-14:
            return H_rest, l_rest

        v = self.velocity
        gamma = self.gamma

        # Rest frame: l_μ = (1, l_i) is ingoing null
        l_0_rest = 1.0
        l_spatial_rest = l_rest

        # v·l (spatial)
        v_dot_l = np.dot(v, l_spatial_rest)

        # Boosted null vector (4-vector)
        l_0_boosted = gamma * (l_0_rest - v_dot_l)

        # Spatial part
        l_parallel = np.dot(l_spatial_rest, self.n_hat) * self.n_hat
        l_perp = l_spatial_rest - l_parallel

        l_spatial_boosted = l_perp + gamma * (l_parallel - v * l_0_rest)

        # Normalize spatial part to have unit magnitude dotted with itself
        # in the boosted frame metric (which is still ~flat far from horizon)
        l_mag = np.sqrt(np.dot(l_spatial_boosted, l_spatial_boosted))
        if l_mag > 1e-14:
            l_spatial_boosted = l_spatial_boosted / l_mag

        # H transformation: H' (l'_0)² = H (l_0)² for preserved Kerr-Schild form
        # So H' = H × (l_0 / l'_0)²
        if abs(l_0_boosted) > 1e-14:
            H_boosted = H_rest * (l_0_rest / l_0_boosted)**2
        else:
            H_boosted = H_rest

        return H_boosted, l_spatial_boosted

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute boosted 3-metric γ_ij = δ_ij + 2H l_i l_j.
        """
        H_rest, l_rest, _ = self._get_rest_frame_H_and_l(x, y, z)
        H, l = self._boost_H_and_l(H_rest, l_rest)

        return np.eye(3) + 2 * H * np.outer(l, l)

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute inverse boosted 3-metric.
        """
        H_rest, l_rest, _ = self._get_rest_frame_H_and_l(x, y, z)
        H, l = self._boost_H_and_l(H_rest, l_rest)

        return np.eye(3) - (2 * H / (1 + 2 * H)) * np.outer(l, l)

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

        # Note: using self.gamma directly would cause recursion
        # We need to compute the 3-metric at each point
        gammas = []
        for cx, cy, cz in coords:
            H_rest, l_rest, _ = self._get_rest_frame_H_and_l(cx, cy, cz)
            H, l = self._boost_H_and_l(H_rest, l_rest)
            gammas.append(np.eye(3) + 2 * H * np.outer(l, l))

        dgamma[0] = (gammas[0] - gammas[1]) / (2 * h)
        dgamma[1] = (gammas[2] - gammas[3]) / (2 * h)
        dgamma[2] = (gammas[4] - gammas[5]) / (2 * h)

        return dgamma

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute extrinsic curvature K_ij in the boosted frame.

        Uses the relation K_ij = -(1/2α)(D_i β_j + D_j β_i) for
        stationary spacetimes.
        """
        h = 1e-6
        alpha = self.lapse(x, y, z)

        # Get 3-metric and Christoffel at this point
        gamma_down = np.eye(3)
        H_rest, l_rest, _ = self._get_rest_frame_H_and_l(x, y, z)
        H, l = self._boost_H_and_l(H_rest, l_rest)
        gamma_down = np.eye(3) + 2 * H * np.outer(l, l)

        chris = self.christoffel(x, y, z)

        # Compute shift and its derivatives
        beta = self.shift(x, y, z)
        beta_down = gamma_down @ beta

        # Spatial derivatives of β_j (lowered)
        d_beta_down = np.zeros((3, 3))
        for i_dir in range(3):
            dx = [0.0, 0.0, 0.0]
            dx[i_dir] = h

            # Plus direction
            xp, yp, zp = x + dx[0], y + dx[1], z + dx[2]
            H_rest_p, l_rest_p, _ = self._get_rest_frame_H_and_l(xp, yp, zp)
            H_p, l_p = self._boost_H_and_l(H_rest_p, l_rest_p)
            gamma_p = np.eye(3) + 2 * H_p * np.outer(l_p, l_p)
            beta_p = 2 * H_p * l_p / (1 + 2 * H_p)
            beta_down_p = gamma_p @ beta_p

            # Minus direction
            xm, ym, zm = x - dx[0], y - dx[1], z - dx[2]
            H_rest_m, l_rest_m, _ = self._get_rest_frame_H_and_l(xm, ym, zm)
            H_m, l_m = self._boost_H_and_l(H_rest_m, l_rest_m)
            gamma_m = np.eye(3) + 2 * H_m * np.outer(l_m, l_m)
            beta_m = 2 * H_m * l_m / (1 + 2 * H_m)
            beta_down_m = gamma_m @ beta_m

            d_beta_down[i_dir, :] = (beta_down_p - beta_down_m) / (2 * h)

        # D_i β_j = ∂_i β_j - Γ^k_ij β_k
        D_beta = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                D_beta[i, j] = d_beta_down[i, j]
                for k in range(3):
                    D_beta[i, j] -= chris[k, i, j] * beta_down[k]

        # K_ij = -(1/2α)(D_i β_j + D_j β_i)
        K = -0.5 / alpha * (D_beta + D_beta.T)

        return K

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute lapse α = 1/√(1 + 2H) in boosted frame.
        """
        H_rest, l_rest, _ = self._get_rest_frame_H_and_l(x, y, z)
        H, _ = self._boost_H_and_l(H_rest, l_rest)
        return 1.0 / np.sqrt(1 + 2 * H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute shift β^i = 2H l^i / (1 + 2H) in boosted frame.
        """
        H_rest, l_rest, _ = self._get_rest_frame_H_and_l(x, y, z)
        H, l = self._boost_H_and_l(H_rest, l_rest)
        return 2 * H * l / (1 + 2 * H)


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
