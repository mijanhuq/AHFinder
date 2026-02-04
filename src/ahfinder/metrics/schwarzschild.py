"""
Schwarzschild metric in Kerr-Schild coordinates.

The Kerr-Schild form of the metric is:
    g_μν = η_μν + 2H l_μ l_ν

For Schwarzschild:
    H = M/r
    l_μ = (1, x/r, y/r, z/r)

This form is horizon-penetrating and regular at r = 2M.

Reference: Huq, Choptuik & Matzner (2000), Eqs. 10-29
"""

import numpy as np
from .base import Metric


class SchwarzschildMetric(Metric):
    """
    Schwarzschild spacetime in Kerr-Schild coordinates.

    The 3+1 decomposition gives:
        α = 1/√(1 + 2H)
        β^i = 2H l^i / (1 + 2H)
        γ_ij = δ_ij + 2H l_i l_j

    where H = M/r and l^i = x^i/r.

    Attributes:
        M: Black hole mass
    """

    def __init__(self, M: float = 1.0):
        """
        Initialize Schwarzschild metric.

        Args:
            M: Black hole mass (default 1.0)
        """
        if M <= 0:
            raise ValueError("Mass must be positive")
        self.M = M

    def _r_and_l(self, x: float, y: float, z: float):
        """
        Compute r and the null vector l^i.

        Returns:
            Tuple of (r, l_vector) where l is shape (3,)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-14:
            # At origin, return small r and arbitrary direction
            return 1e-14, np.array([1.0, 0.0, 0.0])

        l = np.array([x, y, z]) / r
        return r, l

    def _H(self, r: float) -> float:
        """Compute H = M/r."""
        return self.M / r

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute 3-metric γ_ij = δ_ij + 2H l_i l_j.
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r)

        gamma = np.eye(3) + 2 * H * np.outer(l, l)
        return gamma

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute inverse 3-metric γ^ij.

        Using Sherman-Morrison formula for inverse of (I + 2H l⊗l):
        γ^ij = δ^ij - 2H/(1+2H) l^i l^j
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r)

        gamma_inv = np.eye(3) - (2 * H / (1 + 2 * H)) * np.outer(l, l)
        return gamma_inv

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute ∂_k γ_ij analytically.

        γ_ij = δ_ij + 2H l_i l_j
        ∂_k γ_ij = 2 (∂_k H) l_i l_j + 2H (∂_k l_i) l_j + 2H l_i (∂_k l_j)

        where:
        ∂_k H = -M/r² × (x^k/r) = -H/r × l_k
        ∂_k l_i = (δ_ki - l_k l_i) / r
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r)

        dgamma = np.zeros((3, 3, 3))

        for k in range(3):
            # ∂_k H = -H l_k / r
            dH_dk = -H * l[k] / r

            for i in range(3):
                # ∂_k l_i = (δ_ki - l_k l_i) / r
                dl_i_dk = ((1 if k == i else 0) - l[k] * l[i]) / r

                for j in range(3):
                    dl_j_dk = ((1 if k == j else 0) - l[k] * l[j]) / r

                    dgamma[k, i, j] = (
                        2 * dH_dk * l[i] * l[j]
                        + 2 * H * dl_i_dk * l[j]
                        + 2 * H * l[i] * dl_j_dk
                    )

        return dgamma

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute extrinsic curvature K_ij.

        For Kerr-Schild Schwarzschild (Eq. 29 in paper):
        K_ij = -2α H/r [(1 + H) l_i l_j - (l_i n_j + n_i l_j) + (1-H)/(2H) (γ_ij - δ_ij)]

        Simplified form using K_ij = 2H/(r(1+2H)^(3/2)) × [δ_ij - (1+3H) l_i l_j]

        Actually using the standard form:
        K_ij = 2H/[r√(1+2H)] × [(1+H) l_i l_j + 1/2 (γ_ij - δ_ij)/H - (δ_ij - l_i l_j)]
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r)

        # Lapse
        alpha = 1.0 / np.sqrt(1 + 2 * H)

        # Build K_ij using the relation for Kerr-Schild spacetimes
        # K_ij = (2αH/r) × [ (1+H) l_i l_j - (δ_ij - l_i l_j) ]
        # But we need to be more careful...

        # The general formula for Kerr-Schild:
        # K_ij = α [ 2H,_i l_j + 2H l_i,_j + ... ]
        # For Schwarzschild specifically:

        l_outer = np.outer(l, l)
        delta = np.eye(3)

        # Using: K_ij = (2H/r) α [ -(1+H) l_i l_j + (δ_ij - l_i l_j) ]
        # which simplifies to: K_ij = (2Hα/r) [ δ_ij - (2+H) l_i l_j ]

        factor = 2 * H * alpha / r
        K = factor * (delta - (2 + H) * l_outer)

        return K

    def K_trace(self, x: float, y: float, z: float) -> float:
        """
        Compute trace K = γ^ij K_ij.

        For Schwarzschild in Kerr-Schild:
        K = 2M(2r + 3M) / [r²(r + 2M)^(3/2)]
        """
        r, l = self._r_and_l(x, y, z)
        M = self.M
        H = M / r

        alpha = 1.0 / np.sqrt(1 + 2 * H)

        # K = γ^ij K_ij computed directly
        gamma_inv = self.gamma_inv(x, y, z)
        K_tensor = self.extrinsic_curvature(x, y, z)

        return np.einsum('ij,ij->', gamma_inv, K_tensor)

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute lapse α = 1/√(1 + 2H).
        """
        r, _ = self._r_and_l(x, y, z)
        H = self._H(r)
        return 1.0 / np.sqrt(1 + 2 * H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute shift β^i = 2H l^i / (1 + 2H).
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r)
        return 2 * H * l / (1 + 2 * H)

    def horizon_radius(self) -> float:
        """
        Return the coordinate radius of the event horizon.

        For Schwarzschild in Kerr-Schild coordinates: r_H = 2M
        """
        return 2 * self.M
