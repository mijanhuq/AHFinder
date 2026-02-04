"""
Kerr metric in Kerr-Schild coordinates.

The Kerr-Schild form of the Kerr metric is:
    g_μν = η_μν + 2H l_μ l_ν

where:
    H = Mr³ / (r⁴ + a²z²)
    l_μ = (1, (rx + ay)/(r² + a²), (ry - ax)/(r² + a²), z/r)

The radius r is defined implicitly by:
    x² + y² + z² = r² + a²(1 - z²/r²)

Reference: Huq, Choptuik & Matzner (2000), Eqs. 11-29
"""

import numpy as np
from scipy.optimize import brentq
from .base import Metric


class KerrMetric(Metric):
    """
    Kerr spacetime in Kerr-Schild (horizon-penetrating) coordinates.

    The 3+1 decomposition gives:
        α = 1/√(1 + 2H)
        β^i = 2H l^i / (1 + 2H)
        γ_ij = δ_ij + 2H l_i l_j

    Attributes:
        M: Black hole mass
        a: Spin parameter (|a| ≤ M)
    """

    def __init__(self, M: float = 1.0, a: float = 0.0):
        """
        Initialize Kerr metric.

        Args:
            M: Black hole mass (default 1.0)
            a: Spin parameter (default 0, Schwarzschild limit)
        """
        if M <= 0:
            raise ValueError("Mass must be positive")
        if abs(a) > M:
            raise ValueError("|a| must be ≤ M for Kerr black hole")

        self.M = M
        self.a = a

    def _compute_r(self, x: float, y: float, z: float) -> float:
        """
        Compute the Kerr radial coordinate r from Cartesian coordinates.

        r is defined implicitly by: x² + y² + z² = r² + a²(1 - z²/r²)
        which can be rewritten as: r⁴ - (x² + y² + z² - a²)r² - a²z² = 0

        For a = 0, this reduces to r = √(x² + y² + z²).
        """
        a = self.a
        R_sq = x**2 + y**2 + z**2

        if abs(a) < 1e-14:
            return np.sqrt(R_sq)

        # Solve r⁴ - (R² - a²)r² - a²z² = 0
        # Let u = r², then: u² - (R² - a²)u - a²z² = 0
        b = -(R_sq - a**2)
        c = -(a**2) * (z**2)

        discriminant = b**2 - 4 * c
        if discriminant < 0:
            discriminant = 0

        u = (-b + np.sqrt(discriminant)) / 2

        if u < 0:
            u = 0

        return np.sqrt(u)

    def _r_and_l(self, x: float, y: float, z: float):
        """
        Compute Kerr r and the null vector l^i.

        l^i = ((rx + ay)/(r² + a²), (ry - ax)/(r² + a²), z/r)

        Returns:
            Tuple of (r, l_vector) where l is shape (3,)
        """
        r = self._compute_r(x, y, z)
        a = self.a

        if r < 1e-14:
            return 1e-14, np.array([1.0, 0.0, 0.0])

        r2_a2 = r**2 + a**2

        l = np.array([
            (r * x + a * y) / r2_a2,
            (r * y - a * x) / r2_a2,
            z / r
        ])

        return r, l

    def _H(self, r: float, z: float) -> float:
        """
        Compute H = Mr³ / (r⁴ + a²z²).
        """
        a = self.a
        denom = r**4 + a**2 * z**2
        if denom < 1e-28:
            return self.M * r**3 / 1e-28
        return self.M * r**3 / denom

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute 3-metric γ_ij = δ_ij + 2H l_i l_j.
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r, z)

        gamma = np.eye(3) + 2 * H * np.outer(l, l)
        return gamma

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute inverse 3-metric γ^ij.

        γ^ij = δ^ij - 2H/(1+2H) l^i l^j
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r, z)

        gamma_inv = np.eye(3) - (2 * H / (1 + 2 * H)) * np.outer(l, l)
        return gamma_inv

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute ∂_k γ_ij numerically.

        The analytical expressions for Kerr are quite complex, so we
        use numerical differentiation for robustness.
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

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute extrinsic curvature K_ij numerically.

        K_ij can be computed from the time derivative of γ_ij and
        the Lie derivative along β, but for numerical stability we
        use the general Kerr-Schild formula.
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r, z)
        a = self.a

        if r < 1e-10:
            return np.zeros((3, 3))

        alpha = 1.0 / np.sqrt(1 + 2 * H)
        l_outer = np.outer(l, l)
        delta = np.eye(3)

        # For Kerr-Schild, K_ij involves derivatives of H and l
        # We compute numerically for accuracy
        h = 1e-6

        # Compute spatial derivatives of (H l_i l_j)
        dHll = np.zeros((3, 3, 3))
        coords = [
            (x + h, y, z), (x - h, y, z),
            (x, y + h, z), (x, y - h, z),
            (x, y, z + h), (x, y, z - h)
        ]

        Hll_vals = []
        for cx, cy, cz in coords:
            r_c, l_c = self._r_and_l(cx, cy, cz)
            H_c = self._H(r_c, cz)
            Hll_vals.append(H_c * np.outer(l_c, l_c))

        dHll[0] = (Hll_vals[0] - Hll_vals[1]) / (2 * h)
        dHll[1] = (Hll_vals[2] - Hll_vals[3]) / (2 * h)
        dHll[2] = (Hll_vals[4] - Hll_vals[5]) / (2 * h)

        # K_ij = α [∂_i(Hl_j) + ∂_j(Hl_i) + 2H l_k ∂_k(l_i l_j)]
        # This is a simplification; full formula in paper

        # Use finite difference on full extrinsic curvature relation
        # K_ij = -(1/2α)(∂_t γ_ij - D_i β_j - D_j β_i)
        # For stationary spacetime, ∂_t γ = 0

        beta = self.shift(x, y, z)
        gamma_down = self.gamma(x, y, z)

        # Compute Christoffel and covariant derivative of shift
        chris = self.christoffel(x, y, z)

        # D_i β_j = ∂_i β_j - Γ^k_ij β_k
        # First get ∂_i β_j
        d_beta = np.zeros((3, 3))
        coords_list = [(h, 0, 0), (-h, 0, 0), (0, h, 0), (0, -h, 0), (0, 0, h), (0, 0, -h)]

        for k_dir in range(3):
            dx = [0, 0, 0]
            dx[k_dir] = h
            beta_p = self.shift(x + dx[0], y + dx[1], z + dx[2])
            beta_m = self.shift(x - dx[0], y - dx[1], z - dx[2])
            d_beta[k_dir, :] = (beta_p - beta_m) / (2 * h)

        # Lower index on beta: β_j = γ_jk β^k
        beta_down = gamma_down @ beta

        # ∂_i β_j with lowered index
        d_beta_down = np.zeros((3, 3))
        for i_dir in range(3):
            dx = [0, 0, 0]
            dx[i_dir] = h
            gamma_p = self.gamma(x + dx[0], y + dx[1], z + dx[2])
            beta_p = self.shift(x + dx[0], y + dx[1], z + dx[2])
            gamma_m = self.gamma(x - dx[0], y - dx[1], z - dx[2])
            beta_m = self.shift(x - dx[0], y - dx[1], z - dx[2])

            beta_down_p = gamma_p @ beta_p
            beta_down_m = gamma_m @ beta_m

            d_beta_down[i_dir, :] = (beta_down_p - beta_down_m) / (2 * h)

        # D_i β_j = ∂_i β_j - Γ^k_ij β_k (with j lowered)
        D_beta = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                D_beta[i, j] = d_beta_down[i, j]
                for k in range(3):
                    D_beta[i, j] -= chris[k, i, j] * beta_down[k]

        # K_ij = -(1/2α)(D_i β_j + D_j β_i)
        K = -0.5 / alpha * (D_beta + D_beta.T)

        return K

    def K_trace(self, x: float, y: float, z: float) -> float:
        """
        Compute trace K = γ^ij K_ij.
        """
        gamma_inv = self.gamma_inv(x, y, z)
        K = self.extrinsic_curvature(x, y, z)
        return np.einsum('ij,ij->', gamma_inv, K)

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute lapse α = 1/√(1 + 2H).
        """
        r, _ = self._r_and_l(x, y, z)
        H = self._H(r, z)
        return 1.0 / np.sqrt(1 + 2 * H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute shift β^i = 2H l^i / (1 + 2H).
        """
        r, l = self._r_and_l(x, y, z)
        H = self._H(r, z)
        return 2 * H * l / (1 + 2 * H)

    def horizon_radius_equatorial(self) -> float:
        """
        Return the equatorial coordinate radius of the event horizon.

        r_+ = M + √(M² - a²)
        """
        return self.M + np.sqrt(self.M**2 - self.a**2)

    def horizon_radius_polar(self) -> float:
        """
        Return the polar coordinate radius of the event horizon.

        At the poles, the horizon radius is r_+ (same as equatorial).
        """
        return self.horizon_radius_equatorial()

    def horizon_area(self) -> float:
        """
        Return the analytical area of the Kerr horizon.

        A = 4π(r_+² + a²) = 8πMr_+
        """
        r_plus = self.horizon_radius_equatorial()
        return 4 * np.pi * (r_plus**2 + self.a**2)

    def irreducible_mass(self) -> float:
        """
        Compute the irreducible mass from horizon area.

        M_irr = √(A / 16π)
        """
        A = self.horizon_area()
        return np.sqrt(A / (16 * np.pi))
