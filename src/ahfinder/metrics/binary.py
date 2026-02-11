"""
Binary black hole metric using superposed Kerr-Schild data.

The superposed Kerr-Schild form for two black holes:
    g_μν = η_μν + 2H₁ l₁_μ l₁_ν + 2H₂ l₂_μ l₂_ν

where for each black hole (i = 1, 2):
    H_i = M_i / r_i
    l_i = (x - x_i) / |x - x_i|
    r_i = |x - x_i|

This is a commonly used approximation for binary black hole initial data.
It is exactly correct for a single black hole and approximately correct
for well-separated black holes.

Reference:
- Shoemaker, Huq & Matzner (2000) - arXiv:gr-qc/0004062
- Matzner et al. (1999) - arXiv:gr-qc/9812012
"""

import numpy as np
from typing import Tuple, Optional
from .base import Metric


class BinaryBlackHoleMetric(Metric):
    """
    Binary black hole spacetime using superposed Kerr-Schild data.

    This creates initial data for two Schwarzschild black holes
    by superposing their Kerr-Schild representations:

        γ_ij = δ_ij + 2H₁ l₁_i l₁_j + 2H₂ l₂_i l₂_j

    The lapse and shift are computed consistently from the superposed data.

    For well-separated black holes (separation >> M₁ + M₂), this gives
    two distinct apparent horizons. As the separation decreases, the
    horizons merge into a single "peanut-shaped" common horizon.

    Args:
        M1: Mass of first black hole
        M2: Mass of second black hole
        position1: Position (x, y, z) of first black hole
        position2: Position (x, y, z) of second black hole
        momentum1: Initial momentum of first black hole (optional)
        momentum2: Initial momentum of second black hole (optional)
    """

    def __init__(
        self,
        M1: float = 1.0,
        M2: float = 1.0,
        position1: Tuple[float, float, float] = (-3.0, 0.0, 0.0),
        position2: Tuple[float, float, float] = (3.0, 0.0, 0.0),
        momentum1: Optional[Tuple[float, float, float]] = None,
        momentum2: Optional[Tuple[float, float, float]] = None
    ):
        if M1 <= 0 or M2 <= 0:
            raise ValueError("Masses must be positive")

        self.M1 = M1
        self.M2 = M2
        self.pos1 = np.array(position1)
        self.pos2 = np.array(position2)

        # Momenta for boosted case (not yet implemented)
        self.P1 = np.array(momentum1) if momentum1 is not None else np.zeros(3)
        self.P2 = np.array(momentum2) if momentum2 is not None else np.zeros(3)

        # Separation
        self.separation = np.linalg.norm(self.pos2 - self.pos1)

    def _r_and_l(
        self,
        x: float,
        y: float,
        z: float,
        center: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute distance and null vector from a given center.

        Args:
            x, y, z: Point coordinates
            center: Black hole position

        Returns:
            (r, l) where r is distance and l is unit vector toward point
        """
        dx = np.array([x, y, z]) - center
        r = np.linalg.norm(dx)

        if r < 1e-14:
            return 1e-14, np.array([1.0, 0.0, 0.0])

        l = dx / r
        return r, l

    def _H_single(self, M: float, r: float) -> float:
        """Compute H = M/r for a single black hole."""
        return M / max(r, 1e-14)

    def _compute_H_l(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[float, np.ndarray, float, np.ndarray]:
        """
        Compute H and l for both black holes.

        Returns:
            (H1, l1, H2, l2)
        """
        r1, l1 = self._r_and_l(x, y, z, self.pos1)
        r2, l2 = self._r_and_l(x, y, z, self.pos2)

        H1 = self._H_single(self.M1, r1)
        H2 = self._H_single(self.M2, r2)

        return H1, l1, H2, l2

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute 3-metric γ_ij = δ_ij + 2H₁ l₁_i l₁_j + 2H₂ l₂_i l₂_j.
        """
        H1, l1, H2, l2 = self._compute_H_l(x, y, z)

        gamma = (np.eye(3)
                 + 2 * H1 * np.outer(l1, l1)
                 + 2 * H2 * np.outer(l2, l2))
        return gamma

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute inverse 3-metric γ^ij.

        For the superposed case, we need to invert numerically since
        the Sherman-Morrison formula doesn't directly apply.
        """
        gamma = self.gamma(x, y, z)
        return np.linalg.inv(gamma)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute ∂_k γ_ij.

        Uses sum of analytical derivatives from each black hole:
        ∂_k γ_ij = ∂_k(2H₁ l₁_i l₁_j) + ∂_k(2H₂ l₂_i l₂_j)
        """
        dgamma = np.zeros((3, 3, 3))

        # Add contribution from each black hole
        for M, center in [(self.M1, self.pos1), (self.M2, self.pos2)]:
            r, l = self._r_and_l(x, y, z, center)
            H = self._H_single(M, r)

            for k in range(3):
                # ∂_k H = -H l_k / r
                dH_dk = -H * l[k] / r

                for i in range(3):
                    # ∂_k l_i = (δ_ki - l_k l_i) / r
                    dl_i_dk = ((1 if k == i else 0) - l[k] * l[i]) / r

                    for j in range(3):
                        dl_j_dk = ((1 if k == j else 0) - l[k] * l[j]) / r

                        dgamma[k, i, j] += (
                            2 * dH_dk * l[i] * l[j]
                            + 2 * H * dl_i_dk * l[j]
                            + 2 * H * l[i] * dl_j_dk
                        )

        return dgamma

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute lapse for superposed Kerr-Schild.

        α = 1/√(1 + 2H_total) where H_total = H₁ + H₂

        Note: This is an approximation. The exact form is more complex.
        """
        H1, _, H2, _ = self._compute_H_l(x, y, z)
        H_total = H1 + H2
        return 1.0 / np.sqrt(1 + 2 * H_total)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute shift for superposed Kerr-Schild.

        β^i = 2(H₁ l₁^i + H₂ l₂^i) / (1 + 2H_total)
        """
        H1, l1, H2, l2 = self._compute_H_l(x, y, z)
        H_total = H1 + H2

        beta = 2 * (H1 * l1 + H2 * l2) / (1 + 2 * H_total)
        return beta

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute extrinsic curvature K_ij.

        Uses numerical differentiation of the shift vector:
        K_ij = (1/2α)(D_i β_j + D_j β_i)

        for the time-symmetric case (∂_t γ = 0).
        """
        h = 1e-6
        alpha = self.lapse(x, y, z)
        gamma_down = self.gamma(x, y, z)
        chris = self.christoffel(x, y, z)

        # Compute ∂_i β_j (with lowered index)
        d_beta_down = np.zeros((3, 3))

        for i_dir in range(3):
            dx = [0.0, 0.0, 0.0]
            dx[i_dir] = h

            gamma_p = self.gamma(x + dx[0], y + dx[1], z + dx[2])
            beta_p = self.shift(x + dx[0], y + dx[1], z + dx[2])
            gamma_m = self.gamma(x - dx[0], y - dx[1], z - dx[2])
            beta_m = self.shift(x - dx[0], y - dx[1], z - dx[2])

            beta_down_p = gamma_p @ beta_p
            beta_down_m = gamma_m @ beta_m

            d_beta_down[i_dir, :] = (beta_down_p - beta_down_m) / (2 * h)

        # Lower index on beta
        beta = self.shift(x, y, z)
        beta_down = gamma_down @ beta

        # D_i β_j = ∂_i β_j - Γ^k_ij β_k
        D_beta = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                D_beta[i, j] = d_beta_down[i, j]
                for k in range(3):
                    D_beta[i, j] -= chris[k, i, j] * beta_down[k]

        # K_ij = (1/2α)(D_i β_j + D_j β_i)
        K = 0.5 / alpha * (D_beta + D_beta.T)

        return K

    def K_trace(self, x: float, y: float, z: float) -> float:
        """Compute trace K = γ^ij K_ij."""
        gamma_inv = self.gamma_inv(x, y, z)
        K = self.extrinsic_curvature(x, y, z)
        return np.einsum('ij,ij->', gamma_inv, K)

    def horizon_radius_estimate(self, bh_index: int = 1) -> float:
        """
        Estimate the horizon radius for one of the black holes.

        This is an approximation based on the isolated black hole
        horizon radius, ignoring the effect of the companion.

        Args:
            bh_index: 1 or 2 for which black hole

        Returns:
            Estimated horizon radius
        """
        M = self.M1 if bh_index == 1 else self.M2
        return 2 * M

    def get_bh_center(self, bh_index: int = 1) -> np.ndarray:
        """Get the position of a black hole."""
        return self.pos1.copy() if bh_index == 1 else self.pos2.copy()


def create_binary_schwarzschild(
    M1: float = 1.0,
    M2: float = 1.0,
    separation: float = 6.0,
    axis: str = 'x'
) -> BinaryBlackHoleMetric:
    """
    Convenience function to create a binary black hole system.

    Places the black holes symmetrically along the specified axis.

    Args:
        M1: Mass of first black hole
        M2: Mass of second black hole
        separation: Distance between the black holes
        axis: 'x', 'y', or 'z' for orientation

    Returns:
        BinaryBlackHoleMetric instance
    """
    d = separation / 2

    if axis == 'x':
        pos1 = (-d, 0.0, 0.0)
        pos2 = (d, 0.0, 0.0)
    elif axis == 'y':
        pos1 = (0.0, -d, 0.0)
        pos2 = (0.0, d, 0.0)
    elif axis == 'z':
        pos1 = (0.0, 0.0, -d)
        pos2 = (0.0, 0.0, d)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    return BinaryBlackHoleMetric(M1, M2, pos1, pos2)
