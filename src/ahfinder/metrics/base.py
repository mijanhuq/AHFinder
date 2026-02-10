"""
Abstract base class for metric data.

Defines the interface for providing metric quantities needed by the
apparent horizon finder: the 3-metric γ_ij, its inverse, derivatives,
and the extrinsic curvature K_ij.

Reference: Huq, Choptuik & Matzner (2000), Section II.E
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Metric(ABC):
    """
    Abstract base class for spacetime metric data in 3+1 form.

    Subclasses must implement methods to provide:
    - γ_ij: The 3-metric (spatial metric)
    - γ^ij: The inverse 3-metric
    - ∂_k γ_ij: Partial derivatives of the 3-metric
    - K_ij: The extrinsic curvature

    Optional methods provide derived quantities like Christoffel symbols
    and traces.
    """

    @abstractmethod
    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute the 3-metric γ_ij at a point.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            3×3 numpy array representing γ_ij
        """
        pass

    @abstractmethod
    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute the inverse 3-metric γ^ij at a point.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            3×3 numpy array representing γ^ij
        """
        pass

    @abstractmethod
    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute partial derivatives of the 3-metric: ∂_k γ_ij.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            3×3×3 numpy array where result[k, i, j] = ∂_k γ_ij
        """
        pass

    @abstractmethod
    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute the extrinsic curvature K_ij at a point.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            3×3 numpy array representing K_ij
        """
        pass

    def K_trace(self, x: float, y: float, z: float) -> float:
        """
        Compute the trace of extrinsic curvature: K = γ^ij K_ij.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            Trace K
        """
        gamma_up = self.gamma_inv(x, y, z)
        K = self.extrinsic_curvature(x, y, z)
        return np.einsum('ij,ij->', gamma_up, K)

    def christoffel(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^i_jk from the 3-metric.

        Γ^i_jk = (1/2) γ^il (∂_j γ_lk + ∂_k γ_jl - ∂_l γ_jk)

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            3×3×3 numpy array where result[i, j, k] = Γ^i_jk
        """
        gamma_up = self.gamma_inv(x, y, z)
        dg = self.dgamma(x, y, z)

        # Γ^i_jk = (1/2) γ^il (∂_j γ_lk + ∂_k γ_jl - ∂_l γ_jk) (vectorized)
        # dg[k,i,j] = ∂_k γ_{ij}, so we transpose to get the right indices
        bracket = dg.transpose(1, 0, 2) + dg.transpose(2, 1, 0) - dg
        chris = 0.5 * np.einsum('il,ljk->ijk', gamma_up, bracket)
        return chris

    def christoffel_contracted(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute contracted Christoffel symbols Γ^i = γ^jk Γ^i_jk.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            Array of shape (3,) with Γ^i values
        """
        gamma_up = self.gamma_inv(x, y, z)
        chris = self.christoffel(x, y, z)
        return np.einsum('jk,ijk->i', gamma_up, chris)

    def lapse(self, x: float, y: float, z: float) -> float:
        """
        Compute the lapse function α.

        Default implementation returns 1 (geodesic slicing).
        Subclasses should override for non-trivial lapse.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            Lapse α
        """
        return 1.0

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Compute the shift vector β^i.

        Default implementation returns zero shift.
        Subclasses should override for non-trivial shift.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            Array of shape (3,) with β^i
        """
        return np.zeros(3)


class FlatMetric(Metric):
    """
    Flat (Minkowski) spacetime metric for testing.

    γ_ij = δ_ij (identity), K_ij = 0
    """

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        return np.eye(3)

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        return np.eye(3)

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        return np.zeros((3, 3, 3))

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        return np.zeros((3, 3))


def compute_dgamma_numerical(
    metric: Metric,
    x: float,
    y: float,
    z: float,
    h: float = 1e-6
) -> np.ndarray:
    """
    Compute ∂_k γ_ij numerically using central differences.

    Useful for verifying analytical derivatives or when only
    the metric itself is available.

    Args:
        metric: Metric instance
        x, y, z: Point coordinates
        h: Finite difference step size

    Returns:
        3×3×3 array with ∂_k γ_ij
    """
    dgamma = np.zeros((3, 3, 3))

    # Coordinate offsets
    coords = [(x, y, z) for _ in range(6)]
    coords[0] = (x + h, y, z)
    coords[1] = (x - h, y, z)
    coords[2] = (x, y + h, z)
    coords[3] = (x, y - h, z)
    coords[4] = (x, y, z + h)
    coords[5] = (x, y, z - h)

    # Evaluate metric at offset points
    gammas = [metric.gamma(*c) for c in coords]

    # Central differences
    dgamma[0] = (gammas[0] - gammas[1]) / (2 * h)  # ∂_x
    dgamma[1] = (gammas[2] - gammas[3]) / (2 * h)  # ∂_y
    dgamma[2] = (gammas[4] - gammas[5]) / (2 * h)  # ∂_z

    return dgamma
