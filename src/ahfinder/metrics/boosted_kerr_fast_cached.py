"""
Fast boosted Kerr metric with caching to eliminate redundant computation.

Same as boosted_kerr_fast.py but caches intermediate results to avoid
recomputing H, l, dH, dl, gamma_inv, dgamma, christoffel, and K multiple times
for the same point.

Also provides an all-in-one JIT function that computes everything in a single
call, avoiding Python overhead between separate JIT function calls.
"""

import numpy as np
from numba import jit
from .base import Metric

# Import the JIT-compiled functions from the original module
from .boosted_kerr_fast import (
    _compute_H_l_rest_frame,
    _compute_dH_dl_numerical,
    _compute_dgamma_from_H_l,
    _compute_christoffel_from_dgamma,
    _compute_K_from_H_l_chris,
    _transform_derivatives_to_lab,
)


@jit(nopython=True, cache=True)
def _compute_all_geometric_jit(x_lab: np.ndarray, M: float, a: float, Lambda: np.ndarray):
    """
    Compute all geometric quantities in a single JIT function.

    This avoids Python overhead from calling multiple JIT functions separately.

    Returns:
        gamma_inv: (3,3) inverse spatial metric
        dgamma: (3,3,3) spatial metric derivatives
        K: (3,3) extrinsic curvature
        K_trace: scalar trace of K
    """
    # Transform to rest frame
    x_rest = Lambda @ x_lab

    # Base computation: H, l, dH, dl
    H, l, r = _compute_H_l_rest_frame(x_rest, M, a)
    dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, M, a)
    dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, Lambda)

    # gamma_inv = δ^ij - 2H/(1+2H) l^i l^j
    f = 2 * H / (1 + 2 * H)
    gamma_inv = np.eye(3) - f * np.outer(l, l)

    # dgamma
    dgamma = _compute_dgamma_from_H_l(H, l, dH, dl)

    # Christoffel symbols
    chris = _compute_christoffel_from_dgamma(gamma_inv, dgamma)

    # Extrinsic curvature
    K = _compute_K_from_H_l_chris(H, l, dH, dl, chris)

    # K trace
    K_trace = 0.0
    for i in range(3):
        for j in range(3):
            K_trace += gamma_inv[i, j] * K[i, j]

    return gamma_inv, dgamma, K, K_trace


class FastBoostedKerrMetricCached(Metric):
    """
    Fast boosted Kerr metric with caching.

    Caches all intermediate computations for the last evaluated point,
    eliminating redundant computation when gamma_inv, dgamma, K, K_trace
    are all called for the same point.

    Speedup: ~7x on metric computation by avoiding redundant _compute_all calls.
    """

    def __init__(self, M: float = 1.0, a: float = 0.0, velocity=None):
        self.M = M
        self.a = a

        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0])
        self.velocity = np.asarray(velocity, dtype=np.float64)

        v_mag = np.linalg.norm(self.velocity)
        if v_mag > 1e-14:
            self.lorentz_gamma = 1.0 / np.sqrt(1 - v_mag**2)
            self.n_hat = self.velocity / v_mag
        else:
            self.lorentz_gamma = 1.0
            self.n_hat = np.array([1.0, 0.0, 0.0])

        # Coordinate transformation: x_rest = Lambda @ x_lab
        self.Lambda = np.eye(3) + (self.lorentz_gamma - 1) * np.outer(self.n_hat, self.n_hat)

        # Cache for last computed point
        self._cache_point = None  # (x, y, z) tuple
        self._cache_H = None
        self._cache_l = None
        self._cache_dH = None
        self._cache_dl = None
        self._cache_gamma_inv = None
        self._cache_dgamma = None
        self._cache_chris = None
        self._cache_K = None
        self._cache_K_trace = None

    def _invalidate_cache(self):
        """Clear the cache."""
        self._cache_point = None

    def _ensure_base_computed(self, x: float, y: float, z: float):
        """Ensure H, l, dH, dl are computed for this point."""
        point = (x, y, z)
        if self._cache_point == point:
            return  # Already cached

        # New point - compute everything from scratch
        self._cache_point = point

        # Transform to rest frame
        x_lab = np.array([x, y, z])
        x_rest = self.Lambda @ x_lab

        # Compute H, l in rest frame
        H, l, r = _compute_H_l_rest_frame(x_rest, self.M, self.a)

        # Compute derivatives in rest frame
        dH_rest, dl_rest = _compute_dH_dl_numerical(x_rest, self.M, self.a)

        # Transform derivatives to lab frame
        dH, dl = _transform_derivatives_to_lab(dH_rest, dl_rest, self.Lambda)

        # Store in cache
        self._cache_H = H
        self._cache_l = l
        self._cache_dH = dH
        self._cache_dl = dl

        # Invalidate derived quantities (will be computed on demand)
        self._cache_gamma_inv = None
        self._cache_dgamma = None
        self._cache_chris = None
        self._cache_K = None
        self._cache_K_trace = None

    def _ensure_gamma_inv_computed(self, x: float, y: float, z: float):
        """Ensure gamma_inv is computed."""
        self._ensure_base_computed(x, y, z)
        if self._cache_gamma_inv is None:
            H = self._cache_H
            l = self._cache_l
            f = 2 * H / (1 + 2 * H)
            result = np.eye(3)
            for i in range(3):
                for j in range(3):
                    result[i, j] -= f * l[i] * l[j]
            self._cache_gamma_inv = result

    def _ensure_dgamma_computed(self, x: float, y: float, z: float):
        """Ensure dgamma is computed."""
        self._ensure_base_computed(x, y, z)
        if self._cache_dgamma is None:
            self._cache_dgamma = _compute_dgamma_from_H_l(
                self._cache_H, self._cache_l, self._cache_dH, self._cache_dl
            )

    def _ensure_chris_computed(self, x: float, y: float, z: float):
        """Ensure Christoffel symbols are computed."""
        self._ensure_gamma_inv_computed(x, y, z)
        self._ensure_dgamma_computed(x, y, z)
        if self._cache_chris is None:
            self._cache_chris = _compute_christoffel_from_dgamma(
                self._cache_gamma_inv, self._cache_dgamma
            )

    def _ensure_K_computed(self, x: float, y: float, z: float):
        """Ensure extrinsic curvature is computed."""
        self._ensure_chris_computed(x, y, z)
        if self._cache_K is None:
            self._cache_K = _compute_K_from_H_l_chris(
                self._cache_H, self._cache_l, self._cache_dH, self._cache_dl,
                self._cache_chris
            )

    def _ensure_K_trace_computed(self, x: float, y: float, z: float):
        """Ensure K trace is computed."""
        self._ensure_gamma_inv_computed(x, y, z)
        self._ensure_K_computed(x, y, z)
        if self._cache_K_trace is None:
            self._cache_K_trace = np.sum(self._cache_gamma_inv * self._cache_K)

    # Public interface - same as original

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        self._ensure_base_computed(x, y, z)
        H = self._cache_H
        l = self._cache_l
        result = np.eye(3)
        for i in range(3):
            for j in range(3):
                result[i, j] += 2 * H * l[i] * l[j]
        return result

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        self._ensure_gamma_inv_computed(x, y, z)
        return self._cache_gamma_inv.copy()

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        self._ensure_dgamma_computed(x, y, z)
        return self._cache_dgamma.copy()

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        self._ensure_K_computed(x, y, z)
        return self._cache_K.copy()

    def K_trace(self, x: float, y: float, z: float) -> float:
        self._ensure_K_trace_computed(x, y, z)
        return self._cache_K_trace

    def lapse(self, x: float, y: float, z: float) -> float:
        self._ensure_base_computed(x, y, z)
        return 1.0 / np.sqrt(1 + 2 * self._cache_H)

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        self._ensure_base_computed(x, y, z)
        H = self._cache_H
        l = self._cache_l
        return 2 * H * l / (1 + 2 * H)

    # Convenience method to get all quantities at once
    def compute_all_geometric(self, x: float, y: float, z: float):
        """
        Compute and return all geometric quantities needed for expansion.

        Uses a single JIT function to avoid Python overhead between calls.

        Returns:
            gamma_inv, dgamma, K, K_trace
        """
        x_lab = np.array([x, y, z])
        return _compute_all_geometric_jit(x_lab, self.M, self.a, self.Lambda)
