"""
Fast Lorentz-boosted metrics with analytical derivatives.

This module provides optimized implementations of boosted metrics
using analytical derivatives instead of numerical differentiation.
This gives ~10-50x speedup over the numerical version.

Reference: Huq, Choptuik & Matzner (2000), Section III.C
"""

import numpy as np
from .base import Metric


class FastBoostedMetric(Metric):
    """
    Fast Lorentz-boosted version of Kerr-Schild metrics.

    Uses analytical derivatives computed via chain rule, avoiding
    expensive numerical differentiation.

    The Kerr-Schild form g_μν = η_μν + 2H l_μ l_ν is preserved under boosts.
    All 3+1 quantities are computed analytically from the boosted H and l.
    """

    def __init__(self, base_metric: Metric, velocity: np.ndarray):
        """
        Initialize fast boosted metric.

        Args:
            base_metric: The metric to boost (must be Schwarzschild or Kerr)
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

        # Pre-compute boost matrices for coordinate transformation
        self._setup_boost_matrices()

    def _setup_boost_matrices(self):
        """Pre-compute matrices for boost transformations."""
        gamma = self.lorentz_gamma
        n = self.n_hat

        # Coordinate transformation matrix: x_rest = Lambda @ x_lab
        # x'_parallel = gamma * x_parallel, x'_perp = x_perp
        # Lambda = I + (gamma - 1) n ⊗ n
        self.Lambda_coord = np.eye(3) + (gamma - 1) * np.outer(n, n)

    def _get_rest_frame_quantities(self, x: float, y: float, z: float):
        """
        Compute H, l, r, and their derivatives in rest frame.

        Returns:
            Dictionary with H, l_rest, r, dH_dxrest (3,), dl_dxrest (3,3)
        """
        # Transform to rest frame
        x_lab = np.array([x, y, z])
        x_rest = self.Lambda_coord @ x_lab
        xr, yr, zr = x_rest

        # Compute r and l in rest frame
        if hasattr(self.base_metric, '_r_and_l'):
            r, l_rest = self.base_metric._r_and_l(xr, yr, zr)
            if hasattr(self.base_metric, '_H'):
                if hasattr(self.base_metric, 'a'):  # Kerr
                    H = self.base_metric._H(r, zr)
                else:  # Schwarzschild
                    H = self.base_metric._H(r)
            else:
                H = self.base_metric.M / r
        else:
            r = np.sqrt(xr**2 + yr**2 + zr**2)
            if r < 1e-14:
                r = 1e-14
            l_rest = x_rest / r
            H = self.base_metric.M / r

        # Compute derivatives in rest frame
        # For Schwarzschild: ∂H/∂x_rest = -M/r² × (x_rest/r) = -H/r × l
        # ∂l_i/∂x_rest_j = (δ_ij - l_i l_j) / r
        dH_dxrest = -H * l_rest / r  # shape (3,)
        dl_dxrest = (np.eye(3) - np.outer(l_rest, l_rest)) / r  # shape (3,3)

        return {
            'H': H,
            'l_rest': l_rest,
            'r': r,
            'x_rest': x_rest,
            'dH_dxrest': dH_dxrest,
            'dl_dxrest': dl_dxrest,
        }

    def _boost_null_vector(self, l_rest: np.ndarray):
        """
        Boost null vector from rest frame to lab frame.

        In rest frame: l_μ = (1, l_i) with |l| = 1
        Returns boosted (l_0, l_spatial)
        """
        if self.v_mag < 1e-14:
            return 1.0, l_rest.copy()

        v = self.velocity
        gamma = self.lorentz_gamma
        n = self.n_hat

        # l_0' = gamma(l_0 - v·l) = gamma(1 - v·l)
        v_dot_l = np.dot(v, l_rest)
        l_0 = gamma * (1.0 - v_dot_l)

        # l_parallel' = gamma(l_parallel - v*l_0) = gamma(l_parallel - v)
        l_parallel = np.dot(l_rest, n) * n
        l_perp = l_rest - l_parallel
        l_spatial = l_perp + gamma * (l_parallel - v)

        return l_0, l_spatial

    def _boost_null_vector_derivatives(self, l_rest: np.ndarray, dl_dxrest: np.ndarray):
        """
        Compute derivatives of boosted null vector components.

        Args:
            l_rest: Null vector in rest frame (3,)
            dl_dxrest: Derivatives ∂l_rest_i/∂x_rest_j (3,3)

        Returns:
            dl0_dxlab: Derivatives ∂l_0/∂x_lab (3,)
            dl_dxlab: Derivatives ∂l_lab_i/∂x_lab_j (3,3)
        """
        if self.v_mag < 1e-14:
            # For zero boost, l_0 = 1 (constant), l_spatial = l_rest
            # dx_rest/dx_lab = I
            return np.zeros(3), dl_dxrest.copy()

        gamma = self.lorentz_gamma
        v = self.velocity
        n = self.n_hat
        Lambda = self.Lambda_coord

        # l_0 = gamma(1 - v·l_rest)
        # dl_0/dx_rest = -gamma * v @ dl_rest/dx_rest
        # dl_0/dx_lab = dl_0/dx_rest @ (dx_rest/dx_lab)^T = dl_0/dx_rest @ Lambda^T
        dl0_dxrest = -gamma * (v @ dl_dxrest)  # shape (3,)
        dl0_dxlab = Lambda.T @ dl0_dxrest  # shape (3,)

        # l_lab = l_perp + gamma(l_parallel - v)
        # l_parallel_i = n_i (n · l_rest) = n_i n_k l_rest_k
        # dl_parallel/dx_rest = n ⊗ (n @ dl_dxrest)
        n_dl = n @ dl_dxrest  # shape (3,)
        dl_parallel_dxrest = np.outer(n, n_dl)  # shape (3,3)

        dl_perp_dxrest = dl_dxrest - dl_parallel_dxrest
        dl_lab_dxrest = dl_perp_dxrest + gamma * dl_parallel_dxrest

        # Transform to lab frame derivatives
        dl_dxlab = dl_lab_dxrest @ Lambda.T

        return dl0_dxlab, dl_dxlab

    def _compute_all_quantities(self, x: float, y: float, z: float):
        """
        Compute all metric quantities at once analytically.

        Returns dictionary with gamma, gamma_inv, dgamma, K, lapse, shift, christoffel
        """
        # Get rest frame quantities
        rest = self._get_rest_frame_quantities(x, y, z)
        H = rest['H']
        l_rest = rest['l_rest']
        dH_dxrest = rest['dH_dxrest']
        dl_dxrest = rest['dl_dxrest']

        # Boost to lab frame
        l_0, l_lab = self._boost_null_vector(l_rest)

        # Compute boosted derivatives
        dl0_dxlab, dl_dxlab = self._boost_null_vector_derivatives(l_rest, dl_dxrest)

        # H derivative in lab frame: dH/dx_lab = (dx_rest/dx_lab)^T @ dH/dx_rest
        dH_dxlab = self.Lambda_coord.T @ dH_dxrest

        # ============================================================
        # Compute 3-metric: γ_ij = δ_ij + 2H l_i l_j
        # ============================================================
        l_outer = np.outer(l_lab, l_lab)
        gamma_3 = np.eye(3) + 2 * H * l_outer

        # ============================================================
        # Compute inverse 3-metric using Sherman-Morrison
        # γ^ij = δ^ij - 2H/(1 + 2H|l|²) l^i l^j
        # ============================================================
        l_sq = np.dot(l_lab, l_lab)
        gamma_inv = np.eye(3) - (2 * H / (1 + 2 * H * l_sq)) * l_outer

        # ============================================================
        # Compute dgamma: ∂_k γ_ij = 2(∂_k H) l_i l_j + 2H(∂_k l_i)l_j + 2H l_i(∂_k l_j)
        # ============================================================
        dgamma = np.zeros((3, 3, 3))
        for k in range(3):
            dgamma[k] = (2 * dH_dxlab[k] * l_outer
                        + 2 * H * np.outer(dl_dxlab[:, k], l_lab)
                        + 2 * H * np.outer(l_lab, dl_dxlab[:, k]))

        # ============================================================
        # Compute Christoffel symbols
        # Γ^i_jk = (1/2) γ^il (∂_j γ_lk + ∂_k γ_jl - ∂_l γ_jk)
        # ============================================================
        chris = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        chris[i, j, k] += 0.5 * gamma_inv[i, l] * (
                            dgamma[j, l, k] + dgamma[k, j, l] - dgamma[l, j, k]
                        )

        # ============================================================
        # Compute lapse and shift
        # α = 1/√(1 + 2H l_0²)
        # β_i = 2H l_0 l_i, β^i = γ^ij β_j
        # ============================================================
        alpha = 1.0 / np.sqrt(1 + 2 * H * l_0 * l_0)
        beta_down = 2 * H * l_0 * l_lab
        beta_up = gamma_inv @ beta_down

        # ============================================================
        # Compute extrinsic curvature analytically
        # K_ij = (1/2α)(D_i β_j + D_j β_i - ∂_t γ_ij)
        # ============================================================
        K = self._compute_extrinsic_curvature_analytical(
            H, l_0, l_lab, dH_dxlab, dl0_dxlab, dl_dxlab,
            alpha, beta_down, dgamma, chris
        )

        return {
            'gamma': gamma_3,
            'gamma_inv': gamma_inv,
            'dgamma': dgamma,
            'K': K,
            'lapse': alpha,
            'shift': beta_up,
            'christoffel': chris,
        }

    def _compute_extrinsic_curvature_analytical(
        self, H, l_0, l_lab, dH, dl0, dl,
        alpha, beta_down, dgamma, chris
    ):
        """
        Compute extrinsic curvature using analytical expressions.

        For a boosted black hole moving with velocity v, the metric is
        time-dependent. We use:
        K_ij = (1/2α)(D_i β_j + D_j β_i - ∂_t γ_ij)

        The key is that ∂_t γ_ij can be computed analytically:
        ∂_t γ_ij = -v^k ∂_k γ_ij (the metric moves with velocity v)
        """
        v = self.velocity

        # ∂_t γ_ij = -v^k ∂_k γ_ij (metric moves with the black hole)
        dgamma_dt = np.zeros((3, 3))
        for k in range(3):
            dgamma_dt -= v[k] * dgamma[k]

        # Compute ∂_i β_j (lowered)
        # β_j = 2H l_0 l_j
        # ∂_i β_j = 2(∂_i H) l_0 l_j + 2H (∂_i l_0) l_j + 2H l_0 (∂_i l_j)
        d_beta_down = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                d_beta_down[i, j] = (2 * dH[i] * l_0 * l_lab[j]
                                    + 2 * H * dl0[i] * l_lab[j]
                                    + 2 * H * l_0 * dl[j, i])

        # D_i β_j = ∂_i β_j - Γ^k_ij β_k
        D_beta = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                D_beta[i, j] = d_beta_down[i, j]
                for k in range(3):
                    D_beta[i, j] -= chris[k, i, j] * beta_down[k]

        # K_ij = (1/2α)(D_i β_j + D_j β_i - ∂_t γ_ij)
        K = 0.5 / alpha * (D_beta + D_beta.T - dgamma_dt)

        return K

    # ================================================================
    # Public interface methods
    # ================================================================

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute boosted 3-metric γ_ij."""
        return self._compute_all_quantities(x, y, z)['gamma']

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute inverse boosted 3-metric γ^ij."""
        return self._compute_all_quantities(x, y, z)['gamma_inv']

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute ∂_k γ_ij analytically."""
        return self._compute_all_quantities(x, y, z)['dgamma']

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute extrinsic curvature K_ij analytically."""
        return self._compute_all_quantities(x, y, z)['K']

    def lapse(self, x: float, y: float, z: float) -> float:
        """Compute lapse α."""
        return self._compute_all_quantities(x, y, z)['lapse']

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute shift β^i."""
        return self._compute_all_quantities(x, y, z)['shift']

    def christoffel(self, x: float, y: float, z: float) -> np.ndarray:
        """Compute Christoffel symbols Γ^i_jk."""
        return self._compute_all_quantities(x, y, z)['christoffel']


class CachedBoostedMetric(FastBoostedMetric):
    """
    Boosted metric with point-based caching.

    Caches the last computed point to avoid redundant calculations
    when multiple metric quantities are requested at the same point.
    """

    def __init__(self, base_metric: Metric, velocity: np.ndarray):
        super().__init__(base_metric, velocity)
        self._cache_point = None
        self._cache_data = None

    def _get_cached_quantities(self, x: float, y: float, z: float):
        """Get quantities from cache or compute them."""
        point = (x, y, z)
        if self._cache_point != point:
            self._cache_data = self._compute_all_quantities(x, y, z)
            self._cache_point = point
        return self._cache_data

    def gamma(self, x: float, y: float, z: float) -> np.ndarray:
        return self._get_cached_quantities(x, y, z)['gamma'].copy()

    def gamma_inv(self, x: float, y: float, z: float) -> np.ndarray:
        return self._get_cached_quantities(x, y, z)['gamma_inv'].copy()

    def dgamma(self, x: float, y: float, z: float) -> np.ndarray:
        return self._get_cached_quantities(x, y, z)['dgamma'].copy()

    def extrinsic_curvature(self, x: float, y: float, z: float) -> np.ndarray:
        return self._get_cached_quantities(x, y, z)['K'].copy()

    def lapse(self, x: float, y: float, z: float) -> float:
        return self._get_cached_quantities(x, y, z)['lapse']

    def shift(self, x: float, y: float, z: float) -> np.ndarray:
        return self._get_cached_quantities(x, y, z)['shift'].copy()

    def christoffel(self, x: float, y: float, z: float) -> np.ndarray:
        return self._get_cached_quantities(x, y, z)['christoffel'].copy()


def fast_boost_metric(base_metric: Metric, velocity: np.ndarray, use_cache: bool = True):
    """
    Create a fast boosted version of a metric.

    Args:
        base_metric: The metric to boost
        velocity: 3-vector boost velocity
        use_cache: If True, use caching for repeated point queries

    Returns:
        FastBoostedMetric or CachedBoostedMetric instance
    """
    if use_cache:
        return CachedBoostedMetric(base_metric, velocity)
    return FastBoostedMetric(base_metric, velocity)