# Apparent Horizon Finding Algorithm

This document describes the algorithm implemented in AHFinder, based on:

> Huq, M.F., Choptuik, M.W., & Matzner, R.A. (2000). "Locating Boosted Kerr and Schwarzschild Apparent Horizons." arXiv:gr-qc/0002076

## Physical Background

### Apparent Horizons

An **apparent horizon** is the outermost marginally outer trapped surface (MOTS) on a spacelike hypersurface. It is defined as a closed 2-surface where the expansion of outgoing null normals vanishes:

$$\Theta = D_i s^i + K_{ij} s^i s^j - K = 0$$

where:
- $D_i$ is the covariant derivative compatible with the 3-metric $\gamma_{ij}$
- $s^i$ is the outward-pointing unit normal to the surface
- $K_{ij}$ is the extrinsic curvature of the spatial hypersurface
- $K = \gamma^{ij} K_{ij}$ is the trace of the extrinsic curvature

### Surface Parameterization

The surface is parameterized in spherical coordinates as:
$$r = \rho(\theta, \phi)$$

where $(\theta, \phi)$ cover the 2-sphere and $\rho$ is the radial coordinate function to be determined.

## Algorithm Overview

### Key Innovation: Cartesian Finite Differences

The main innovation of the Huq-Choptuik-Matzner algorithm is using Cartesian finite differences to evaluate derivatives, even though the surface is parameterized in spherical coordinates. This avoids coordinate singularities at the poles ($\theta = 0, \pi$).

### Level Set Function

We define a level set function:
$$\phi(x, y, z) = r - \rho(\theta, \phi)$$

The surface is the zero level set $\phi = 0$. The outward normal is:
$$s^i = \frac{\gamma^{ij} \partial_j \phi}{\sqrt{\omega}}$$

where $\omega = \gamma^{ij} \partial_i \phi \partial_j \phi$.

### Residual Equation

The expansion equation becomes a nonlinear elliptic PDE for $\rho$:

$$F[\rho] = \gamma^{ab} \partial_a\partial_b \phi + \gamma^{ab}_{,a} \partial_b \phi
         - \frac{1}{2}\omega^{-1} \gamma^{ab} \gamma^{cd}_{,a} \partial_b \phi \partial_c \phi \partial_d \phi
         - \omega^{-1} \gamma^{ab} \gamma^{cd} \partial_b \phi \partial_a\partial_c \phi \partial_d \phi
         + \Gamma^a_{ab} \gamma^{bc} \partial_c \phi
         + \omega^{-1/2} K_{ab} \gamma^{ac} \gamma^{bd} \partial_c \phi \partial_d \phi
         - \omega^{1/2} K = 0$$

## Implementation Details

### 1. Surface Mesh

The $(\theta, \phi)$ domain is discretized on an $N_s \times N_s$ grid:
- $\theta \in [0, \pi]$ with $N_s$ points (including poles)
- $\phi \in [0, 2\pi)$ with $N_s$ points (periodic)

**Pole Handling**: At $\theta = 0$ and $\theta = \pi$, all $\phi$ values correspond to the same physical point. The number of independent unknowns is:
$$N_{indep} = N_s^2 - 2(N_s - 1) = N_s^2 - 2N_s + 2$$

### 2. Biquartic Interpolation

To evaluate $\rho$ at arbitrary $(\theta, \phi)$, we use biquartic (4th order) Lagrange interpolation on a 4×4 stencil, giving $O(h^4)$ accuracy.

### 3. Cartesian Stencil

At each surface point $(x_0, y_0, z_0)$:
1. Create a 3×3×3 cube of points with spacing $h \propto d\theta$
2. At each stencil point, compute $\phi = r - \rho(\theta, \phi)$ via interpolation
3. Use standard finite differences to compute:
   - First derivatives: $\partial \phi / \partial x^i$
   - Second derivatives: $\partial^2 \phi / \partial x^i \partial x^j$

This gives $O(h^2)$ accurate derivatives.

### 4. Newton Iteration

The nonlinear equation $F[\rho] = 0$ is solved by Newton iteration:
$$\rho^{(n+1)} = \rho^{(n)} - J^{-1} F[\rho^{(n)}]$$

where the Jacobian is computed numerically:
$$J_{\mu\nu} = \frac{1}{\epsilon} \left[ F_\mu[\rho + \epsilon e_\nu] - F_\mu[\rho] \right]$$

### 5. Jacobian Computation

**Important Implementation Note**: The Jacobian must be computed as a **dense matrix**, not sparse.

While the Cartesian stencil is local in $(x, y, z)$ space, the coupling in $(\theta, \phi)$ space is **global** due to:

1. **Pole coupling**: The poles ($\theta = 0, \pi$) are single physical points that couple to all angular positions through the interpolation and stencil computations.

2. **Interpolation stencil**: The biquartic interpolator uses a 4×4 stencil that creates longer-range couplings than the 3×3×3 Cartesian stencil might suggest.

3. **Periodic boundary**: Points near $\phi = 0$ and $\phi = 2\pi$ are coupled through the periodic boundary.

A sparse approximation that only considers local coupling in angular space will miss these couplings and produce incorrect Newton directions.

### 6. Linear Solver

The linear system $J \delta\rho = -F$ is solved using direct dense factorization (numpy.linalg.solve). For typical resolutions ($N_s = 9$ to $33$), the matrix size ($N_{indep} \times N_{indep}$) is small enough that dense solve is efficient.

### 7. Jacobian-Free Newton-Krylov (Optional)

For larger problems, a Jacobian-Free Newton-Krylov (JFNK) solver is available. Instead of computing the full Jacobian matrix, JFNK uses matrix-vector products computed via finite differences:

$$J \mathbf{v} \approx \frac{F[\rho + \epsilon \mathbf{v}] - F[\rho]}{\epsilon}$$

The linear system is solved iteratively using GMRES. To accelerate convergence, a lagged Jacobian preconditioner is used:
- On the first Newton iteration, compute the full Jacobian and its LU factorization
- Use this as a preconditioner for GMRES in subsequent iterations

**Epsilon Scaling**: Following Knoll & Keyes (2004), the optimal perturbation is:
$$\epsilon = \sqrt{\epsilon_{machine}} \cdot \frac{1 + \|\rho\|}{\|\mathbf{v}\|}$$

This balances truncation error (favors larger ε) against roundoff error (favors smaller ε).

**Performance Note**: The JFNK implementation includes a workaround for a subtle caching bug that requires re-evaluating $F[\rho]$ in each matvec call. This doubles the cost per GMRES iteration. For typical problem sizes ($N_s \leq 33$), the dense Jacobian approach is recommended.

## Kerr-Schild Metrics

The algorithm is designed to work with Kerr-Schild metrics:
$$g_{\mu\nu} = \eta_{\mu\nu} + 2H l_\mu l_\nu$$

### Schwarzschild
$$H = \frac{M}{r}, \quad l_\mu = \left(1, \frac{x}{r}, \frac{y}{r}, \frac{z}{r}\right)$$

Horizon at $r = 2M$.

### Kerr
$$H = \frac{Mr^3}{r^4 + a^2 z^2}, \quad l_\mu = \left(1, \frac{rx + ay}{r^2 + a^2}, \frac{ry - ax}{r^2 + a^2}, \frac{z}{r}\right)$$

where $r$ satisfies $x^2 + y^2 + z^2 = r^2 + a^2(1 - z^2/r^2)$.

Horizon at $r_+ = M + \sqrt{M^2 - a^2}$.

### Boosted Metrics

The Kerr-Schild form is preserved under Lorentz boosts. Under a boost with velocity $\vec{v}$:
- The horizon becomes Lorentz contracted in the boost direction
- The horizon area is an invariant

**Important**: A boosted black hole is **not stationary** in the lab frame—it's moving. The extrinsic curvature formula must include the time derivative of the 3-metric:

$$K_{ij} = \frac{1}{2\alpha}\left(D_i \beta_j + D_j \beta_i - \partial_t \gamma_{ij}\right)$$

Since the black hole center moves with velocity $\vec{v}$, the time derivative can be computed analytically:

$$\partial_t \gamma_{ij} = -v^k \partial_k \gamma_{ij}$$

### Fast Boosted Metric Implementation

The `FastBoostedMetric` class provides optimized boosted metric computations using analytical derivatives instead of numerical differentiation, achieving ~5x speedup.

**Key optimizations**:

1. **Analytical derivatives via chain rule**: All derivatives are computed through the boost transformation:
   - $\partial_k H$ in lab frame from rest frame derivatives
   - $\partial_k l_i$ from boosted null vector derivatives
   - $\partial_k \gamma_{ij} = 2(\partial_k H) l_i l_j + 2H(\partial_k l_i)l_j + 2H l_i(\partial_k l_j)$

2. **Analytical time derivative**: Uses $\partial_t \gamma_{ij} = -v^k \partial_k \gamma_{ij}$ (exact for moving black hole)

3. **Point caching**: `CachedBoostedMetric` caches all quantities at the most recently computed point, avoiding redundant calculations when multiple metric quantities are needed at the same location

## Convergence Properties

- **Interpolation**: $O(h^4)$
- **Finite differences**: $O(h^2)$
- **Overall**: $O(h^2)$ convergence in the solution

Typical parameters:
- $N_s$: 33-97
- $\epsilon$: $10^{-4}$ to $10^{-6}$
- Convergence tolerance: $\|\delta\rho\|_2 < 10^{-9}$
- Iterations: 5-15

## References

1. Huq, M.F., Choptuik, M.W., & Matzner, R.A. (2000). "Locating Boosted Kerr and Schwarzschild Apparent Horizons." Physical Review D, 66, 084024. arXiv:gr-qc/0002076

2. Thornburg, J. (1996). "Finding apparent horizons in numerical relativity." Physical Review D, 54, 4899-4918.

3. Alcubierre, M. (2008). "Introduction to 3+1 Numerical Relativity." Oxford University Press.
