# Journal

This file records prompts and responses from Claude sessions to maintain context across conversations.

---

## Session: 2026-02-01 - Initial Implementation

### Prompt 1: Implement the Apparent Horizon Finder

**User Request**: Implement the apparent horizon location algorithm from Huq, Choptuik & Matzner (2000) - arXiv:gr-qc/0002076, following a detailed implementation plan.

**Actions Taken**:
1. Created project structure:
   ```
   AHFinder/
   ├── src/ahfinder/
   │   ├── __init__.py
   │   ├── surface.py
   │   ├── stencil.py
   │   ├── interpolation.py
   │   ├── residual.py
   │   ├── jacobian.py
   │   ├── solver.py
   │   ├── finder.py
   │   ├── visualization.py
   │   └── metrics/
   │       ├── __init__.py
   │       ├── base.py
   │       ├── schwarzschild.py
   │       ├── kerr.py
   │       ├── boosted.py
   │       └── numerical.py
   ├── tests/
   ├── examples/
   ├── doc/
   └── pyproject.toml
   ```

2. Implemented all core modules following the plan

**Initial Test Result**: Newton solver failed to converge
```
Iter   0: ||F|| = 1.091853e+01, ||δρ|| = 2.926390e+01
...
Iter  19: ||F|| = 4.072693e+01, ||δρ|| = 1.339930e+02
WARNING: Failed to converge!
```

---

### Prompt 2: Run Battery of Tests

**User Request**: Run a battery of tests and create doc/ImplementationTests.md with all numerical tests to verify implementation. Put together a test plan and review.

**Actions Taken**:
1. Created comprehensive test suite covering:
   - Surface mesh properties
   - Interpolation accuracy
   - Cartesian stencil derivatives
   - Schwarzschild metric components
   - Extrinsic curvature
   - Christoffel symbols

**Test Results Summary**:

| Component | Status |
|-----------|--------|
| Surface Mesh | ✓ PASS (6/6 tests) |
| Interpolation | ✓ PASS (3/3 tests) |
| Cartesian Stencil | ✓ PASS (2/2 tests) |
| Schwarzschild Metric | ✓ PASS (5/5 tests) |
| **Expansion Formula** | **❌ FAIL** |

**Critical Issue Identified**: The expansion Θ did not vanish at r = 2M (Schwarzschild horizon). Instead of Θ ≈ 0, got Θ ≈ +1.32.

**Expansion Test Results (before fix)**:
| r | Θ | Expected |
|---|---|----------|
| 1.5 | +1.91 | negative |
| 2.0 | +1.32 | zero |
| 2.5 | +1.01 | positive |

**Root Cause**: The divergence calculation was using coordinate derivatives instead of covariant derivatives.

**Output**: Created `doc/ImplementationTests.md` with full test documentation.

---

### Prompt 3: Fix the Expansion Formula

**User Request**: Fix the expansion formula.

**The Correction Made**:

**Original (incorrect)** - Using coordinate Laplacian:
```python
laplacian_phi = γ^{ij} ∂_i ∂_j φ           # Missing Christoffel correction
proj_term = (n^i n^j / ω) ∂_i ∂_j φ        # Missing Christoffel correction
```

**Corrected** - Using covariant Laplacian:
```python
# Covariant Laplacian: Δφ = γ^{ij} ∇_i ∇_j φ = γ^{ij} ∂_i ∂_j φ - Γ^k ∂_k φ
laplacian = coord_laplacian - np.dot(Gamma_up, grad_phi)

# Covariant projection: (n^i n^j / ω) ∇_i ∇_j φ
n_n_chris = np.einsum('i,j,kij->k', n_up, n_up, chris)
chris_proj = np.dot(n_n_chris, grad_phi) / omega
proj_term = coord_proj - chris_proj
```

**Expansion Test Results (after fix)**:
| r | Θ | Expected |
|---|---|----------|
| 1.50 | -0.293 | Θ < 0 (trapped) ✓ |
| 1.80 | -0.086 | Θ < 0 (trapped) ✓ |
| 2.00 | -0.001 | Θ ≈ 0 (horizon) ✓ |
| 2.10 | +0.032 | Θ > 0 (untrapped) ✓ |
| 2.50 | +0.119 | Θ > 0 (untrapped) ✓ |

**Newton Solver Test (N_s=9)**:
```
Newton iteration for apparent horizon:
  N_s = 9, tol = 0.0001
--------------------------------------------------
  Iter   0: ||F|| = 1.013730e-02, ||δρ|| = 1.070744e-05 (α=0.001)
--------------------------------------------------
Converged in 1 iterations

Results:
  Mean radius: 2.000001 (expected: 2.0)
  Error: 0.000%
  Status: PASSED
```

---

### Prompt 4: Comparison with Paper Formula

**User Question**: What was the correction you made? Does the expression match back to the paper?

**Response**:

The correction was adding Christoffel symbol terms to convert coordinate derivatives to covariant derivatives:
- Δφ = γ^{ij} ∂_i ∂_j φ - Γ^k ∂_k φ (covariant Laplacian)
- ∇_i ∇_j φ = ∂_i ∂_j φ - Γ^k_{ij} ∂_k φ (covariant Hessian)

**Comparison with Paper (Eq. 5)**:

The paper uses metric derivatives ∂_a γ^{bc} explicitly:
```
F[ρ] = γ^{ab} ∂_a ∂_b φ + γ^{ab}_{,a} ∂_b φ
     - (1/2) ω^{-1} γ^{ab} γ^{cd}_{,a} ∂_b φ ∂_c φ ∂_d φ
     - ω^{-1} γ^{ab} γ^{cd} ∂_b φ ∂_a ∂_c φ ∂_d φ
     + Γ^a_{ab} γ^{bc} ∂_c φ
     + ω^{-1/2} K_{ab} γ^{ac} γ^{bd} ∂_c φ ∂_d φ
     - ω^{1/2} K
```

My implementation uses Christoffel symbols Γ^k_{ij} computed from metric derivatives. These are **mathematically equivalent** formulations - both compute the expansion of outgoing null normals.

---

### Prompt 5: Measure and Improve Performance

**User Request**: Measure the performance and see if you can improve it.

**Performance Profiling Results**:

| N_s | Points | Stencil Time | Metric Time | Full Residual | Est. Jacobian |
|-----|--------|--------------|-------------|---------------|---------------|
| 9   | 65     | 0.702 ms/pt  | 0.044 ms/pt | 0.045 s       | 2.9 s         |
| 17  | 257    | 0.608 ms/pt  | 0.040 ms/pt | 0.174 s       | 44.7 s        |
| 33  | 1025   | 0.601 ms/pt  | 0.039 ms/pt | 0.692 s       | 708.9 s       |

**Bottleneck Identified**: Stencil computation (27-point interpolation) takes ~90% of time.

**Interpolation Method Comparison**:
| Method | Time for 1000 points | Per-point |
|--------|---------------------|-----------|
| Custom Biquartic | 0.017 s | 0.017 ms |
| SciPy Cubic | 0.001 s | 0.001 ms |

**Speedup potential**: 16.6x using SciPy's RegularGridInterpolator

**Optimization Actions Started**:
1. Vectorized stencil offset computation
2. Added batch interpolation method
3. Testing SciPy interpolator as faster alternative

---

## Files Created This Session

1. `pyproject.toml` - Package configuration
2. `src/ahfinder/__init__.py` - Package init
3. `src/ahfinder/surface.py` - Surface mesh management
4. `src/ahfinder/interpolation.py` - Biquartic interpolation
5. `src/ahfinder/stencil.py` - 27-point Cartesian stencil
6. `src/ahfinder/metrics/base.py` - Abstract metric interface
7. `src/ahfinder/metrics/schwarzschild.py` - Schwarzschild metric
8. `src/ahfinder/metrics/kerr.py` - Kerr metric
9. `src/ahfinder/metrics/boosted.py` - Lorentz-boosted metrics
10. `src/ahfinder/metrics/numerical.py` - Numerical grid metrics
11. `src/ahfinder/residual.py` - Expansion Θ evaluation
12. `src/ahfinder/jacobian.py` - Numerical Jacobian
13. `src/ahfinder/solver.py` - Newton iteration with line search
14. `src/ahfinder/finder.py` - High-level API
15. `src/ahfinder/visualization.py` - Plotting utilities
16. `tests/test_surface.py` - Surface mesh tests
17. `tests/test_interpolation.py` - Interpolation tests
18. `tests/test_residual.py` - Residual tests
19. `tests/test_schwarzschild.py` - Schwarzschild tests
20. `tests/test_kerr.py` - Kerr tests
21. `tests/test_boosted.py` - Boosted metric tests
22. `examples/find_horizon.py` - Example usage
23. `examples/visualize_horizon.py` - Visualization examples
24. `doc/algorithm.md` - Algorithm documentation
25. `doc/ImplementationTests.md` - Test results documentation

---

### Prompt 6: Performance Optimization (continued)

**User Request**: Continue with performance optimization using SciPy-based interpolation.

**Actions Taken**:
1. Implemented `FastInterpolator` class using SciPy's `RectBivariateSpline` with quintic (5th order) interpolation
2. Modified `CartesianStencil.compute_phi_stencil()` to use batch interpolation
3. Added `use_fast_interpolator` option to `NewtonSolver` and `ApparentHorizonFinder`

**Key Finding**: Different interpolation methods produce different Jacobians, but both converge to correct solutions as long as the method is used consistently for both residual and Jacobian evaluation.

**Performance Results**:

| N_s | Fast (s) | Biquartic (s) | Speedup | Fast radius | Biq radius |
|-----|----------|---------------|---------|-------------|------------|
| 9   | 0.71     | 1.74          | 2.4x    | 2.000223    | 2.000001   |
| 17  | 1.67     | 7.73          | 4.6x    | 2.000000    | 2.000000   |
| 25  | 10.75    | 18.08         | 1.7x    | 1.999990    | 2.000000   |
| 33  | 6.22     | 32.19         | 5.2x    | 2.000000    | 2.000000   |

**Conclusion**: Fast interpolator (quintic spline) provides 2-5x speedup with equivalent accuracy for practical tolerances (tol=1e-4). Set as default.

---

---

### Prompt 7: Documentation with Graphs and Truncation Error Analysis

**User Request**: Document the tests with graphs and investigate convergence behavior.

**Actions Taken**:
1. Created `doc/graphs/` folder with visualization graphs
2. Investigated Newton solver convergence behavior
3. Analyzed truncation error distribution

**Key Finding - Truncation Error**:

The residual ||F|| at the exact solution (r=2M) does not vanish - it equals the truncation error from finite difference discretization. The Newton solver converges in 1 iteration because it immediately reaches this truncation error level.

**Truncation Error Analysis**:
- Pole points (θ=0°, 180°): Larger absolute error (~8e-4)
- Interior points (θ≈45°): Smaller absolute error (~7e-5)
- Both regions show **O(h²) convergence** with mesh refinement

**Convergence Rate Verification**:
| Region | Measured Order |
|--------|----------------|
| Pole (θ=0°) | 2.00 |
| Interior (θ≈45°) | 2.01 |

This confirms the second-order accuracy of the Cartesian finite difference scheme.

**Graphs Created**:
- `expansion_profile.png` - Θ vs radius
- `newton_convergence.png` - Solver convergence metrics
- `horizon_surface_3d.png` - 3D horizon visualization
- `horizon_cross_section.png` - Equatorial cross-section
- `interpolation_accuracy.png` - Interpolator comparison
- `mesh_convergence.png` - Solution accuracy vs resolution
- `performance_comparison.png` - Fast vs Biquartic timing
- `residual_distribution.png` - Truncation error by θ position
- `truncation_error_convergence.png` - O(h²) verification

---

## Session: 2026-02-03 - Jacobian Bug Fix

### Prompt: Test Initial Condition Sensitivity

**User Request**: Check if the Newton solver converges from different initial guesses.

**Problem Identified**: The solver only converged when starting very close to the solution (r₀ ≈ 2.0). Starting from r₀ = 2.5 or further caused divergence.

**Debugging Process**:

1. **Removed line search** - Made Newton iteration match the paper's pseudocode exactly
2. **Tested basin of attraction** - Found convergence only for r₀ ≤ 2.0
3. **Analyzed Newton step direction** - Steps were in correct direction but wrong magnitude
4. **Checked Jacobian condition number** - High but not singular
5. **Verified linear solve** - Both GMRES and numpy.linalg.solve gave identical results
6. **Examined Jacobian structure** - Found row sums were negative when they should be positive

**Key Diagnostic**:
```
Uniform ρ increase test at r=2.5:
  Actual dF/dr:        mean = +0.15  (correct - Θ increases with r)
  Jacobian row sums:   mean = -0.39  (WRONG - should match dF/dr)
```

**Root Cause Found**: The sparse Jacobian computation in `_affected_points()` used a radius of 3 in angular space, missing important couplings:

```
Perturbing point (θ=4, φ=2):
  'Affected' points:     49
  Actually non-zero:     65  (ALL points!)

Missed entries included:
  - North pole (μ=0): value = 15.7  ← HUGE contribution dropped!
  - Various φ=6,7 points due to periodic boundary
```

**The Bug**: The sparse Jacobian assumed local coupling in angular space, but:
1. The interpolation stencil creates longer-range couplings
2. Poles couple to all points through the spherical geometry
3. Periodic φ boundary creates additional couplings

**The Fix**: Changed solver to use dense Jacobian computation:
```python
# Before (buggy):
J = self.jacobian_computer.compute_sparse(rho)
delta_rho_flat = self._solve_linear_system(J, -F)

# After (correct):
J = self.jacobian_computer.compute_dense(rho)
delta_rho_flat = np.linalg.solve(J, -F)
```

**Results After Fix**:

| Initial r₀ | Before Fix | After Fix |
|------------|------------|-----------|
| 1.0 | FAIL | ✓ 6 iter |
| 1.5 | ✓ (barely) | ✓ 5 iter |
| 2.0 | ✓ 1 iter | ✓ 2 iter |
| 2.5 | FAIL | ✓ 5 iter |
| 3.0 | FAIL | ✓ 9 iter |
| 4.0 | FAIL | FAIL (too far) |

Basin of attraction expanded from r₀ ∈ [1.9, 2.0] to r₀ ∈ [1.0, 3.0].

---

### Lesson Learned: Test Each Computation Independently

**The existing tests passed but missed this bug because**:
1. Test 4.6 (Jacobian Validation) only verified Taylor expansion accuracy at a single point
2. Test 6.1 (Newton Solver) started at r₀ = 2.0 (already at the solution)
3. No test verified that Jacobian row sums matched the actual dF/dr

**What should have been tested**:
```python
# Test: Jacobian row sums should equal dF/dr for uniform perturbation
rho1 = create_sphere(r)
rho2 = create_sphere(r + eps)
dF_dr = (F(rho2) - F(rho1)) / eps

J = compute_jacobian(rho1)
row_sums = J.sum(axis=1)

assert np.allclose(row_sums, dF_dr)  # This would have caught the bug!
```

**Principle**: Every mathematical identity that the code relies on should have a corresponding test. If J·δρ = -F and we expect uniform δρ for uniform F, then row_sums(J) must be consistent with dF/dr.

---

---

## Session: 2026-02-04 - Kerr and Boosted Metric Testing

### Prompt: Test Boosted Kerr Metric

**User Request**: Test the boosted Kerr metric implementation.

### Bugs Found and Fixed

**1. Extrinsic Curvature Sign Error (Kerr and Boosted)**

Both Kerr and Boosted metrics had wrong sign in extrinsic curvature formula:
```python
# Before (wrong):
K = -0.5 / alpha * (D_beta + D_beta.T)

# After (correct):
K = 0.5 / alpha * (D_beta + D_beta.T)
```

This was discovered by comparing Kerr(a=0) with Schwarzschild:
- Schwarzschild K_xx = -0.53 (correct, from analytical formula)
- Kerr(a=0) K_xx = +0.53 (wrong, opposite sign)

**2. Finder API Mismatch**

`finder.py` was passing `linear_solver` and `jacobian_type` arguments that `NewtonSolver` no longer accepts. Fixed by removing these deprecated arguments.

**3. Boost Transformation Rewrite**

Rewrote the boost transformation to properly handle the 4-metric decomposition:
- Added `_boost_4vector()` for correct Lorentz transformation of null vectors
- Added `_get_4metric_components()` to compute g_00, g_0i, g_ij from boosted H and l
- Updated `lapse()` and `shift()` to use proper formulas: α² = 1/(1 + 2H l_0²)

### Test Results

**Unboosted Kerr (a=0.5)**:
- Converges in 4 iterations
- Mean radius 1.897 (expected r_+ = 1.866)
- Area 46.4 (expected 46.9)
- Status: WORKING

**Boosted Schwarzschild (v=0.3)**:
| N_s | Unboosted Area | Boosted Area | Ratio | x/y Ratio |
|-----|----------------|--------------|-------|-----------|
| 9   | 49.67          | 56.25        | 1.13  | 0.932     |
| 13  | 50.00          | 56.64        | 1.13  | 0.937     |
| 17  | 50.12          | 56.79        | 1.13  | 0.939     |

Expected: Area ratio ~1.0 (invariant), x/y ratio = 0.954

### Root Cause and Fix: Boosted Area Not Invariant

**Problem**: Boosted area was ~13% larger than unboosted.

**Root Cause**: The boosted metric is NOT stationary in the lab frame (black hole is moving). The extrinsic curvature formula was missing the time derivative term.

**Wrong formula** (assumes stationary):
```python
K_ij = (1/2α)(D_i β_j + D_j β_i)
```

**Correct formula** (for moving black hole):
```python
K_ij = (1/2α)(D_i β_j + D_j β_i - ∂_t γ_ij)
```

The time derivative ∂_t γ_ij is non-zero because the black hole center moves from (0,0,0) at t=0 to (v*t, 0, 0) at time t.

**Fix implemented**: Added `_dgamma_dt()` method to compute ∂_t γ_ij numerically.

**Results after fix**:

| Metric | Area Ratio | x/y Ratio | Expected x/y |
|--------|------------|-----------|--------------|
| Boosted Schwarzschild (N_s=9) | 0.9999 | 0.943 | 0.954 |
| Boosted Schwarzschild (N_s=13) | 0.9998 | 0.949 | 0.954 |
| Boosted Kerr (a=0.5, N_s=9) | 0.9996 | 0.943 | 0.954 |

Area invariance restored! x/y ratio converges toward expected value with resolution.

### Performance Profiling

**Top 3 bottlenecks for boosted metrics**:
1. `extrinsic_curvature()`: 0.3 ms/call (75% of metric ops)
2. `christoffel()`: 0.1 ms/call (called inside K)
3. `dgamma()`: 0.1 ms/call

**Jacobian dominates**: O(n_points²) complexity
- N_s=9: Jacobian takes 2.7s
- N_s=17: Jacobian takes ~45s (estimated)

---

## Current Status

- **Core algorithm**: Working correctly with O(h²) convergence
- **Schwarzschild test**: PASSING (finds r = 2M)
- **Kerr test**: PASSING (finds correct horizon after K sign fix)
- **Boosted Schwarzschild**: PASSING (area ratio ~0.9999, Lorentz contraction observed)
- **Boosted Kerr**: PASSING (area ratio ~0.9996, converges in 4 iterations)
- **Initial condition sensitivity**: FIXED - converges from r₀ ∈ [1.0, 3.0]
- **Jacobian**: Uses dense computation
- **Performance**: Boosted metrics slower due to time derivative computation
- **Next steps**:
  - Optimize boosted metric performance (cache computations)
  - Add higher-resolution validation tests
