#!/usr/bin/env python3
"""
Clean summary of profiling results.
"""

# Profile results from FastBoostedKerrMetric (a=0.5, v=0.3), N_s=25
# Total time: 177.8 seconds for 2 full finds (4 iterations each)

profile_data = """
================================================================================
PROFILE SUMMARY: FastBoostedKerrMetric (a=0.5, v=0.3), N_s=25
================================================================================

Total time: 177.8 seconds (2 finds × 4 iterations each)
Total function calls: 307 million

TOP FUNCTIONS BY INTERNAL TIME (tottime):
--------------------------------------------------------------------------------
Rank  Function                              tottime   %     ncalls      Description
--------------------------------------------------------------------------------
1     gamma_inv                             21.4s    12%    8M          Inverse metric computation
2     compute_phi_stencil                   20.1s    11%    2M          Phi-direction stencil points
3     _compute_all                          12.8s    7%     14M         Main metric computation
4     _compute_dH_dl_numerical              11.8s    7%     14M         Numerical H,l derivatives
5     interpolate_batch                     10.4s    6%     2M          Biquartic interpolation
6     _get_rest_coords                      10.0s    6%     14M         Lorentz transform coords
7     scipy RectBivariateSpline.__call__    9.0s     5%     2M          Scipy spline evaluation
8     _transform_derivatives_to_lab         8.3s     5%     14M         Transform derivs to lab
9     evaluate_at_point                     7.8s     4%     2M          Residual at single point
10    numba _numba_unpickle                 7.6s     4%     84M         Numba serialization overhead
11    stencil compute_derivatives           5.9s     3%     2M          Derivative computation
12    numpy.eye                             5.7s     3%     8M          Identity matrix creation
13    _compute_H_l_rest_frame               5.6s     3%     14M         H,l in rest frame
14    numpy.array                           4.6s     3%     18M         Array creation
15    extrinsic_curvature                   4.5s     3%     4M          K_ij computation
--------------------------------------------------------------------------------
                                            ~145s    82%

REMAINING (~18%): numpy operations, memory allocation, etc.

================================================================================
GPU ACCELERATION ANALYSIS
================================================================================

CANDIDATE FUNCTIONS FOR GPU:
--------------------------------------------------------------------------------
Function                    Time    Parallelizable?   GPU Potential    Notes
--------------------------------------------------------------------------------
gamma_inv                   21.4s   Yes (per point)   HIGH             8M independent calls
_compute_all                12.8s   Yes (per point)   HIGH             Main bottleneck
_compute_dH_dl_numerical    11.8s   Yes (per point)   HIGH             6 offset evals
_get_rest_coords            10.0s   Yes (per point)   HIGH             Simple Lorentz transform
_transform_derivatives      8.3s    Yes (per point)   HIGH             Matrix multiply
_compute_H_l_rest_frame     5.6s    Yes (per point)   MEDIUM           Contains sqrt, conditionals
--------------------------------------------------------------------------------
Subtotal (parallelizable)   69.9s   39% of total

FUNCTIONS HARDER TO GPU-IFY:
--------------------------------------------------------------------------------
Function                    Time    Why Hard                          Alternative
--------------------------------------------------------------------------------
compute_phi_stencil         20.1s   Sequential dependency on phi      Restructure algorithm
interpolate_batch           10.4s   Uses scipy splines                Custom CUDA spline
scipy __call__              9.0s    scipy internal                    Custom CUDA spline
--------------------------------------------------------------------------------
Subtotal (hard)             39.5s   22% of total

OVERHEAD (cannot eliminate):
--------------------------------------------------------------------------------
numba _numba_unpickle       7.6s    Numba JIT overhead                Already cached
numpy.eye                   5.7s    Identity matrix creation          Preallocate
numpy.array                 4.6s    Array creation                    Preallocate
--------------------------------------------------------------------------------
Subtotal (overhead)         17.9s   10% of total

================================================================================
ESTIMATED GPU SPEEDUP
================================================================================

Assumptions:
- GPU compute speedup: 20x for parallel math
- Memory transfer overhead: 1ms per batch
- Only parallelize the "HIGH" potential functions

Current "parallelizable" time: 69.9s
GPU estimate: 69.9s / 20 + 0.01s overhead = 3.5s

Current "hard to GPU" time: 39.5s
(stays same without custom implementation)

Current "overhead" time: 17.9s
(mostly stays same, some reduction from batching)

ESTIMATED TOTAL:
  Current:  177.8s
  With GPU: 3.5s + 39.5s + 15s = ~58s
  Speedup:  ~3x

If we also implement custom GPU interpolation:
  GPU estimate: 3.5s + 5s (GPU spline) + 10s = ~18s
  Speedup:  ~10x

================================================================================
RECOMMENDATIONS
================================================================================

1. QUICK WIN - Batch metric evaluation (est. 3x speedup):
   - Move gamma_inv, _compute_all, etc. to process arrays not single points
   - Use Numba with parallel=True or CUDA

2. MEDIUM EFFORT - GPU metric kernel (additional 2-3x):
   - Single CUDA kernel for all metric quantities at all points
   - Eliminates per-call overhead

3. HIGH EFFORT - GPU interpolation (additional 2-3x):
   - Replace scipy RectBivariateSpline with custom CUDA spline
   - Significant implementation effort

4. ALTERNATIVE - CPU batching only (est. 2x speedup):
   - Restructure to process all points together
   - Use numpy vectorization + Numba parallel
   - Lower effort than full GPU implementation
"""

print(profile_data)
