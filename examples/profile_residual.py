"""
Profile residual evaluation to understand bottlenecks for vectorization.
"""

import numpy as np
import time
from ahfinder.metrics.schwarzschild_fast import SchwarzschildMetricFast
from ahfinder.surface import SurfaceMesh
from ahfinder.jacobian_sparse import create_sparse_residual_evaluator


def profile_residual_breakdown(N_s: int = 17):
    """Profile the components of residual evaluation."""
    print(f"\nProfiling residual evaluation for N_s={N_s}")
    print("=" * 60)

    M = 1.0
    r_guess = 2 * M

    metric = SchwarzschildMetricFast(M=M)
    mesh = SurfaceMesh(N_s=N_s)

    sparse_residual = create_sparse_residual_evaluator(
        mesh, metric, center=(0.0, 0.0, 0.0), spacing_factor=0.5
    )

    rho = np.full((N_s, 2*N_s - 1), r_guess)
    indices = mesh.independent_indices()
    n = len(indices)

    print(f"Grid: {N_s}x{2*N_s-1}, Independent DOFs: {n}")

    # Time full residual evaluation
    t0 = time.time()
    F = sparse_residual.evaluate(rho)
    t_total = time.time() - t0
    print(f"\nFull residual evaluation: {t_total:.4f}s ({n} points)")
    print(f"  Per-point average: {1000*t_total/n:.3f}ms")

    # Break down per-point evaluation
    cx, cy, cz = sparse_residual.center

    # Time component by component
    times = {
        'coords': 0,
        'stencil': 0,
        'metric': 0,
        'expansion': 0
    }

    for i_th, i_ph in indices:
        theta = mesh.theta[i_th]
        phi = mesh.phi[i_ph]
        r = rho[i_th, i_ph]

        # 1. Coordinate computation
        t0 = time.time()
        x0 = cx + r * np.sin(theta) * np.cos(phi)
        y0 = cy + r * np.sin(theta) * np.sin(phi)
        z0 = cz + r * np.cos(theta)
        times['coords'] += time.time() - t0

        # 2. Stencil computation (interpolation)
        t0 = time.time()
        grad_phi, hess_phi = sparse_residual.stencil.compute_all_derivatives(
            rho, x0, y0, z0, sparse_residual.center
        )
        times['stencil'] += time.time() - t0

        # 3. Metric evaluation
        t0 = time.time()
        gamma_inv = metric.gamma_inv(x0, y0, z0)
        dgamma = metric.dgamma(x0, y0, z0)
        K_tensor = metric.extrinsic_curvature(x0, y0, z0)
        K_trace = metric.K_trace(x0, y0, z0)
        times['metric'] += time.time() - t0

        # 4. Expansion computation
        t0 = time.time()
        _ = sparse_residual._compute_expansion(
            grad_phi, hess_phi, gamma_inv, dgamma, K_tensor, K_trace
        )
        times['expansion'] += time.time() - t0

    total_timed = sum(times.values())
    print(f"\nBreakdown ({1000*total_timed:.2f}ms total):")
    for name, t in times.items():
        pct = 100 * t / total_timed if total_timed > 0 else 0
        print(f"  {name:12s}: {1000*t:.3f}ms ({pct:.1f}%)")

    # Count stencil points
    n_stencil_pts = n * 27
    print(f"\nTotal stencil evaluations: {n_stencil_pts}")
    print(f"  Per stencil point (interp): {1e6*times['stencil']/n_stencil_pts:.1f}µs")

    return times


if __name__ == "__main__":
    for N_s in [13, 17, 21]:
        profile_residual_breakdown(N_s)
