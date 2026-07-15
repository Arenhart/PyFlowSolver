"""Enhanced vs original Arns model on the circular duct.

Two questions:
  1. How well does each Arns/Darcy velocity match the analytic Hagen-Poiseuille
     parabola w(rho) ~ (R^2 - rho^2)?  (the enhanced model targets this shape;
     the original gives ~(R - rho)^2, right endpoints but convex curvature.)
  2. How well does each work as the initial velocity guess for the Stokes
     solver (iterations / wall-clock to converge)?

Timings for the Arns conductivity build are reported separately, since the
enhanced model (footprint / local-thickness) is much heavier than the original.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver
from pyflowsolver.fastLaplacian import fast_laplacian_volume_generator

RADIUS = 8
LENGTH = 16
SCALE = 1.0
VISCOSITY = 1.0
TARGET_ERROR = 1e-6
MAX_ITERATIONS = 40000


def make_circular_duct(radius, length, margin=3):
    n = int(np.ceil(2 * radius)) + 2 * margin
    vol = np.zeros((n, n, length), dtype=np.float64)
    c = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    vol[(xx - c) ** 2 + (yy - c) ** 2 <= radius ** 2, :] = 1.0
    return vol


def _harmonic_face(c_lo, c_hi):
    both = (c_lo > 0) & (c_hi > 0)
    k = np.zeros_like(c_lo)
    k[both] = 2.0 / (1.0 / c_lo[both] + 1.0 / c_hi[both])
    return k


def arns_conductivity(binary, scale, enhanced):
    """Build the Arns conductivity from a binary geometry; returns (cond, secs)."""
    por = (binary > 0).astype(np.uint8) * 100
    t0 = time.perf_counter()
    cond = fast_laplacian_volume_generator(
        por, np.array([scale, scale, scale], dtype=float),
        closed_border=False, enhanced_model=enhanced,
    )
    return np.asarray(cond, dtype=np.float64), time.perf_counter() - t0


def darcy_velocity(cond, scale):
    """Darcy solve on a conductivity volume -> MAC face velocity + pressure."""
    vm = VolumeManager(np.array(cond, copy=True), scale=scale)
    dx, dy, dz = (float(s) for s in vm.scale[:3])
    a_sparse, b = vm.get_sparse_system_jit()
    s = DarcySolver()
    s.set_linear_system(a_sparse, b)
    s.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = s.solve_pcg()
    p = np.asarray(vm.ravel_sparse_solution(x), dtype=np.float64)
    c = np.asarray(vm.volume, dtype=np.float64)
    W, H, D = cond.shape
    u = np.zeros((W + 1, H, D)); v = np.zeros((W, H + 1, D)); w = np.zeros((W, H, D + 1))
    u[1:W, :, :] = _harmonic_face(c[:-1, :, :], c[1:, :, :]) * (p[:-1, :, :] - p[1:, :, :]) / dx
    v[:, 1:H, :] = _harmonic_face(c[:, :-1, :], c[:, 1:, :]) * (p[:, :-1, :] - p[:, 1:, :]) / dy
    w[:, :, 1:D] = _harmonic_face(c[:, :, :-1], c[:, :, 1:]) * (p[:, :, :-1] - p[:, :, 1:]) / dz
    return (u, v, w), p


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_stokes(volume, scale, initial_velocity=None):
    vm = VolumeManager(np.array(volume, copy=True), scale=scale)
    solver = StokesSolver(vm, initial_velocity=initial_velocity,
                          target_error=TARGET_ERROR, max_iterations=MAX_ITERATIONS)
    t0 = time.perf_counter()
    res = solver.solve()
    res["time"] = time.perf_counter() - t0
    return res


def main():
    volume = make_circular_duct(RADIUS, LENGTH)
    W, H, D = volume.shape
    print(f"Duct: shape={volume.shape}, R={RADIUS}, fluid={int((volume>0).sum())}")

    # JIT warmup (footprint + solvers)
    print("\nWarming up JIT...")
    warm = make_circular_duct(3, 5)
    arns_conductivity(warm, SCALE, enhanced=True)
    vel_w, _ = darcy_velocity(arns_conductivity(warm, SCALE, enhanced=False)[0], SCALE)
    run_stokes(warm, SCALE, initial_velocity=vel_w)

    # --- Build both Arns velocities ------------------------------------------
    results = {}
    for name, enhanced in (("original", False), ("enhanced", True)):
        cond, t_cond = arns_conductivity(volume, SCALE, enhanced)
        (u, v, w), p = darcy_velocity(cond, SCALE)
        results[name] = {"vel": (u, v, w), "t_cond": t_cond}
        print(f"\n[{name}] conductivity build: {t_cond:.3f} s")

    # --- Profile vs analytic Hagen-Poiseuille --------------------------------
    k = D // 2
    fluid = volume[:, :, k] > 0
    c = (W - 1) / 2.0
    yy, xx = np.mgrid[0:W, 0:H]
    rho2 = ((xx - c) ** 2 + (yy - c) ** 2)[fluid]
    hp = np.maximum(RADIUS ** 2 - rho2, 0.0)  # analytic parabola shape

    print("\n--- Darcy velocity profile vs analytic HP (mid-slice) ---")
    print(f"  {'model':>9} {'cosine(w, R^2-rho^2)':>22} {'cosine(w,(R-rho)^2)':>21}")
    conv = (np.sqrt(rho2))
    convex = (RADIUS - conv) ** 2
    for name in ("original", "enhanced"):
        wc = 0.5 * (results[name]["vel"][2][:, :, k] + results[name]["vel"][2][:, :, k + 1])
        wp = wc[fluid]
        print(f"  {name:>9} {cosine(wp, hp):>22.4f} {cosine(wp, convex):>21.4f}")

    # --- As Stokes initial guess ---------------------------------------------
    # The raw Arns/Darcy velocity overshoots the Stokes magnitude: because the
    # enhanced conductivity IS the Hagen-Poiseuille conductance, v_arns ~ 4*nu *
    # v_stokes, so an un-scaled guess sits ~4x too high in exactly the slow mode
    # and can hurt. Scaling by 1/(4*nu) recovers the right magnitude.
    norm = 1.0 / (4.0 * VISCOSITY)
    print("\n--- Stokes convergence with each guess ---")
    cold = run_stokes(volume, SCALE)
    print(f"  {'cold':>20}: {cold['iterations']:>6d} iters, {cold['time']:6.2f} s")
    for name in ("original", "enhanced"):
        u, v, w = results[name]["vel"]
        for label, factor in ((f"{name} raw", 1.0), (f"{name} x1/(4nu)", norm)):
            warm = run_stokes(volume, SCALE,
                              initial_velocity=(u * factor, v * factor, w * factor))
            speedup = cold["iterations"] / max(warm["iterations"], 1)
            print(f"  {label:>20}: {warm['iterations']:>6d} iters, {warm['time']:6.2f} s "
                  f"({speedup:.2f}x)")


if __name__ == "__main__":
    main()
