"""Benchmark: does an Arns fast-Laplacian guess speed up the Stokes solve?

The "Arns fast Laplacian" path builds an EDT-based conductivity
(``VolumeManager.convert_pore_volume_to_laplacian_conductivity``) and solves a
single condensed Darcy system. From that solve we get both an approximate
pressure field and, via the per-face Darcy flux ``k_face * grad(p)``, an
approximate MAC velocity field. Either can seed the Stokes solver as a warm
start.

We compare, on the same circular duct:
  1. Stokes cold                       (no guess)
  2. Stokes warm, Arns pressure guess  (initial_pressure only)
  3. Stokes warm, Arns velocity guess  (initial_velocity from Darcy flux)

The key insight (verified below): the pseudo-transient outer loop tracks the
*velocity* diffusing to steady state, so the pressure guess barely helps while
the velocity guess does. All timings are taken after a JIT warm-up so Numba
compilation is excluded.
"""

import os
import sys
import time

import numpy as np

# Ensure the repo root (not any stale installed copy) is imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver

RADIUS = 8
LENGTH = 16
SCALE = 1.0
TARGET_ERROR = 1e-6
MAX_ITERATIONS = 20000


def make_circular_duct(radius, length, margin=2):
    n = int(np.ceil(2 * radius)) + 2 * margin
    volume = np.zeros((n, n, length), dtype=np.float64)
    center = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    disk = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    volume[disk, :] = 1.0
    return volume


def _harmonic_face(c_lo, c_hi):
    """2 / (1/c_lo + 1/c_hi) on faces where both cells are fluid, else 0."""
    both = (c_lo > 0) & (c_hi > 0)
    k = np.zeros_like(c_lo)
    k[both] = 2.0 / (1.0 / c_lo[both] + 1.0 / c_hi[both])
    return k


def arns_estimate(volume, scale):
    """Return (pressure, (u, v, w)) estimates from one Arns/Darcy solve.

    Pressure is the raveled Darcy solution. The MAC face velocities are the
    Darcy fluxes  v_face = k_face * (p_lo - p_hi) / d_axis  (flow from high to
    low pressure). Interior faces only; the solver's BC step reopens the
    inlet/outlet faces from the interior when the guess is applied.
    """
    vm = VolumeManager(np.array(volume, copy=True), scale=scale)
    dx, dy, dz = (float(s) for s in vm.scale[:3])
    vm.convert_pore_volume_to_laplacian_conductivity()
    a_sparse, b = vm.get_sparse_system_jit()
    solver = DarcySolver()
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = solver.solve_pcg()

    p = np.asarray(vm.ravel_sparse_solution(x), dtype=np.float64)
    c = np.asarray(vm.volume, dtype=np.float64)  # Arns conductivity per voxel
    W, H, D = volume.shape

    u = np.zeros((W + 1, H, D))
    v = np.zeros((W, H + 1, D))
    w = np.zeros((W, H, D + 1))

    ku = _harmonic_face(c[:-1, :, :], c[1:, :, :])
    u[1:W, :, :] = ku * (p[:-1, :, :] - p[1:, :, :]) / dx
    kv = _harmonic_face(c[:, :-1, :], c[:, 1:, :])
    v[:, 1:H, :] = kv * (p[:, :-1, :] - p[:, 1:, :]) / dy
    kw = _harmonic_face(c[:, :, :-1], c[:, :, 1:])
    w[:, :, 1:D] = kw * (p[:, :, :-1] - p[:, :, 1:]) / dz

    return p, (u, v, w)


def run_stokes(volume, scale, initial_pressure=None, initial_velocity=None):
    vm = VolumeManager(np.array(volume, copy=True), scale=scale)
    solver = StokesSolver(
        vm,
        initial_pressure=initial_pressure,
        initial_velocity=initial_velocity,
        target_error=TARGET_ERROR,
        max_iterations=MAX_ITERATIONS,
    )
    t0 = time.perf_counter()
    result = solver.solve()
    result["time"] = time.perf_counter() - t0
    return result


def main():
    volume = make_circular_duct(RADIUS, LENGTH)
    print(f"Circular duct: shape={volume.shape}, "
          f"fluid voxels={int((volume > 0).sum())}/{volume.size}")

    # Warm up the JIT kernels so compilation time is excluded from timings.
    print("\nWarming up JIT (small duct)...")
    warm = make_circular_duct(3, 5)
    p_w, vel_w = arns_estimate(warm, SCALE)
    run_stokes(warm, SCALE)
    run_stokes(warm, SCALE, initial_velocity=vel_w)

    # --- 1. Cold --------------------------------------------------------------
    print("\n--- Stokes cold (no guess) ---")
    cold = run_stokes(volume, SCALE)
    print(f"  iterations={cold['iterations']}, time={cold['time']:.3f} s")

    # Upper bound: seed the (now known) steady field. A perfect guess must stop
    # almost immediately -- this confirms the warm-start machinery is correct
    # and sets the best case any approximate guess could approach.
    print("\n--- Stokes warm (exact steady field -> upper bound) ---")
    exact = run_stokes(volume, SCALE,
                       initial_velocity=(cold["u"].copy(), cold["v"].copy(), cold["w"].copy()))
    print(f"  iterations={exact['iterations']} (perfect guess)")

    # --- Arns estimate --------------------------------------------------------
    print("\n--- Arns fast-Laplacian estimate (one Darcy solve) ---")
    t0 = time.perf_counter()
    p_guess, vel_guess = arns_estimate(volume, SCALE)
    t_estimate = time.perf_counter() - t0
    print(f"  estimate time={t_estimate:.3f} s")

    # --- 2. Warm, pressure guess ---------------------------------------------
    print("\n--- Stokes warm (Arns pressure guess) ---")
    warm_p = run_stokes(volume, SCALE, initial_pressure=p_guess)
    print(f"  iterations={warm_p['iterations']}, time={warm_p['time']:.3f} s")

    # --- 3. Warm, velocity guess ---------------------------------------------
    print("\n--- Stokes warm (Arns velocity guess) ---")
    warm_v = run_stokes(volume, SCALE,
                        initial_pressure=p_guess, initial_velocity=vel_guess)
    print(f"  iterations={warm_v['iterations']}, time={warm_v['time']:.3f} s")

    # --- Comparison -----------------------------------------------------------
    print("\n--- Comparison ---")
    print(f"  cold            : {cold['iterations']:>6d} iters, {cold['time']:7.3f} s")
    print(f"  exact seed      : {exact['iterations']:>6d} iters  (upper bound)")
    print(f"  pressure guess  : {warm_p['iterations']:>6d} iters, {warm_p['time']:7.3f} s")
    print(f"  velocity guess  : {warm_v['iterations']:>6d} iters, {warm_v['time']:7.3f} s"
          f"  (+ {t_estimate:.3f} s estimate)")
    print(f"  iteration reduction (velocity guess): "
          f"{cold['iterations'] / max(warm_v['iterations'], 1):.1f}x")
    print(f"  wall-clock speedup (total, velocity): "
          f"{cold['time'] / (warm_v['time'] + t_estimate):.2f}x")

    for name, res in (("pressure", warm_p), ("velocity", warm_v)):
        max_diff = np.abs(cold["w"] - res["w"]).max()
        print(f"  max|w_cold - w_{name}|: {max_diff:.2e}  (same field: {max_diff < 1e-4})")


if __name__ == "__main__":
    main()
