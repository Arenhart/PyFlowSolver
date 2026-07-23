"""How does the Arns first-guess speedup depend on stop tolerance & criterion?

Hypothesis: the Arns/Darcy guess already matches the *bulk* flow (~5%) but has
the wrong profile shape. So at a loose tolerance the guess is nearly "done"
(big speedup), while at a tight tolerance both cold and warm must pay for the
profile refinement (guess-independent) and the speedup shrinks.

We sweep the tolerance for each stop criterion:
  - "step"     : max|du| / max|u|                 (relative change; scales w/ dt)
  - "residual" : max|du|/dt / max|u|              (steady momentum residual; dt-free)

and, at each tolerance, run the solver cold vs. warm (Arns velocity guess),
reporting iterations, the cold/warm speedup, and the accuracy each run actually
reaches versus a finely-converged reference. All timings follow a JIT warm-up.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver

RADIUS = 6
LENGTH = 10
SCALE = 1.0
MAX_ITERATIONS = 40000

TOLERANCES = {
    "step": [1e-3, 1e-4, 1e-5, 1e-6],
    "residual": [1e-2, 1e-3, 1e-4, 1e-5],
}


def make_circular_duct(radius, length, margin=2):
    n = int(np.ceil(2 * radius)) + 2 * margin
    volume = np.zeros((n, n, length), dtype=np.float64)
    center = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    volume[(xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2, :] = 1.0
    return volume


def _harmonic_face(c_lo, c_hi):
    both = (c_lo > 0) & (c_hi > 0)
    k = np.zeros_like(c_lo)
    k[both] = 2.0 / (1.0 / c_lo[both] + 1.0 / c_hi[both])
    return k


def arns_velocity(volume, scale):
    """MAC velocity guess from one Arns/Darcy solve (per-face Darcy flux)."""
    vm = VolumeManager(np.array(volume, copy=True), scale=scale)
    dx, dy, dz = (float(s) for s in vm.scale[:3])
    vm.convert_pore_volume_to_laplacian_conductivity()
    a_sparse, b = vm.get_sparse_system_jit()
    s = DarcySolver()
    s.set_linear_system(a_sparse, b)
    s.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = s.solve_pcg()
    p = np.asarray(vm.ravel_sparse_solution(x), dtype=np.float64)
    c = np.asarray(vm.volume, dtype=np.float64)
    W, H, D = volume.shape
    u = np.zeros((W + 1, H, D)); v = np.zeros((W, H + 1, D)); w = np.zeros((W, H, D + 1))
    u[1:W, :, :] = _harmonic_face(c[:-1, :, :], c[1:, :, :]) * (p[:-1, :, :] - p[1:, :, :]) / dx
    v[:, 1:H, :] = _harmonic_face(c[:, :-1, :], c[:, 1:, :]) * (p[:, :-1, :] - p[:, 1:, :]) / dy
    w[:, :, 1:D] = _harmonic_face(c[:, :, :-1], c[:, :, 1:]) * (p[:, :, :-1] - p[:, :, 1:]) / dz
    return u, v, w


def run(volume, scale, criterion, tol, initial_velocity=None):
    vm = VolumeManager(np.array(volume, copy=True), scale=scale)
    solver = StokesSolver(
        vm.volume, scale=vm.scale, initial_velocity=initial_velocity,
        convergence_criterion=criterion, target_error=tol,
        max_iterations=MAX_ITERATIONS,
    )
    return solver.solve()


def rel_error(w, w_ref):
    denom = np.abs(w_ref).max()
    return np.abs(w - w_ref).max() / denom if denom > 0 else np.nan


def main():
    volume = make_circular_duct(RADIUS, LENGTH)
    print(f"Circular duct: shape={volume.shape}, "
          f"fluid={int((volume > 0).sum())}/{volume.size}")

    print("\nWarming up JIT...")
    run(make_circular_duct(3, 4), SCALE, "step", 1e-3)

    guess = arns_velocity(volume, SCALE)

    print("Computing fine reference solution...")
    ref = run(volume, SCALE, "residual", 1e-6)
    w_ref = ref["w"]
    print(f"  reference: {ref['iterations']} iters, converged={ref['converged']}")
    print(f"  Arns guess accuracy vs reference: rel_err(w) = {rel_error(guess[2], w_ref):.3e}")

    for criterion in ("step", "residual"):
        print(f"\n=== criterion = {criterion} ===")
        print(f"  {'tol':>8} | {'cold it':>8} {'warm it':>8} {'speedup':>8} | "
              f"{'cold err':>9} {'warm err':>9}")
        for tol in TOLERANCES[criterion]:
            cold = run(volume, SCALE, criterion, tol)
            warm = run(volume, SCALE, criterion, tol, initial_velocity=guess)
            speedup = cold["iterations"] / max(warm["iterations"], 1)
            print(f"  {tol:>8.0e} | {cold['iterations']:>8d} {warm['iterations']:>8d} "
                  f"{speedup:>7.1f}x | {rel_error(cold['w'], w_ref):>9.2e} "
                  f"{rel_error(warm['w'], w_ref):>9.2e}")

    # ---------------------------------------------------------------- #
    # Why the residual criterion is worth having: it is dt-independent.
    # Shrinking time_step_factor makes the "step" metric smaller for the same
    # field, so it stops early at a WORSE accuracy; the "residual" metric
    # (max|R|/max|u|) depends only on the field, so its accuracy is stable.
    # ---------------------------------------------------------------- #
    print("\n=== dt-independence (vary time_step_factor at fixed tol) ===")
    print(f"  {'criterion':>9} {'tol':>7} {'factor':>7} | {'iters':>6} {'accuracy':>9}")
    for criterion, tol in (("step", 1e-4), ("residual", 1e-3)):
        for factor in (0.5, 0.25, 0.1):
            vm = VolumeManager(np.array(volume, copy=True), scale=SCALE)
            solver = StokesSolver(vm.volume, scale=vm.scale, convergence_criterion=criterion,
                                  target_error=tol, time_step_factor=factor,
                                  max_iterations=MAX_ITERATIONS)
            res = solver.solve()
            print(f"  {criterion:>9} {tol:>7.0e} {factor:>7.2f} | "
                  f"{res['iterations']:>6d} {rel_error(res['w'], w_ref):>9.2e}")


if __name__ == "__main__":
    main()
