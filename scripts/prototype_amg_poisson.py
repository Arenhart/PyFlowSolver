"""Benchmark: AMG (pyamg backend) vs diagonal-PCG on the pressure-Poisson system.

Purpose (stokes_solver.md s10.3, "Prototype then port"): pull a *real* Poisson
system out of the same assembly `StokesSolver._build_pressure_poisson_system`
uses (VolumeManager -> condensed CSR Laplacian), then compare:

  1. `DarcySolver.solve_pcg`             -- diagonal-preconditioned CG (baseline)
  2. `MultigridSolver(backend="pyamg")`  -- MG-PCG and standalone V-cycle

across a size sweep (ducts + blob "rock" subvolumes). The point is to confirm
the two multigrid signatures before porting the hot loops to a native backend:

  * iteration count stays ~flat as N grows (the baseline grows), and
  * the MG solution matches the PCG solution to tolerance.

This script is kept permanently as the benchmark harness, not a throwaway.
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.multigridSolver import MultigridSolver

TARGET_ERROR = 1e-8


def make_circular_duct(radius, length, margin=2):
    n = int(np.ceil(2 * radius)) + 2 * margin
    volume = np.zeros((n, n, length), dtype=np.float64)
    c = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    volume[(xx - c) ** 2 + (yy - c) ** 2 <= radius ** 2, :] = 1.0
    return volume


def make_blob_rock(side, porosity=0.35, seed=0):
    """A percolating blob "rock" subvolume via porespy (fallback: random duct)."""
    try:
        import porespy as ps
        im = ps.generators.blobs(shape=(side, side, side), porosity=porosity,
                                  blobiness=1.0, seed=seed)
        return im.astype(np.float64)
    except Exception as exc:  # porespy optional / API drift -> skip gracefully
        print(f"  (porespy unavailable: {exc}; skipping blob case)")
        return None


def poisson_system(volume):
    """Bare-Laplacian condensed CSR + boundary RHS, as StokesSolver builds it."""
    vm = VolumeManager((volume > 0).astype(np.float64), scale=1.0)
    a_sparse, b = vm.get_sparse_system_jit()
    return a_sparse, b, vm.nonzeros


def run_pcg(a_sparse, b, x0):
    solver = DarcySolver(target_error=TARGET_ERROR)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    t = time.perf_counter()
    x, err, it = solver.solve_pcg(X0=x0.copy())
    return x, err, it, time.perf_counter() - t


def run_amg(a_sparse, b, x0, backend, accel):
    solver = MultigridSolver(backend=backend, target_error=TARGET_ERROR)
    solver.set_linear_system(a_sparse, b)
    t_setup = time.perf_counter()
    solver.generate_preconditioner()
    setup = time.perf_counter() - t_setup
    t = time.perf_counter()
    x, err, it = solver._solve(X0=x0.copy(), accel=accel)
    return x, err, it, time.perf_counter() - t, setup, solver


def bench_case(name, volume):
    a_sparse, b, n = poisson_system(volume)
    x0 = np.zeros_like(b)

    x_pcg, e_pcg, it_pcg, t_pcg = run_pcg(a_sparse, b, x0)
    x_nat, e_nat, it_nat, t_nat, s_nat, ms = run_amg(a_sparse, b, x0, "native", "cg")
    x_py, e_py, it_py, t_py, s_py, _ = run_amg(a_sparse, b, x0, "pyamg", "cg")

    denom = np.linalg.norm(x_pcg) or 1.0
    d_nat = np.linalg.norm(x_nat - x_pcg) / denom
    d_py = np.linalg.norm(x_py - x_pcg) / denom
    sizes = [lvl["n"] for lvl in ms.levels]

    print(f"\n=== {name}  (N={n}, nnz={a_sparse['val'].size}) ===")
    print(f"  native AMG levels: {len(sizes)}  sizes: {sizes}")
    print(f"  {'method':<20}{'iters':>8}{'rel.res':>12}{'time[s]':>10}{'setup[s]':>10}")
    print(f"  {'diagonal-PCG':<20}{it_pcg:>8}{e_pcg:>12.2e}{t_pcg:>10.3f}{'-':>10}")
    print(f"  {'native MG-PCG':<20}{it_nat:>8}{e_nat:>12.2e}{t_nat:>10.3f}{s_nat:>10.3f}")
    print(f"  {'pyamg MG-PCG':<20}{it_py:>8}{e_py:>12.2e}{t_py:>10.3f}{s_py:>10.3f}")
    print(f"  solution rel.diff vs PCG:  native {d_nat:.2e}   pyamg {d_py:.2e}")
    return {
        "name": name, "N": n, "it_pcg": it_pcg, "it_nat": it_nat, "it_py": it_py,
        "t_pcg": t_pcg, "t_nat": t_nat, "t_py": t_py,
        "rel_diff": max(d_nat, d_py),
    }


def main():
    print("Warming up Numba (DarcySolver JIT compile)...")
    warm = make_circular_duct(4, 6)
    a, b, _ = poisson_system(warm)
    run_pcg(a, b, np.zeros_like(b))

    rows = []
    # Duct sweep: fixed cross-section, growing length -> growing N.
    for radius, length in [(8, 16), (8, 48), (12, 48), (16, 64)]:
        rows.append(bench_case(f"duct r{radius} L{length}",
                               make_circular_duct(radius, length)))
    # Blob "rock" sweep: growing cubic side.
    for side in [40, 60, 80]:
        vol = make_blob_rock(side)
        if vol is not None:
            rows.append(bench_case(f"blob {side}^3", vol))

    print("\n\n===== SUMMARY: iterations vs N (flat AMG = win) =====")
    print(f"{'case':<18}{'N':>8}{'PCG it':>9}{'nat it':>9}{'py it':>9}"
          f"{'PCG t':>9}{'nat t':>9}{'py t':>9}")
    for r in rows:
        print(f"{r['name']:<18}{r['N']:>8}{r['it_pcg']:>9}{r['it_nat']:>9}"
              f"{r['it_py']:>9}{r['t_pcg']:>9.3f}{r['t_nat']:>9.3f}{r['t_py']:>9.3f}")
    worst = max(r["rel_diff"] for r in rows)
    print(f"\nworst AMG vs PCG solution rel.diff: {worst:.2e}")


if __name__ == "__main__":
    main()
