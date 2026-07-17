"""Scaling / RAM stress test for the multigrid pressure-Poisson solve.

Loads a real 250^3 Bentheimer sample and tiles it to progressively larger
volumes (250, 500, ... up to 1500^3), running the full scalable pipeline at each
size -- VolumeManager assembly -> native algebraic-multigrid Poisson solve
(MG-PCG) -- while tracking wall-clock time and RAM. Prints one row per size.

This measures the load-bearing primitive for large complex volumes (the Phase-1
multigrid Poisson solve, stokes_solver.md s10.3), and, just as importantly, where
memory becomes the limit -- the RAM columns show the high-water mark so you can
see which size your machine can actually handle.

Run it directly; it prints each row as soon as that size finishes (flushed), so
you can watch progress and stop it (Ctrl-C) whenever a size is too big or slow:

    python scripts/stress_test_poisson.py            # code 000, up to 1500^3
    python scripts/stress_test_poisson.py 002 4      # sample 002, up to 1000^3

Notes
-----
* Upscaling is by *tiling* (np.tile), which preserves the pore-scale features and
  porosity and keeps the z-percolation of the base sample -- a realistic "more of
  the same rock" stress rather than a blurred upsample.
* The geometry is carried as float32 and the base is freed between sizes to keep
  the geometry footprint as small as possible; the CSR matrix + AMG hierarchy
  dominate RAM at scale.
* A solve that does not reach the tolerance (residual column) at large sizes is a
  red flag that the assembly overflowed some index type -- report it rather than
  trusting the numbers.
"""

import os
import sys
import gc
import time

import numpy as np
from scipy.io import netcdf_file
import psutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.multigridSolver import MultigridSolver

BASE = os.path.join(REPO, "tests", "unit", "resources", "numerical_solved", "netcdf")
TARGET_ERROR = 1e-6
MAX_MG_ITERS = 200


def load_base(code):
    """Full-resolution binary Bentheimer geometry as a bool array (True = pore).

    BIN files store 0 = pore, 1 = solid; flow axis x is moved to z (our driving
    axis), matching the OpenFOAM comparison scripts.
    """
    path = f"{BASE}/BIN_Bentheimer{code}.nc"
    bf = netcdf_file(path, "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    pore = np.transpose(geom == 0, (1, 2, 0))
    return np.ascontiguousarray(pore)


def ram_gb():
    """(current RSS, process peak) in GB. peak_wset on Windows, else RSS."""
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    return mi.rss / 1e9, (peak / 1e9 if peak is not None else mi.rss / 1e9)


def run_size(pore_base, factor):
    side = pore_base.shape[0] * factor
    # Tile to (side)^3, carried as float32 (VolumeManager wants a float volume).
    pore = np.tile(pore_base, (factor, factor, factor))
    geom = pore.astype(np.float32)
    del pore
    gc.collect()
    porosity = float(geom.mean())

    t = time.perf_counter()
    vm = VolumeManager(geom, scale=1.0)
    a_sparse, b = vm.get_sparse_system_jit()
    t_assemble = time.perf_counter() - t
    n = int(b.size)
    nnz = int(a_sparse["val"].size)
    rss_after_assemble, _ = ram_gb()

    solver = MultigridSolver(backend="native", target_error=TARGET_ERROR,
                             max_iterations=MAX_MG_ITERS)
    solver.set_linear_system(a_sparse, b)
    t = time.perf_counter()
    solver.generate_preconditioner()
    t_setup = time.perf_counter() - t
    levels = [lvl["n"] for lvl in solver.levels]
    rss_after_setup, _ = ram_gb()

    t = time.perf_counter()
    x, err, iters = solver.solve_pcg()
    t_solve = time.perf_counter() - t
    rss_now, rss_peak = ram_gb()

    # Free everything before the next (larger) size.
    del geom, vm, a_sparse, b, solver, x
    gc.collect()

    return {
        "side": side, "n": n, "nnz": nnz, "porosity": porosity,
        "t_assemble": t_assemble, "t_setup": t_setup, "t_solve": t_solve,
        "mg_iters": iters, "residual": err, "levels": len(levels),
        "rss_assemble": rss_after_assemble, "rss_setup": rss_after_setup,
        "rss_peak": rss_peak,
    }


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "000"
    max_factor = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print(f"Loading base Bentheimer{code} ...", flush=True)
    pore_base = load_base(code)
    b = pore_base.shape[0]
    print(f"  base shape = {pore_base.shape}, porosity = {pore_base.mean():.3f}", flush=True)
    print(f"  tiling up to factor {max_factor} -> {b * max_factor}^3\n", flush=True)

    # Warm up the Numba kernels on a tiny system so the first timed row is clean.
    print("Warming up JIT ...", flush=True)
    warm = np.zeros((16, 16, 16), dtype=np.float32); warm[4:12, 4:12, :] = 1.0
    _wvm = VolumeManager(warm, scale=1.0); _wa, _wb = _wvm.get_sparse_system_jit()
    _ws = MultigridSolver(backend="native"); _ws.set_linear_system(_wa, _wb)
    _ws.generate_preconditioner(); _ws.solve_pcg()
    del warm, _wvm, _wa, _wb, _ws; gc.collect()

    header = (f"{'size':>7} {'N_fluid':>13} {'nnz':>14} {'por':>5} "
              f"{'asm[s]':>9} {'setup[s]':>9} {'solve[s]':>9} {'lvls':>4} "
              f"{'mg_it':>6} {'resid':>9} {'RAM_asm':>8} {'RAM_set':>8} {'RAM_pk':>8}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for factor in range(1, max_factor + 1):
        try:
            r = run_size(pore_base, factor)
        except MemoryError:
            print(f"{b*factor:>6}^3  MemoryError during assembly/solve -- stopping.",
                  flush=True)
            break
        except Exception as exc:
            print(f"{b*factor:>6}^3  ERROR: {type(exc).__name__}: {exc}", flush=True)
            break
        print(f"{str(r['side'])+'^3':>7} {r['n']:>13,} {r['nnz']:>14,} "
              f"{r['porosity']:>5.3f} {r['t_assemble']:>9.1f} {r['t_setup']:>9.1f} "
              f"{r['t_solve']:>9.1f} {r['levels']:>4} {r['mg_iters']:>6} "
              f"{r['residual']:>9.1e} {r['rss_assemble']:>7.1f}G "
              f"{r['rss_setup']:>7.1f}G {r['rss_peak']:>7.1f}G", flush=True)


if __name__ == "__main__":
    main()
