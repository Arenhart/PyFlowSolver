"""Benchmark: is the solver already fast enough for large voxel volumes and pore
networks? Two independent parts, both printed as they finish (flush), Ctrl-C safe.

PART A -- voxel volumes via the distributed Schwarz path (schwarzSolver).
    Tiles a 250^3 base up to a target side and, per size, reports the metrics
    that decide cluster feasibility:
      * per-slab RAM   -- what ONE cluster node must hold (the RAM win). Measured
                          by assembling+setting-up a single slab in isolation.
      * full-grid RAM  -- the geometry + global fluid-index overhead that the
                          current serial build still holds on the driver (a known
                          optimization target for the true distributed version).
      * rounds / time  -- from a full serial solve, run only when the whole
                          problem still fits this machine (below FULL_SOLVE_CAP);
                          for bigger sizes only the per-slab (per-node) numbers
                          are reported, since the full solve is what the cluster
                          is for. Estimated distributed wall-clock ~= rounds x
                          (per-slab local-solve time + halo comm [not measured]).

PART B -- pore networks via the multigrid solver (NetworkManager + MultigridSolver).
    Cubic pore lattices of growing size; reports MG-PCG iters, solve time, RAM.

By default the convergence (round-count) measurement is OFF -- it is a slow
serial simulation. Pass "conv" to enable it (up to FULL_SOLVE_CAP cells).

Usage:
    python scripts/benchmark_distributed_scaling.py                 # defaults, fast
    python scripts/benchmark_distributed_scaling.py 000 1000 40     # code, max side, slab thickness
    python scripts/benchmark_distributed_scaling.py 000 1000 40 voxel   # only PART A
    python scripts/benchmark_distributed_scaling.py 000 1000 40 network # only PART B
    python scripts/benchmark_distributed_scaling.py 000 500 100 conv    # also measure rounds (slow)
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

from pyflowsolver.multigridSolver import MultigridSolver
from pyflowsolver.networkManager import NetworkManager
from pyflowsolver.schwarzSolver import (
    SchwarzSolver, global_fluid_index, slab_ranges, assemble_slab,
    filter_percolating)

BASE = os.path.join(REPO, "tests", "unit", "resources", "numerical_solved", "netcdf")
TARGET_ERROR = 1e-6
# Convergence (round count) is measured by a full SERIAL Schwarz solve -- all
# slabs solved sequentially per round, so it is ~n_partitions x the true
# distributed wall-clock and genuinely slow (minutes) even at 250^3. It is OFF by
# default so the benchmark returns the fast per-node RAM/assembly answer; pass
# "conv" on the command line to also measure rounds, up to this fluid-cell cap.
CONV_ENABLED = False
FULL_SOLVE_CAP = 8_000_000
# The full solve is a SERIAL simulation (all slabs solved sequentially per round),
# so it is ~n_partitions x the true distributed wall-clock. Cap the rounds so it
# can't spin: hitting the cap means the config needs a richer coarse space.
BENCH_MAX_ROUNDS = 400


def ram_gb():
    mi = psutil.Process().memory_info()
    return mi.rss / 1e9, getattr(mi, "peak_wset", mi.rss) / 1e9


def load_base(code):
    bf = netcdf_file(f"{BASE}/BIN_Bentheimer{code}.nc", "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    return np.ascontiguousarray(np.transpose(geom == 0, (1, 2, 0)))


# --------------------------------------------------------------------------- #
# PART A -- voxel volumes via distributed Schwarz
# --------------------------------------------------------------------------- #
def bench_voxel(pore_base, max_side, slab_thickness):
    base = pore_base.shape[0]
    print("\n=== PART A: voxel volumes (distributed Schwarz) ===", flush=True)
    hdr = (f"{'side':>7} {'N_fluid':>13} {'slabs':>6} {'slab_N':>11} "
           f"{'slabRAM':>8} {'gridRAM':>8} {'asm[s]':>8}")
    print("per-node metrics (fast) print first; convergence line follows.", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    factor = 1
    while base * factor <= max_side:
        side = base * factor
        print(f"  [computing {side}^3: tiling + percolating filter ...]", flush=True)
        # Filter to the percolating cluster (driver-side, O(N) scipy.ndimage.label
        # -- a real preprocessing cost, ~20s at 500^3). This is required: on the
        # filtered geometry every fluid cell keeps >=1 fluid neighbour, so slab
        # matrices are non-singular -- the local solve converges (real per-node
        # timing) and the full solve converges. On unfiltered geometry, isolated
        # pores are singular and an iterative solve just spins to its cap.
        t = time.perf_counter()
        mask = filter_percolating(np.tile(pore_base, (factor, factor, factor)))
        gc.collect()
        gidx, N = global_fluid_index(mask)
        t_prep = time.perf_counter() - t
        n_part = max(2, side // slab_thickness)
        ranges = slab_ranges(side, n_part)
        print(f"  [filter+index {t_prep:.1f}s; assembling per-node slab ...]", flush=True)

        # Per-node RAM + assembly cost for ONE slab (matrix + AMG hierarchy).
        # This is well-defined and fast; it is the cluster-node RAM requirement.
        # (We do NOT time an isolated single-slab solve: a slab in isolation can
        # be near-singular -- its inter-slab coupling is a halo/Dirichlet term
        # that is only supplied during the real iteration -- so an isolated solve
        # doesn't converge. Solve timing comes from the full solve below.)
        _, grid_ram = ram_gb()                      # geometry + gidx overhead
        z_lo, z_hi = ranges[0]
        rss0, _ = ram_gb()
        t = time.perf_counter()
        slab = assemble_slab(mask, gidx, z_lo, z_hi)
        mg = MultigridSolver(backend="native", target_error=TARGET_ERROR)
        mg.set_linear_system(slab["a_sparse"], slab["b_base"].copy())
        mg.generate_preconditioner()
        t_asm = time.perf_counter() - t
        rss1, _ = ram_gb()
        slab_ram = max(rss1 - rss0, 0.0)
        slab_N = slab["local_n"]
        del slab, mg
        gc.collect()

        # Per-node line first (this is the fast, RAM-relevant answer).
        print(f"{str(side)+'^3':>7} {N:>13,} {n_part:>6} {slab_N:>11,} "
              f"{slab_ram:>7.1f}G {grid_ram:>7.1f}G {t_asm:>8.1f}", flush=True)

        # Convergence + solve timing from the full solve (well-posed: the halo
        # pins every slab each round). Capped; serial (all slabs sequentially per
        # round). The distributed wall-clock ~= serial / n_partitions (+ comm),
        # since a cluster runs the n_partitions slab solves of a round in parallel.
        if CONV_ENABLED and N <= FULL_SOLVE_CAP:
            print("        -> measuring convergence (slow serial solve) ...",
                  flush=True)
            solver = SchwarzSolver(mask, n_partitions=n_part,
                                   target_error=TARGET_ERROR,
                                   max_rounds=BENCH_MAX_ROUNDS,
                                   filter_disconnected=False)   # already filtered
            t = time.perf_counter()
            _, rounds, res = solver.solve_serial()
            t_full = time.perf_counter() - t
            capped = rounds >= BENCH_MAX_ROUNDS and res > TARGET_ERROR
            est_dist = t_full / n_part           # ~ distributed wall-clock (ex-comm)
            tag = f" [CAPPED at {BENCH_MAX_ROUNDS}: needs richer coarse space]" if capped else ""
            print(f"        -> converge: rounds={rounds} res={res:.1e}  "
                  f"serial={t_full:.1f}s  est.distributed~={est_dist:.1f}s "
                  f"(serial/n_part){tag}", flush=True)
            del solver
            gc.collect()
        elif not CONV_ENABLED:
            print("        -> convergence not measured (pass 'conv' to enable; "
                  "slow serial solve)", flush=True)
        else:
            print(f"        -> convergence skipped (N > {FULL_SOLVE_CAP:,})",
                  flush=True)

        del mask, gidx
        gc.collect()
        factor += 1


# --------------------------------------------------------------------------- #
# PART B -- pore networks via multigrid
# --------------------------------------------------------------------------- #
def cubic_network(nx, ny, nz):
    def idx(x, y, z):
        return (x * ny + y) * nz + z
    conn, cond = [], []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = idx(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if xx < nx and yy < ny and zz < nz:
                        conn.append((i, idx(xx, yy, zz)))
                        cond.append(1.0 + 0.1 * ((i * 7) % 5))
    n = nx * ny * nz
    inlets = np.zeros(n, dtype=bool); outlets = np.zeros(n, dtype=bool)
    for y in range(ny):
        for z in range(nz):
            inlets[idx(0, y, z)] = True
            outlets[idx(nx - 1, y, z)] = True
    return (np.array(conn, dtype=np.int64), np.array(cond, dtype=np.float64),
            inlets, outlets)


def bench_network(sides):
    print("\n=== PART B: pore networks (multigrid) ===", flush=True)
    hdr = f"{'lattice':>12} {'pores':>12} {'throats':>12} {'mg_it':>6} {'resid':>9} {'solve[s]':>9} {'RAM_pk':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    print("(single-node multigrid -- pore networks have no distributed path yet; "
          "expect a RAM wall around ~10^8 pores)", flush=True)
    for side in sides:
        try:
            conn, cond, inlets, outlets = cubic_network(side, side, side)
            nm = NetworkManager(conn, cond, inlets, outlets)
            nm.generate_sparse_system()
            a, b = nm.get_sparse_system()
            mg = MultigridSolver(backend="native", target_error=TARGET_ERROR)
            mg.set_linear_system(a, b)
            mg.generate_preconditioner()          # AMG setup: the RAM bottleneck
            t = time.perf_counter()
            _, err, it = mg.solve_pcg()
            t_solve = time.perf_counter() - t
            _, ram_pk = ram_gb()
            print(f"{f'{side}^3':>12} {side**3:>12,} {b.size:>12,} {it:>6} "
                  f"{err:>9.1e} {t_solve:>9.2f} {ram_pk:>7.1f}G", flush=True)
            del conn, cond, nm, a, b, mg
        except MemoryError:
            print(f"{f'{side}^3':>12} {side**3:>12,}  -- MemoryError (single-node "
                  f"AMG RAM wall; needs a distributed/graph-partitioned solver) -- "
                  f"stopping", flush=True)
            gc.collect()
            break
        gc.collect()


def main():
    global CONV_ENABLED
    CONV_ENABLED = "conv" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "conv"]   # positional args sans flag
    code = args[0] if len(args) > 0 else "000"
    max_side = int(args[1]) if len(args) > 1 else 1000
    slab_thickness = int(args[2]) if len(args) > 2 else 50
    which = args[3] if len(args) > 3 else "both"

    print("Warming up JIT ...", flush=True)
    warm = np.zeros((16, 16, 16), dtype=np.float64); warm[4:12, 4:12, :] = 1.0
    SchwarzSolver(warm, n_partitions=2, max_rounds=5).solve_serial()
    del warm; gc.collect()

    if which in ("both", "voxel"):
        pore_base = load_base(code)
        print(f"base Bentheimer{code}: {pore_base.shape}, porosity={pore_base.mean():.3f}",
              flush=True)
        try:
            bench_voxel(pore_base, max_side, slab_thickness)
        except MemoryError:
            print("  -- MemoryError in PART A (driver holds full-grid arrays; the "
                  "per-node slab RAM is the cluster figure) -- stopping PART A",
                  flush=True)
            gc.collect()
    if which in ("both", "network"):
        bench_network([20, 40, 60, 80, 100, 200, 300, 500, 600, 800, 1000])


if __name__ == "__main__":
    main()
