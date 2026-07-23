"""Self-consistency check for the Stokes warm-start machinery.

Idea: warm-starting a solve with a *partial* result of the same solve must not
throw work away. Concretely, with the explicit predictor (a stationary linear
iteration whose entire state is carried by the MAC velocity field):

    1. Cold solve to convergence            -> N_total iterations.
    2. Cold solve capped at N1 = N_total//3 -> grab (u, v, w, p) mid-transient.
    3. Fresh solve warm-started from those   -> N2 iterations to converge.

If the warm start is applied correctly, run 3 resumes exactly where run 2 left
off, so it should finish in the remaining ~2/3:  N1 + N2 ~= N_total.

Failure signatures:
  * N2 ~= N_total (so N1 + N2 ~= 4/3 N_total): the seed was dropped / re-zeroed
    and the field re-develops from scratch.
  * run 3 converges to a *different* field: the seed corrupts the state.

Run it yourself (it is not auto-run):  python scripts/check_warmstart_selfconsistency.py
Tune the geometry with the constants below.
"""

import os
import numpy as np
from scipy.io import netcdf_file

from pyflowsolver.stokesSolver import StokesSolver

# ---- knobs -------------------------------------------------------------------
DOWNSCALE_FACTOR = 2          # Bentheimer f: 5 -> 50^3 (fast); 2 -> 125^3
PREDICTOR = "explicit"        # explicit: state = velocity only -> exact expectation
TARGET_ERROR = 1e-8           # below the floor on purpose; the stagnation stop ends it
VISCOSITY = 1.0
DENSITY = 1.0
TOLERANCE = 0.15              # accept N1+N2 within +/-15% of N_total

BASE = os.path.join("tests", "unit", "resources", "numerical_solved", "netcdf")


def block_mean(a, f):
    W, H, D = a.shape
    return a.reshape(W // f, f, H // f, f, D // f, f).mean(axis=(1, 3, 5))


def load_pore(code, f):
    bf = netcdf_file(f"{BASE}/BIN_Bentheimer{code}.nc", "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    pore = np.transpose(geom == 0, (1, 2, 0))          # 0 = pore; flow x -> z
    return block_mean(pore.astype(np.float64), f) >= 0.5


def make_solver(pore, max_iterations, seed=None, stagnation_window=25):
    kw = dict(
        volume=pore, scale=1.0, viscosity=VISCOSITY, density=DENSITY,
        target_error=TARGET_ERROR, max_iterations=max_iterations,
        predictor=PREDICTOR, stagnation_window=stagnation_window,
    )
    if seed is not None:
        p0, (u0, v0, w0) = seed
        kw["initial_pressure"] = p0
        kw["initial_velocity"] = (u0, v0, w0)
    return StokesSolver(**kw)


def mid_conductivity(w, shape):
    Wd, Hd, Dd = shape
    mid = Dd // 2
    wc = 0.5 * (w[:, :, mid] + w[:, :, mid + 1])
    return wc.sum() * Dd / (Wd * Hd)


def rel_l2(a, b):
    d = a - b
    denom = np.sqrt((b * b).sum())
    return np.sqrt((d * d).sum()) / denom if denom > 0 else np.inf


def main():
    pore = load_pore("000", DOWNSCALE_FACTOR)
    shape = pore.shape
    print(f"geometry: f={DOWNSCALE_FACTOR} shape={shape} fluid={int(pore.sum())} "
          f"predictor={PREDICTOR}")

    # --- 1. cold reference to convergence ------------------------------------
    ref = make_solver(pore, max_iterations=20000)
    r_ref = ref.solve()
    n_total = r_ref["iterations"]
    w_ref = ref.w.copy()
    cond_ref = mid_conductivity(w_ref, shape)
    print(f"\n[1] cold reference : N_total={n_total} iters  stop={ref.stop_reason}  "
          f"converged={r_ref['converged']}  cond={cond_ref:.6e}")

    # --- 2. cold, capped at one third (stagnation OFF so it runs exactly N1) --
    n1 = max(1, n_total // 3)
    partial = make_solver(pore, max_iterations=n1, stagnation_window=0)
    r_partial = partial.solve()
    seed = (
        partial.p.copy(),
        (partial.u.copy(), partial.v.copy(), partial.w.copy()),
    )
    print(f"\n[2] cold, capped   : ran N1={r_partial['iterations']} iters "
          f"(=N_total//3)  cond={mid_conductivity(partial.w, shape):.6e}")

    # --- 3. warm start from the partial fields -------------------------------
    warm = make_solver(pore, max_iterations=20000, seed=seed)
    r_warm = warm.solve()
    n2 = r_warm["iterations"]
    cond_warm = mid_conductivity(warm.w, shape)
    print(f"\n[3] warm from [2]  : N2={n2} iters  stop={warm.stop_reason}  "
          f"converged={r_warm['converged']}  cond={cond_warm:.6e}")

    # --- verdict -------------------------------------------------------------
    total = n1 + n2
    ratio = total / n_total
    field_match = rel_l2(warm.w, w_ref)
    print("\n" + "=" * 66)
    print(f"N_total (cold)            : {n_total}")
    print(f"N1 (=N_total//3)          : {n1}")
    print(f"N2 (warm to convergence)  : {n2}")
    print(f"N1 + N2                   : {total}   ({ratio:.2f} x N_total)")
    print(f"final field relL2 vs ref  : {field_match:.2e}")
    print(f"final cond vs ref         : {abs(cond_warm - cond_ref) / abs(cond_ref):.3%}")
    print("-" * 66)
    ok_work = abs(ratio - 1.0) <= TOLERANCE
    ok_field = field_match < 1e-3
    if ok_work and ok_field:
        print("PASS: warm start resumes correctly (N1+N2 ~= N_total, same field).")
    elif not ok_field:
        print("FAIL: warm-started run converged to a DIFFERENT field (seed corrupts state).")
    elif ratio > 1.0 + TOLERANCE:
        print(f"FAIL: warm start wasted work (N1+N2 = {ratio:.2f} x N_total); the seed "
              "was likely dropped/re-zeroed and the field re-developed.")
    else:
        print(f"NOTE: N1+N2 = {ratio:.2f} x N_total (< expected). Inspect: the "
              "stagnation stop may be firing at a different point for the two runs.")


if __name__ == "__main__":
    main()
