"""Compare Arns (original & enhanced) and the Stokes solver to OpenFOAM.

For each Bentheimer sample that has an OpenFOAM solution, we:
  - load the binary geometry (BIN_*.nc; 0 = pore, 1 = solid) and the OpenFOAM
    velocity magnitude (solved_openfoam/*.nc, variable "float");
  - transpose so the flow axis (x) becomes z (our solvers drive z);
  - downscale by DOWNSCALE (block-mean; geometry thresholded at 0.5, velocity
    pore-weighted) to a tractable size;
  - build the original and enhanced Arns conductivity -> Darcy velocity, and run
    the Stokes solver;
  - compare each predicted |v| to the (downscaled) OpenFOAM |v| over the pore
    voxels, after normalizing each field to unit mean (magnitudes are in
    different unit systems). Metrics: Pearson correlation, cosine similarity,
    relative L2 error.
"""

import os
import sys
import time

import numpy as np
import h5py
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyflowsolver.fastLaplacian import fast_laplacian_volume_generator
from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "tests", "unit", "resources", "numerical_solved", "netcdf")
DOWNSCALE = 5
STOKES_TOL = 1e-4
STOKES_MAXIT = 30000

CODES = ["000", "002", "020", "022", "111", "113", "131", "133",
         "200", "202", "220", "222", "311", "313", "331", "333"]


def block_mean(a, f):
    W, H, D = a.shape
    return a.reshape(W // f, f, H // f, f, D // f, f).mean(axis=(1, 3, 5))


def load_coarse(code, f):
    bf = netcdf_file(f"{BASE}/BIN_Bentheimer{code}.nc", "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    hf = h5py.File(f"{BASE}/solved_openfoam/Bentheimer_DRP_{code}_OpenFOAM.nc", "r")
    vel = hf["float"][:].astype(np.float64); hf.close()
    pore = np.transpose(geom == 0, (1, 2, 0))   # 0 = pore; flow x -> z
    vel = np.transpose(vel, (1, 2, 0))
    coarse_pore = block_mean(pore.astype(np.float64), f) >= 0.5
    W, H, D = vel.shape
    vp = (vel * pore).reshape(W // f, f, H // f, f, D // f, f).sum(axis=(1, 3, 5))
    cnt = pore.astype(np.float64).reshape(W // f, f, H // f, f, D // f, f).sum(axis=(1, 3, 5))
    coarse_vel = np.zeros_like(vp); m = cnt > 0; coarse_vel[m] = vp[m] / cnt[m]
    return coarse_pore, coarse_vel


def _hface(a, b):
    both = (a > 0) & (b > 0); k = np.zeros_like(a)
    k[both] = 2.0 / (1.0 / a[both] + 1.0 / b[both]); return k


def _cell_speed(u, v, w):
    uc = 0.5 * (u[:-1] + u[1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:]); wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)


def arns_speed(pore, enhanced):
    por_map = pore.astype(np.uint8) * 100
    cond = np.asarray(fast_laplacian_volume_generator(
        por_map, np.array([1.0, 1.0, 1.0]), closed_border=False, enhanced_model=enhanced), np.float64)
    vm = VolumeManager(cond.copy(), scale=1.0)
    A, b = vm.get_sparse_system_jit()
    s = DarcySolver(target_error=1e-6, max_iterations=100000)
    s.set_linear_system(A, b); s.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = s.solve_pcg()
    p = np.asarray(vm.ravel_sparse_solution(x)); c = np.asarray(vm.volume)
    W, H, D = pore.shape
    u = np.zeros((W + 1, H, D)); v = np.zeros((W, H + 1, D)); w = np.zeros((W, H, D + 1))
    u[1:W] = _hface(c[:-1], c[1:]) * (p[:-1] - p[1:])
    v[:, 1:H] = _hface(c[:, :-1], c[:, 1:]) * (p[:, :-1] - p[:, 1:])
    w[:, :, 1:D] = _hface(c[:, :, :-1], c[:, :, 1:]) * (p[:, :, :-1] - p[:, :, 1:])
    return _cell_speed(u, v, w)


def stokes_speed(pore):
    vm = VolumeManager(pore.astype(np.float64), scale=1.0)
    s = StokesSolver(vm, target_error=STOKES_TOL, max_iterations=STOKES_MAXIT)
    r = s.solve()
    return _cell_speed(r["u"], r["v"], r["w"]), r["iterations"], r["converged"]


def metrics(pred, ref, mask):
    a = pred[mask].astype(np.float64); b = ref[mask].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, np.nan
    a = a / a.mean(); b = b / b.mean()
    corr = np.corrcoef(a, b)[0, 1]
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    return corr, cos, rel


def main():
    # JIT warmup on a small percolating duct.
    warm = np.zeros((20, 20, 20), bool); warm[7:13, 7:13, :] = True
    arns_speed(warm, True); stokes_speed(warm)

    rows = []
    print(f"downscale factor {DOWNSCALE}  (250 -> {250 // DOWNSCALE})\n", flush=True)
    hdr = f"{'code':>5} | {'orig corr/cos/relL2':>22} | {'enh corr/cos/relL2':>22} | {'stokes corr/cos/relL2':>24} | stokes_it"
    print(hdr, flush=True)
    for code in CODES:
        pore, vel = load_coarse(code, DOWNSCALE)
        mask = pore
        o = metrics(arns_speed(pore, False), vel, mask)
        e = metrics(arns_speed(pore, True), vel, mask)
        t0 = time.perf_counter()
        sp, it, conv = stokes_speed(pore)
        s = metrics(sp, vel, mask)
        rows.append((code, o, e, s))
        print(f"{code:>5} | {o[0]:5.3f} {o[1]:5.3f} {o[2]:5.3f} | "
              f"{e[0]:5.3f} {e[1]:5.3f} {e[2]:5.3f} | "
              f"{s[0]:5.3f} {s[1]:5.3f} {s[2]:5.3f} | {it}{'' if conv else '*'} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    arr = lambda i: np.array([[r[i][j] for j in range(3)] for r in rows])
    for name, i in (("original", 1), ("enhanced", 2), ("stokes", 3)):
        m = np.nanmean(arr(i), axis=0)
        print(f"\nMEAN {name:>9}: corr={m[0]:.3f} cosine={m[1]:.3f} relL2={m[2]:.3f}", flush=True)


if __name__ == "__main__":
    main()
