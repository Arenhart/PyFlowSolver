"""High-resolution Stokes vs low-resolution Brinkman on a Bentheimer sample.

Validation-in-spirit for the multiscale (Darcy-Stokes-Brinkman) model: resolve a
pore-scale geometry with Stokes, then check that solving the *downscaled* image
with the Brinkman model -- using a subresolution radius distribution estimated
from the high-res footprint (local-thickness) field -- recovers the same
block-averaged velocity and pressure. This is a MODEL comparison, so only loose
agreement is expected (there is no exact answer).

This is a benchmark/exploration script, not a unit test: high-resolution Stokes
is expensive, so keep `HR_SIZE` tractable. Run:

    python scripts/benchmark_brinkman_multiscale.py

Convention (see the numerical_solved dataset): BIN_*.nc stores 0 = pore, 1 =
solid; flow is driven along the array's last axis here (our solvers drive z).
"""

import os
import sys
import time

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.stokesSolver import StokesSolver
from pyflowsolver.brinkmanSolver import BrinkmanSolver
from pyflowsolver.multiscaleVolumeManager import REGION_SUBRES_FIRST
from pyflowsolver.tubeBundle import (
    EmpiricalTubeRadiusDistribution, TruncatedGaussianTubeDistribution)
from pyflowsolver.fastLaplacian import _calculate_footprint
from pyedt import edt

# ----- knobs (keep HR_SIZE tractable: HR^3 Stokes is the cost) -------------- #
BIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "tests", "unit", "resources", "numerical_solved",
                        "netcdf", "BIN_Bentheimer000.nc")
HR_SIZE = 250        # high-res crop side (voxels)
DOWNSCALE = 5       # low-res voxel = DOWNSCALE^3 high-res voxels
SEED = 0

# Subresolution radius statistics (high-res voxel units), assumed Gaussian and
# truncated to [R_MIN, DOWNSCALE]: after downsampling by F the voxel is F wide,
# so a pore of radius up to ~F can fall between voxels (i.e. be unresolved).
R_MEAN, R_STD, R_MIN = 4.25, 2.86, 1.0


def load_pore_crop(path, size):
    bf = netcdf_file(path, "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data)
    bf.close()
    pore = (geom == 0)                       # 0 = pore
    # Centred crop (the sample corners are less representative of bulk porosity).
    o = [(n - size) // 2 for n in pore.shape]
    crop = pore[o[0]:o[0] + size, o[1]:o[1] + size, o[2]:o[2] + size]
    return crop.astype(np.float64)


def cell_centered_speed(res):
    """|velocity| at cell centres from a MAC (u, v, w) result dict."""
    u, v, w = res["u"], res["v"], res["w"]
    uc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
    vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)


def block_mean(a, f):
    w, h, d = a.shape
    return a.reshape(w // f, f, h // f, f, d // f, f).mean(axis=(1, 3, 5))


def radius_fields(pore_mask):
    """Two candidate per-voxel radius fields (high-res units):

    - ``edt``: distance-to-solid = the LOCAL pore radius. Physically the right
      tube radius for a subresolution throat.
    - ``footprint``: local thickness = radius of the largest inscribed sphere
      covering the voxel. Correct for the open-voxel Arns/Stokes conductance, but
      for subresolution voxels it BLEEDS the size of adjacent open pores in, so it
      badly overestimates the subres tube radius (and K ~ r^4 amplifies it).
    """
    e = edt(pore_mask.astype(np.uint8), scale=(1.0, 1.0, 1.0),
            force_method="cpu").astype(np.float32)
    fp = np.zeros_like(e)
    _calculate_footprint(e, fp, spacing=(1.0, 1.0, 1.0))
    return {"edt": e, "footprint": fp}


def rel_l2(a, b):
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else np.nan


def high_res_stokes(pore_hr):
    """Load the cached high-res Stokes reference from precompute_stokes (matching
    this crop size), or solve and cache it if absent. Returns MAC fields + the
    pore mask actually solved (so downscaling is consistent)."""
    from scripts.precompute_stokes import cache_path, load_stokes, CACHE_DIR

    expected_u = (pore_hr.shape[0] + 1, pore_hr.shape[1], pore_hr.shape[2])
    for crop in (None, HR_SIZE):                     # full-volume cache, then crop
        p = cache_path(crop)
        if os.path.exists(p):
            r = load_stokes(p)
            if r["u"].shape == expected_u:
                print(f"loaded cached Stokes: {os.path.basename(p)}  "
                      f"K_eff={r['meta'].get('k_eff'):.4g}", flush=True)
                return ({k: r[k].astype(np.float64) for k in ("u", "v", "w", "p")},
                        r["pore"].astype(np.float64))

    t0 = time.time()
    hr = StokesSolver(pore_hr, scale=1.0, predictor="implicit",
                      fast_laplacian_guess=True, max_iterations=20000,
                      target_error=1e-7).solve()
    print(f"high-res Stokes: converged={hr['converged']} "
          f"iters={hr['iterations']} ({time.time() - t0:.1f}s)", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    crop = None if pore_hr.shape[0] >= 250 else HR_SIZE
    np.savez_compressed(cache_path(crop),
                        u=hr["u"].astype(np.float32), v=hr["v"].astype(np.float32),
                        w=hr["w"].astype(np.float32), p=hr["p"].astype(np.float32),
                        pore=pore_hr.astype(np.uint8),
                        meta=np.array({"crop": crop, "k_eff": float("nan")}, dtype=object))
    return hr, pore_hr


def solve_low_res(porosity_lr, dist):
    br = BrinkmanSolver(porosity_map=porosity_lr, scale=float(DOWNSCALE),
                        distributions={REGION_SUBRES_FIRST: dist},
                        predictor="implicit", fast_laplacian_guess=False,
                        max_iterations=20000, target_error=1e-7)
    return br.solve()


def main():
    print(f"loading {HR_SIZE}^3 crop, downscale x{DOWNSCALE} "
          f"-> {HR_SIZE // DOWNSCALE}^3 low-res\n", flush=True)
    pore_hr = load_pore_crop(BIN_PATH, HR_SIZE)
    hr, pore_hr = high_res_stokes(pore_hr)           # pore from cache = solved geom
    print(f"high-res porosity = {pore_hr.mean():.3f}", flush=True)

    porosity_lr = block_mean(pore_hr, DOWNSCALE)
    subres = (porosity_lr > 0.0) & (porosity_lr < 1.0)
    print(f"low-res: solid={(porosity_lr == 0).sum()} open={(porosity_lr == 1).sum()} "
          f"subres={subres.sum()} of {porosity_lr.size}", flush=True)

    subres_hr = np.repeat(np.repeat(np.repeat(subres, DOWNSCALE, 0),
                                    DOWNSCALE, 1), DOWNSCALE, 2)
    fields = radius_fields(pore_hr)

    # reference (downscaled Stokes)
    speed_ref = block_mean(cell_centered_speed(hr), DOWNSCALE)
    p_ref = block_mean(hr["p"], DOWNSCALE)
    print(f"\nreference mean speed (Stokes downscaled) = {speed_ref.mean():.4g}")

    # candidate subresolution radius distributions
    dists = {}
    for source in ("footprint", "edt"):
        r = fields[source]
        radii = r[(r > 0) & subres_hr]
        if radii.size == 0:
            radii = r[r > 0]
        dists[source] = EmpiricalTubeRadiusDistribution(radii, seed=SEED)
    # supplied throat-ish statistics: Gaussian truncated to [R_MIN, resolution=F]
    dists["stats(gauss<F)"] = TruncatedGaussianTubeDistribution(
        R_MEAN, R_STD, R_MIN, float(DOWNSCALE), seed=SEED)

    for name, dist in dists.items():
        s = np.asarray(dist.sample(20000, rng=np.random.default_rng(1)))
        lr = solve_low_res(porosity_lr, dist)
        speed_lr = cell_centered_speed(lr)
        print(f"\n[{name}] effective radii: mean={s.mean():.2f} "
              f"range=[{dist.min():.2f},{dist.max():.2f}]")
        print(f"[{name}] speed rel-L2={rel_l2(speed_lr, speed_ref):.3f}  "
              f"press rel-L2={rel_l2(lr['p'], p_ref):.3f}  "
              f"mean speed={speed_lr.mean():.4g} "
              f"(x{speed_lr.mean() / speed_ref.mean():.2f} vs ref)")


if __name__ == "__main__":
    main()
