"""Precompute and cache a high-resolution Stokes solution on a Bentheimer volume.

The high-resolution Stokes solve is expensive (the full 250^3 takes ~hours), so
run this ONCE and let the multiscale/throat benchmarks load the cached fields
instead of re-solving. Fields are stored as float32, compressed.

Usage:
    python scripts/precompute_stokes.py            # full volume (BIN_NAME)
    python scripts/precompute_stokes.py 150         # centred 150^3 crop
    python scripts/precompute_stokes.py 150 force   # overwrite an existing cache

Load in a benchmark:
    from scripts.precompute_stokes import load_stokes
    r = load_stokes(cache_path(crop=150))
    u, v, w, p = r["u"], r["v"], r["w"], r["p"]      # MAC fields
    meta = r["meta"]                                  # porosity, scale, k_eff, ...
"""

import os
import sys
import time

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.stokesSolver import StokesSolver

_HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(_HERE, os.pardir, "tests", "unit",
                            "resources", "numerical_solved", "netcdf")
CACHE_DIR = os.path.join(_HERE, "stokes_cache")

BIN_NAME = "BIN_Bentheimer000.nc"   # 0 = pore, 1 = solid; flow driven along z
SCALE = 1.0
TARGET_ERROR = 1e-7
MAX_ITERATIONS = 40000


def load_pore(bin_name, crop):
    """Pore mask (1 = pore) from a BIN_*.nc file; centred crop or full volume."""
    bf = netcdf_file(os.path.join(RESOURCE_DIR, bin_name), "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data)
    bf.close()
    pore = (geom == 0)
    if crop is not None:
        o = [(n - crop) // 2 for n in pore.shape]
        pore = pore[o[0]:o[0] + crop, o[1]:o[1] + crop, o[2]:o[2] + crop]
    return pore.astype(np.float64)


def cache_path(crop, bin_name=BIN_NAME):
    tag = "full" if crop is None else f"crop{crop}"
    stem = bin_name.replace(".nc", "")
    return os.path.join(CACHE_DIR, f"stokes_{stem}_{tag}.npz")


def measured_keff(u, v, w, p, scale):
    """Convention-free effective permeability: superficial velocity over the
    measured pressure gradient (mu = nu*rho = 1 here)."""
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    u_super = float(wc.mean())
    nz = p.shape[2]
    fluid = p != 0.0
    pz = np.array([p[:, :, k][fluid[:, :, k]].mean() if fluid[:, :, k].any() else np.nan
                   for k in range(nz)])
    z = np.arange(nz) * scale
    ok = np.isfinite(pz)
    if ok.sum() < 2:
        return np.nan
    G = -np.polyfit(z[ok], pz[ok], 1)[0]
    return u_super * scale / G if G > 0 else np.nan   # *scale = physical mu-less factor


def load_stokes(path):
    """Load a cached solution: {u, v, w, p, pore, meta}. meta is a plain dict."""
    z = np.load(path, allow_pickle=True)
    r = {k: z[k] for k in ("u", "v", "w", "p", "pore")}
    r["meta"] = z["meta"].item()
    return r


def bin_images():
    """All BIN_*.nc geometry files in the resource directory, sorted."""
    import glob
    return sorted(glob.glob(os.path.join(RESOURCE_DIR, "BIN_*.nc")))


def precompute_one(bin_name, crop=None, force=False):
    """Solve + cache one image. Returns 'skipped' | 'done'. The write is atomic
    (temp file + os.replace), so an interrupted run never leaves a half-written
    cache -- a restart safely resumes from the completed ones."""
    out = cache_path(crop, bin_name)
    if os.path.exists(out) and not force:
        print(f"[skip] {bin_name}: cache exists ({os.path.basename(out)})", flush=True)
        return "skipped"

    pore = load_pore(bin_name, crop)
    label = "full" if crop is None else f"{crop}^3 crop"
    print(f"[solve] {bin_name}: {label} {pore.shape} porosity={pore.mean():.3f}",
          flush=True)

    t0 = time.time()
    res = StokesSolver(pore, scale=SCALE, predictor="implicit",
                       fast_laplacian_guess=True, max_iterations=MAX_ITERATIONS,
                       target_error=TARGET_ERROR).solve()
    dt = time.time() - t0
    keff = measured_keff(res["u"], res["v"], res["w"], res["p"], SCALE)
    print(f"        converged={res['converged']} iters={res['iterations']} "
          f"residual={res['residual']:.2e} K_eff={keff:.5g} ({dt:.1f}s)", flush=True)

    meta = dict(bin_name=bin_name, crop=crop, shape=tuple(pore.shape),
                scale=SCALE, porosity=float(pore.mean()),
                iterations=int(res["iterations"]), converged=bool(res["converged"]),
                residual=float(res["residual"]), k_eff=float(keff),
                solve_seconds=float(dt))

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = out[:-4] + ".tmp.npz"          # out ends in ".npz"; keep .npz so savez
    np.savez_compressed(                 # writes exactly `tmp` (no auto-append)
        tmp,
        u=res["u"].astype(np.float32), v=res["v"].astype(np.float32),
        w=res["w"].astype(np.float32), p=res["p"].astype(np.float32),
        pore=pore.astype(np.uint8), meta=np.array(meta, dtype=object))
    os.replace(tmp, out)                 # atomic: half-written runs never land here
    print(f"        saved -> {os.path.basename(out)} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)
    return "done"


def main():
    """Batch-precompute Stokes for EVERY BIN_*.nc image (full volume by default),
    skipping any already cached so the run is restartable.

        python scripts/precompute_stokes.py          # all images, full volume
        python scripts/precompute_stokes.py 150       # all images, 150^3 crop
        python scripts/precompute_stokes.py force      # recompute all (overwrite)
    """
    crop = None
    force = False
    for arg in sys.argv[1:]:
        if arg.lower() == "force":
            force = True
        elif arg.lower() != "full":
            crop = int(arg)

    images = bin_images()
    print(f"{len(images)} images -> cache dir {CACHE_DIR}"
          f"{'  (force overwrite)' if force else ''}\n", flush=True)
    counts = {"done": 0, "skipped": 0, "error": 0}
    for i, path in enumerate(images, 1):
        name = os.path.basename(path)
        print(f"--- [{i}/{len(images)}] {name} ---", flush=True)
        try:
            counts[precompute_one(name, crop, force)] += 1
        except Exception as e:               # keep going; a restart retries this one
            counts["error"] += 1
            print(f"[error] {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\ndone={counts['done']} skipped={counts['skipped']} "
          f"errors={counts['error']}", flush=True)


if __name__ == "__main__":
    main()
