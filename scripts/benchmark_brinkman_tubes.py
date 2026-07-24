"""Synthetic parallel-tube bundle: resolved Stokes vs analytical vs Brinkman.

The cleanest end-to-end check of the multiscale model, because the geometry IS a
bundle of parallel tubes -- exactly the model's assumption -- so the analytical
answer is exact:

    K_bundle = (pi/8) * sum_i r_i^4 / A        (parallel Hagen-Poiseuille)

which is identically what get_permeability_field computes. We build resolved
tubes (cylinders along the flow z), then compare:

  * K_bundle             -- analytical, from the nominal tube radii
  * K_eff (high-res Stokes)  -- resolves every tube; loses a little to voxelisation
  * K_eff (low-res Brinkman, nominal radii)     -- the model with exact radii
  * K_eff (low-res Brinkman, footprint radii)   -- radii ESTIMATED from the image
    (as the Bentheimer path does) -- expected to track the voxelised Stokes better

plus a block-averaged velocity/pressure field comparison. Exploration script:
prints a table, no assertions.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.stokesSolver import StokesSolver
from pyflowsolver.brinkmanSolver import BrinkmanSolver
from pyflowsolver.multiscaleVolumeManager import REGION_SUBRES_FIRST
from pyflowsolver.tubeBundle import (
    TruncatedGaussianTubeDistribution, EmpiricalTubeRadiusDistribution)
from pyflowsolver.fastLaplacian import _calculate_footprint
from pyedt import edt

W = H = 64          # high-res cross-section
D = 20              # flow-direction length (tubes are uniform along z)
F = 4               # downscale factor (isotropic) -> low-res W/F x H/F x D/F
TARGET_POROSITY = 0.35
SEED = 0
R_MEAN, R_SIG, R_LO, R_HI = 3.0, 0.7, 2.0, 4.5


def make_tubes(radius_dist, rng):
    """Non-overlapping circular tubes (extruded along z). Returns pore mask and
    the array of nominal radii actually placed."""
    pore = np.zeros((W, H, D), dtype=bool)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
    centers, radii = [], []
    attempts = 0
    while pore[:, :, 0].mean() < TARGET_POROSITY and attempts < 200000:
        attempts += 1
        r = float(radius_dist.sample(rng=rng))
        cx = rng.uniform(r + 1, W - r - 1)
        cy = rng.uniform(r + 1, H - r - 1)
        if any((cx - px) ** 2 + (cy - py) ** 2 < (r + pr + 1.5) ** 2
               for (px, py), pr in zip(centers, radii)):
            continue
        pore[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, :] = True
        centers.append((cx, cy))
        radii.append(r)
    return pore.astype(np.float64), np.asarray(radii)


def block_mean(a, f):
    w, h, d = a.shape
    return a.reshape(w // f, f, h // f, f, d // f, f).mean(axis=(1, 3, 5))


def cell_speed(res):
    u, v, w = res["u"], res["v"], res["w"]
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)


def footprint_radii(pore):
    e = edt(pore.astype(np.uint8), scale=(1.0, 1.0, 1.0), force_method="cpu")
    fp = np.zeros_like(e, dtype=np.float32)
    _calculate_footprint(e.astype(np.float32), fp, spacing=(1.0, 1.0, 1.0))
    return fp[fp > 0]


def rel_l2(a, b):
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n > 0 else np.nan


def measured_keff(res, scale, mu=1.0):
    """Convention-free K_eff: superficial velocity / measured pressure gradient.

    Removes the inlet/outlet length ambiguity (uses the actual dp/dz slope from
    the field) so high-res and low-res values are directly comparable despite
    different z-cell counts.
    """
    w, p = res["w"], res["p"]
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])          # cell-centred flow velocity
    u_super = float(wc.mean())                        # Q / A_total
    nz = p.shape[2]
    fluid = p != 0.0                                  # p==0 on solid/excluded
    pz = np.array([p[:, :, k][fluid[:, :, k]].mean() if fluid[:, :, k].any() else np.nan
                   for k in range(nz)])
    z = np.arange(nz) * scale
    ok = np.isfinite(pz)
    G = -np.polyfit(z[ok], pz[ok], 1)[0]              # -dp/dz
    return u_super * mu / G


def keff_brinkman(porosity_lr, dist):
    br = BrinkmanSolver(porosity_map=porosity_lr, scale=float(F),
                        distributions={REGION_SUBRES_FIRST: dist},
                        predictor="implicit", fast_laplacian_guess=False,
                        max_iterations=20000, target_error=1e-9)
    res = br.solve()
    return measured_keff(res, float(F)), res


def main():
    rng = np.random.default_rng(SEED)
    dist_nominal = TruncatedGaussianTubeDistribution(R_MEAN, R_SIG, R_LO, R_HI, seed=SEED)
    pore, radii = make_tubes(dist_nominal, rng)
    phi = pore[:, :, 0].mean()
    A = float(W * H)
    k_bundle = (np.pi / 8.0) * float(np.sum(radii ** 4)) / A
    print(f"tubes: n={radii.size} porosity={phi:.3f} "
          f"(~{radii.size / (W / F) / (H / F):.1f} tubes / low-res voxel) "
          f"r[min/mean/max]={radii.min():.2f}/{radii.mean():.2f}/{radii.max():.2f}")
    print(f"K_bundle (analytical, nominal radii) = {k_bundle:.5g}\n")

    # high-res resolved-tube Stokes
    t0 = time.time()
    hr = StokesSolver(pore, scale=1.0, predictor="implicit",
                      fast_laplacian_guess=True, max_iterations=20000,
                      target_error=1e-8).solve()
    k_hr = measured_keff(hr, 1.0)                                  # convention-free
    print(f"high-res Stokes: converged={hr['converged']} iters={hr['iterations']} "
          f"({time.time() - t0:.1f}s)  K_eff={k_hr:.5g}  "
          f"(ratio to bundle {k_hr / k_bundle:.3f})")

    # low-res porosity + two radius distributions
    porosity_lr = block_mean(pore, F)
    dist_emp = EmpiricalTubeRadiusDistribution(radii, seed=SEED)     # exact radii
    dist_fp = EmpiricalTubeRadiusDistribution(footprint_radii(pore), seed=SEED)

    k_lr_nom, lr = keff_brinkman(porosity_lr, dist_emp)
    k_lr_fp, lr_fp = keff_brinkman(porosity_lr, dist_fp)
    print(f"low-res Brinkman (nominal radii):   K_eff={k_lr_nom:.5g}  "
          f"(ratio to bundle {k_lr_nom / k_bundle:.3f}, to Stokes {k_lr_nom / k_hr:.3f})")
    print(f"low-res Brinkman (footprint radii): K_eff={k_lr_fp:.5g}  "
          f"(ratio to bundle {k_lr_fp / k_bundle:.3f}, to Stokes {k_lr_fp / k_hr:.3f})")

    # field comparison (footprint variant vs downscaled Stokes)
    speed_hr = block_mean(cell_speed(hr), F)
    p_hr = block_mean(hr["p"], F)
    print("\n--- block-averaged fields: Brinkman(footprint) vs Stokes-downscaled ---")
    print(f"speed  rel-L2 = {rel_l2(cell_speed(lr_fp), speed_hr):.3f}")
    print(f"press. rel-L2 = {rel_l2(lr_fp['p'], p_hr):.3f}")


if __name__ == "__main__":
    main()
