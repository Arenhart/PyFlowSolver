"""Run the Stokes solver on a circular duct and save speed/pressure volumes.

Two purposes:
  1. Analytic check: the steady z-velocity in a pressure-driven pipe must match
     the Hagen-Poiseuille parabola  w(r) = (G / 4mu) (R^2 - r^2), with
     G = dp/dz = 1/L and mu = rho * nu. We fit w vs r^2 at a mid-duct slice and
     compare the slope to -G/(4mu).
  2. Visualization: the final cell-centered speed |u| and pressure fields are
     written as TIFF stacks (z, y, x order, float32) for a viewer.
"""

import os
import sys

import numpy as np
from tifffile import imwrite

# Ensure the repo root (not any stale installed copy) is imported.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pyflowsolver.stokesSolver import StokesSolver

RADIUS = 8
LENGTH = 16
SCALE = 1.0
VISCOSITY = 1.0
DENSITY = 1.0
TARGET_ERROR = 1e-6
MAX_ITERATIONS = 20000


def make_circular_duct(radius, length, margin=2):
    """Float voxel volume for a z-aligned circular duct (1.0 fluid, 0.0 wall)."""
    n = int(np.ceil(2 * radius)) + 2 * margin
    volume = np.zeros((n, n, length), dtype=np.float64)
    center = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    disk = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    volume[disk, :] = 1.0
    return volume


def cell_centered_velocity(u, v, w):
    """Average the MAC face velocities to cell centers -> (uc, vc, wc)."""
    uc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
    vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return uc, vc, wc


def save_zyx(array, filename):
    """Save a (w, h, d) volume as a (z, y, x) float32 TIFF stack in the repo root."""
    path = os.path.join(REPO_ROOT, filename)
    imwrite(path, np.transpose(array, (2, 1, 0)).astype(np.float32))
    print(f"  saved {path}  shape(z,y,x)={array.shape[::-1]}")


def main():
    volume = make_circular_duct(RADIUS, LENGTH)
    print(f"Circular duct: shape={volume.shape}, "
          f"fluid voxels={int((volume > 0).sum())}/{volume.size}")

    solver = StokesSolver(
        volume,
        scale=SCALE,
        viscosity=VISCOSITY,
        density=DENSITY,
        target_error=TARGET_ERROR,
        max_iterations=MAX_ITERATIONS,
    )
    result = solver.solve()
    print(f"Solve: converged={result['converged']}, "
          f"iterations={result['iterations']}, residual={result['residual']:.2e}")

    u, v, w, p = result["u"], result["v"], result["w"], result["p"]
    uc, vc, wc = cell_centered_velocity(u, v, w)
    speed = np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)
    fluid = volume > 0
    speed *= fluid  # zero the solid so viewers show a clean void

    # ---------------------------------------------------------------- #
    # Analytic Hagen-Poiseuille check at a mid-duct cross-section
    # ---------------------------------------------------------------- #
    W, H, D = volume.shape
    k = D // 2
    fluid_slice = fluid[:, :, k]
    c = (W - 1) / 2.0
    yy, xx = np.mgrid[0:W, 0:H]
    r2 = ((xx - c) ** 2 + (yy - c) ** 2)[fluid_slice]
    w_prof = wc[:, :, k][fluid_slice]

    A = np.vstack([r2, np.ones_like(r2)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, w_prof, rcond=None)
    pred = A @ np.array([slope, intercept])
    r_squared = 1 - ((w_prof - pred) ** 2).sum() / ((w_prof - w_prof.mean()) ** 2).sum()

    G = 1.0 / (LENGTH * SCALE)           # pressure drop per unit length
    mu = DENSITY * VISCOSITY             # dynamic viscosity
    analytic_slope = -G / (4.0 * mu)
    r_eff = np.sqrt(intercept / (-slope)) if slope < 0 else float("nan")

    print("\n--- Hagen-Poiseuille check (mid-duct slice) ---")
    print(f"  parabolic fit R^2      : {r_squared:.5f}")
    print(f"  fitted slope           : {slope:+.6f}")
    print(f"  analytic slope -G/(4mu): {analytic_slope:+.6f}")
    print(f"  relative slope error   : {abs(slope - analytic_slope) / abs(analytic_slope):.2%}")
    print(f"  w_max (numerical)      : {w_prof.max():.5f}")
    print(f"  effective radius       : {r_eff:.3f}  (nominal {RADIUS})")

    # ---------------------------------------------------------------- #
    # Save volumes for visualization
    # ---------------------------------------------------------------- #
    print("\n--- Saving volumes ---")
    save_zyx(speed, "stokes_duct_speed.tiff")
    save_zyx(p, "stokes_duct_pressure.tiff")


if __name__ == "__main__":
    main()
