import os
import time

import h5py
import numpy as np
from scipy.io import netcdf_file

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver
from pyflowsolver.multigridSolver import MultigridSolver

np.set_printoptions(linewidth=200)
RADIUS = 20  # in voxel
LENGTH = 20  # in voxel
SCALE = 0.00105  # mm
RUN_DUCT = True
RUN_PYAMG = False
# Bentheimer coarsening sweep. The volume is 250^3, so 250/f must be an integer:
#   f=10 -> 25^3, f=5 -> 50^3, f=2 -> 125^3, f=1 -> 250^3 (hours).
BENTHEIMER_FS = [10, 5, 2, 1]
STOKES_TARGET_ERROR = 1e-4      # explicit residual floor ~1e-7; stagnation stop is the backstop

BASE = os.path.join("tests", "unit", "resources", "numerical_solved", "netcdf")


def make_circular_duct(radius, length, margin=2):
    """Float voxel volume for a z-aligned circular duct (1.0 fluid, 0.0 wall)."""
    n = int(np.ceil(2 * radius)) + 2 * margin
    volume = np.zeros((n, n, length), dtype=np.float64)
    center = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    disk = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    volume[disk, :] = 1.0
    return volume


def block_mean(a, f=1):
    W, H, D = a.shape
    return a.reshape(W // f, f, H // f, f, D // f, f).mean(axis=(1, 3, 5))


def load_coarse(code, f=1):
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


def harmonic_face(c_lo, c_hi):
    """Face conductivity 2/(1/c_lo + 1/c_hi) where both cells are fluid, else 0."""
    both = (c_lo > 0) & (c_hi > 0)
    k = np.zeros_like(c_lo)
    k[both] = 2.0 / (1.0 / c_lo[both] + 1.0 / c_hi[both])
    return k


def solve_fast_laplacian(pore, scale, enhanced=True):
    """Enhanced/standard fast-Laplacian (Arns) solve via the ALGEBRAIC-MULTIGRID
    solver (not diagonal PCG -- MG solves this Laplacian in a handful of iterations
    instead of thousands, cutting the FL solve from ~tens of seconds to seconds).

    Returns (vm, pressure, flow, permeability); vm.volume holds the Arns
    conductivity per voxel.
    """
    poremap = (np.asarray(pore) * 100.0).astype(np.float32)
    vm = VolumeManager(poremap, scale=scale)
    vm.convert_pore_volume_to_laplacian_conductivity(enhanced_model=enhanced)
    a_sparse, b = vm.get_sparse_system_jit()
    solver = MultigridSolver(backend="native")
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()
    solution, _, _ = solver.solve_pcg()
    pressure = np.asarray(vm.ravel_sparse_solution(solution), dtype=np.float64)
    flow, permeability = vm.get_conductivity(pressure)
    return vm, pressure, flow, permeability


def darcy_velocity_guess(pressure, conductivity, scale, velocity_scale=0.8):
    """Algebraic k*grad(p) warm start (used for the duct / channels).

    The MAC face velocities are the per-face Darcy flux v = k_face*grad(p); the
    inlet/outlet z-faces are copied from the interior. `velocity_scale` corrects
    the enhanced-Arns magnitude overshoot (~1/0.8). NOTE: on COMPLEX media this
    seed does not reduce the Stokes iteration count -- the viscous seed below does
    (it is what compare_bentheimer uses); k*grad(p) is kept for channels.
    """
    p = np.asarray(pressure, dtype=np.float64)
    c = np.asarray(conductivity, dtype=np.float64)
    W, H, D = p.shape
    s = velocity_scale / scale
    u = np.zeros((W + 1, H, D)); v = np.zeros((W, H + 1, D)); w = np.zeros((W, H, D + 1))
    u[1:W, :, :] = harmonic_face(c[:-1, :, :], c[1:, :, :]) * (p[:-1, :, :] - p[1:, :, :]) * s
    v[:, 1:H, :] = harmonic_face(c[:, :-1, :], c[:, 1:, :]) * (p[:, :-1, :] - p[:, 1:, :]) * s
    w[:, :, 1:D] = harmonic_face(c[:, :, :-1], c[:, :, 1:]) * (p[:, :, :-1] - p[:, :, 1:]) * s
    return p, (u, v, w)


def stokes_conductivity(solver):
    """Mid-plane axial permeability from a solved StokesSolver."""
    w = solver.w
    W, H, D = solver.volume.shape
    mid = D // 2
    wc = 0.5 * (w[:, :, mid] + w[:, :, mid + 1])
    sc = solver.scale
    flux = wc.sum() * (sc[0] * sc[1])
    return flux * D * sc[2] / (W * sc[0] * H * sc[1])


# ====================================================================== #
# Circular duct (channel): implicit predictor + k*grad(p) warm start
# ====================================================================== #
if RUN_DUCT:
    vol = make_circular_duct(radius=RADIUS, length=LENGTH, margin=2)
    W, H, D = vol.shape
    expected_flow = np.pi * (RADIUS * SCALE) ** 4 / (8 * LENGTH * SCALE)
    expected_permeability = expected_flow * (LENGTH * SCALE) / (W * SCALE * H * SCALE)
    print("Circular duct:")
    print("Expected flow: ", expected_flow, "   Expected perm: ", expected_permeability)

    _, _, flow_e, perm_e = solve_fast_laplacian(vol, SCALE, enhanced=True)
    print("\nEnhanced Fast Laplacian:  Flow:", flow_e, "  Perm:", perm_e)
    vm_s, p_s, flow_s, perm_s = solve_fast_laplacian(vol, SCALE, enhanced=False)
    print("Standard Fast Laplacian:  Flow:", flow_s, "  Perm:", perm_s)

    p_guess, vel_guess = darcy_velocity_guess(p_s, vm_s.volume, SCALE)
    for backend in (["native"] + (["pyamg"] if RUN_PYAMG else [])):
        ss = StokesSolver(volume=vol, scale=SCALE, viscosity=1.0, density=1.0,
                          target_error=1e-8, max_iterations=20000,
                          predictor="implicit", backend=backend,
                          initial_pressure=p_guess, initial_velocity=vel_guess)
        res = ss.solve()
        print(f"Stokes ({backend}): conductivity {stokes_conductivity(ss):.6e}  "
              f"iters {res['iterations']}")


# ====================================================================== #
# Bentheimer (complex media): explicit predictor, cold vs viscous warm start,
# swept over coarsening factors. Reports TOTAL wall time (the target metric).
# ====================================================================== #
def compare_bentheimer(f):
    print("\n" + "#" * 70)
    print(f"# Bentheimer 000  f={f}")
    print("#" * 70)
    pore, vel = load_coarse("000", f=f)
    W, H, D = pore.shape
    scale = 1.0
    mid = D // 2
    of_flux = vel[:, :, mid].sum() * scale * scale
    of_perm = of_flux * D / (W * H)
    print(f"shape={pore.shape} fluid={int(pore.sum())} porosity={pore.mean():.3f}")
    print(f"OpenFOAM permeability   : {of_perm:.6e}")

    # Fast-Laplacian (multigrid) -- shared warm-start setup.
    t0 = time.perf_counter()
    vm, p_fl, fl_flow, fl_perm = solve_fast_laplacian(pore, scale, enhanced=True)
    t_fl = time.perf_counter() - t0
    print(f"Enhanced FL permeability: {fl_perm:.6e}   (MG solve {t_fl:.1f}s)")

    # Cold Stokes (baseline: no setup; opt out of the default FL guess).
    t0 = time.perf_counter()
    cold = StokesSolver(volume=pore, scale=scale, viscosity=1.0, density=1.0,
                        target_error=STOKES_TARGET_ERROR, max_iterations=20000,
                        predictor="explicit", fast_laplacian_guess=False)
    rc = cold.solve()
    t_cold = time.perf_counter() - t0
    perm_cold = stokes_conductivity(cold)

    # Warm Stokes: the promoted library feature derives the seed velocity from the
    # FL pressure internally (StokesSolver initial_velocity="from_pressure"), so
    # the seed build is folded into the warm solve time.
    t0 = time.perf_counter()
    warm = StokesSolver(volume=pore, scale=scale, viscosity=1.0, density=1.0,
                        target_error=STOKES_TARGET_ERROR, max_iterations=20000,
                        predictor="explicit",
                        initial_pressure=p_fl,
                        initial_velocity=StokesSolver.SEED_FROM_PRESSURE)
    rw = warm.solve()
    t_warm = time.perf_counter() - t0
    perm_warm = stokes_conductivity(warm)

    total_warm = t_fl + t_warm
    print(f"\n  {'strategy':22} {'iters':>6} {'solve(s)':>9} {'total(s)':>9} {'perm':>13}")
    print(f"  {'cold (baseline)':22} {rc['iterations']:>6} {t_cold:>9.1f} {t_cold:>9.1f} {perm_cold:>13.6e}")
    print(f"  {'warm (from_pressure)':22} {rw['iterations']:>6} {t_warm:>9.1f} {total_warm:>9.1f} {perm_warm:>13.6e}")
    print(f"    warm total = FL {t_fl:.1f}s + warm solve {t_warm:.1f}s (seed folded into solve)")
    if total_warm > 0:
        print(f"    TOTAL wall time: cold {t_cold:.1f}s  vs  warm {total_warm:.1f}s"
              f"  -> {t_cold / total_warm:.2f}x  (iters {rc['iterations']} -> {rw['iterations']})")


for _f in BENTHEIMER_FS:
    compare_bentheimer(_f)
