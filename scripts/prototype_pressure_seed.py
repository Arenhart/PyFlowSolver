"""Prototype: seed the Stokes velocity from the (excellent) FL pressure via a
viscous solve, instead of the algebraic k*grad(p) guess.

Diagnostics showed FL pressure is accurate (axial ~1%, fluctuations ~14%) while
the k*grad(p) velocity overshoots ~2x in the near-wall layer because it does not
enforce no-slip. The momentum equation with a FIXED pressure,

        -nu * laplacian(u) = -grad(p_FL)/rho  (+ body force),  u = 0 at walls,

is exactly the Stokes velocity that pressure implies -- it zeroes u at the walls
and gives the viscous profile. This turns the good pressure into a good velocity.

We reuse the solver's OWN tested machinery: the implicit predictor at large dt
solves (I - dt*nu*L) u* = u^n - (dt/rho) grad(p^n), which for u^n = 0 and large dt
reduces to the steady viscous solve above. So we set p = p_FL, u = 0, and take one
implicit predictor step.

Compares three seeds by (a) how close the SEED is to the converged velocity and
(b) iterations / wall-time to converge with the explicit solver:
    cold  |  k*grad(p) (current, ×s)  |  viscous-from-p_FL

Run it yourself (not auto-run):  python scripts/prototype_pressure_seed.py
"""

import os
import time
import numpy as np
from scipy.io import netcdf_file

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver

# ---- knobs -------------------------------------------------------------------
VOLUME_CODE = "000"
DOWNSCALE_FACTOR = 2
SCALE = 1.0
VISCOSITY = 1.0
DENSITY = 1.0
PREDICTOR = "explicit"          # the actual solve uses explicit (correct on complex media)
TARGET_ERROR = 1e-8
KGRADP_SCALE = 0.8              # the tuned magnitude for the k*grad(p) seed
BASE = os.path.join("tests", "unit", "resources", "numerical_solved", "netcdf")


def block_mean(a, f):
    W, H, D = a.shape
    return a.reshape(W // f, f, H // f, f, D // f, f).mean(axis=(1, 3, 5))


def load_pore(code, f):
    bf = netcdf_file(f"{BASE}/BIN_Bentheimer{code}.nc", "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    pore = np.transpose(geom == 0, (1, 2, 0))
    return block_mean(pore.astype(np.float64), f) >= 0.5


def harmonic_face(c_lo, c_hi):
    both = (c_lo > 0) & (c_hi > 0)
    k = np.zeros_like(c_lo)
    k[both] = 2.0 / (1.0 / c_lo[both] + 1.0 / c_hi[both])
    return k


def kgradp_velocity(pressure, conductivity, scale, vscale=1.0):
    """Algebraic Darcy-flux seed: v = vscale * k_face * grad(p)."""
    p = np.asarray(pressure, np.float64); c = np.asarray(conductivity, np.float64)
    W, H, D = p.shape
    s = vscale / scale
    u = np.zeros((W + 1, H, D)); v = np.zeros((W, H + 1, D)); w = np.zeros((W, H, D + 1))
    u[1:W] = harmonic_face(c[:-1], c[1:]) * (p[:-1] - p[1:]) * s
    v[:, 1:H] = harmonic_face(c[:, :-1], c[:, 1:]) * (p[:, :-1] - p[:, 1:]) * s
    w[:, :, 1:D] = harmonic_face(c[:, :, :-1], c[:, :, 1:]) * (p[:, :, :-1] - p[:, :, 1:]) * s
    fluid = c > 0
    w[:, :, 0] = w[:, :, 1] * fluid[:, :, 0]
    w[:, :, D] = w[:, :, D - 1] * fluid[:, :, D - 1]
    return u, v, w


def fast_laplacian(pore, scale):
    poremap = (pore * 100.0).astype(np.float32)
    vm = VolumeManager(poremap, scale=scale)
    vm.convert_pore_volume_to_laplacian_conductivity(enhanced_model=True)
    a, b = vm.get_sparse_system_jit()
    ds = DarcySolver(target_error=1e-9); ds.set_linear_system(a, b)
    ds.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = ds.solve_pcg()
    p = np.asarray(vm.ravel_sparse_solution(x), np.float64)
    return p, np.asarray(vm.volume, np.float64)


def viscous_velocity_from_pressure(pore, p_fl):
    """Solve -nu*laplacian(u) = -grad(p_FL)/rho with no-slip, reusing the solver's
    implicit diffusion machinery (one large-dt implicit predictor step from u=0)."""
    s = StokesSolver(volume=pore, scale=SCALE, viscosity=VISCOSITY, density=DENSITY,
                     predictor="implicit", initial_pressure=p_fl, max_iterations=1)
    s.create_velocity_arrays()            # self.p = p_FL, velocities = 0
    s._build_pressure_poisson_system()
    dt = s._compute_timestep()            # large (implicit_dt_factor default 1e6)
    s._build_diffusion_systems(dt)
    us, vs, ws = s._implicit_predictor_step(dt)   # -> viscous u from grad(p_FL)
    s._apply_velocity_boundary_conditions((us, vs, ws))   # open inlet/outlet
    return us.copy(), vs.copy(), ws.copy()


def cell_speed(u, v, w):
    uc = 0.5 * (u[:-1] + u[1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)


def stack(u, v, w):
    return np.concatenate([u.ravel(), v.ravel(), w.ravel()])


def rel_l2(a, b):
    d = a - b; n = np.sqrt((b * b).sum())
    return np.sqrt((d * d).sum()) / n if n > 0 else np.inf


def run_stokes(pore, seed):
    kw = dict(volume=pore, scale=SCALE, viscosity=VISCOSITY, density=DENSITY,
              target_error=TARGET_ERROR, max_iterations=20000, predictor=PREDICTOR)
    if seed is not None:
        p0, vel = seed
        kw.update(initial_pressure=p0, initial_velocity=vel)
    s = StokesSolver(**kw)
    t0 = time.perf_counter(); r = s.solve(); dt = time.perf_counter() - t0
    return s, r, dt


def main():
    pore = load_pore(VOLUME_CODE, DOWNSCALE_FACTOR)
    print(f"Bentheimer {VOLUME_CODE} f={DOWNSCALE_FACTOR} shape={pore.shape} "
          f"fluid={int(pore.sum())}")

    # FL Darcy solve -- shared setup for BOTH warm seeds (p_FL). Timed so we can
    # judge total wall time honestly (this is overhead cold never pays, but is
    # "free" if you compute the FL permeability anyway).
    t0 = time.perf_counter(); p_fl, cond = fast_laplacian(pore, SCALE)
    t_fl = time.perf_counter() - t0
    print(f"FL Darcy solve (shared by warm seeds): {t_fl:.1f}s")

    # converged reference (cold: no setup at all)
    ref, r_ref, t_ref = run_stokes(pore, None)
    ref_vec = stack(ref.u, ref.v, ref.w)
    n_total = r_ref["iterations"]
    print(f"cold reference: N_total={n_total}  solve={t_ref:.1f}s  stop={ref.stop_reason}")

    # build the seeds, timing each build separately
    t0 = time.perf_counter(); kgp = kgradp_velocity(p_fl, cond, SCALE, vscale=KGRADP_SCALE)
    t_build_kgp = time.perf_counter() - t0
    print("building viscous-from-pressure seed (one large-dt implicit solve)...")
    t0 = time.perf_counter(); visc = viscous_velocity_from_pressure(pore, p_fl)
    t_build_visc = time.perf_counter() - t0

    seeds = {
        f"k*grad(p) x{KGRADP_SCALE}": ((p_fl, kgp), t_build_kgp),
        "viscous from p_FL": ((p_fl, visc), t_build_visc),
    }

    # seed quality (note: relL2 does NOT predict convergence -- see iters below)
    print("\n--- seed quality vs converged velocity ---")
    print(f"  {'seed':28} {'relL2(seed,conv)':>18} {'build(s)':>10}")
    for name, (seed, tb) in seeds.items():
        print(f"  {name:28} {rel_l2(stack(*seed[1]), ref_vec):>18.4f} {tb:>10.1f}")

    # convergence + TOTAL wall time. Two totals:
    #   total(+FL)      = FL + build + solve   (FL counted as overhead)
    #   total(FL amort) = build + solve        (FL computed anyway for permeability)
    print("\n--- convergence & TOTAL wall time (explicit) ---")
    print(f"  {'seed':28} {'iters':>6} {'solve':>8} {'build':>7} {'+FL total':>10} "
          f"{'FL-amort':>9} {'final relL2':>12}")
    print(f"  {'cold (reference)':28} {n_total:>6} {t_ref:>8.1f} {'-':>7} "
          f"{t_ref:>10.1f} {t_ref:>9.1f} {0.0:>12.2e}")
    for name, (seed, tb) in seeds.items():
        s, r, dt = run_stokes(pore, seed)
        fin = rel_l2(stack(s.u, s.v, s.w), ref_vec)
        total_fl = t_fl + tb + dt
        total_amort = tb + dt
        print(f"  {name:28} {r['iterations']:>6} {dt:>8.1f} {tb:>7.1f} "
              f"{total_fl:>10.1f} {total_amort:>9.1f} {fin:>12.2e}")
    print("\n  (+FL total counts the FL Darcy solve as overhead; FL-amort assumes you")
    print("   compute the FL permeability anyway. cold pays neither.)")


if __name__ == "__main__":
    main()
