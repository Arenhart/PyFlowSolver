"""Diagnostics: where and why the enhanced fast-Laplacian (FL) approximation
diverges from the resolved Stokes solution.

For each Bentheimer volume it:
  1. Builds the enhanced FL fields (pressure + Darcy-flux MAC velocity) and, from
     the SAME model, the EDT map (distance to nearest wall) and the footprint map
     (local pore thickness) used to bin the errors.
  2. Solves Stokes to convergence (the reference) and snapshots several
     intermediate iterations (start/stop via warm-start) to see how the transient
     moves relative to the FL guess.
  3. Compares FL vs Stokes: global relL2 of pressure and speed, the L2-optimal
     velocity scale, per-z-slice conductance/pressure/speed metrics, and
     consolidations of the error binned by EDT (near vs far from wall) and by
     footprint (small vs medium vs large pores).
  4. Saves the 3D fields and difference maps as .nc, and reports the z-slices
     where FL and Stokes agree best / worst.

NOTHING is auto-run beyond `main()` when executed directly. Tune the knobs below;
`VOLUME_CODES` accepts any BIN_Bentheimer* code (OpenFOAM columns appear only for
codes that also have a solved_openfoam file).

    python scripts/fast_laplacian_diagnostics.py
"""

import os
import numpy as np
from scipy.io import netcdf_file

try:
    import h5py
except ImportError:
    h5py = None

from pyedt import edt

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.stokesSolver import StokesSolver
from pyflowsolver.fastLaplacian import _calculate_footprint

# ---- knobs -------------------------------------------------------------------
VOLUME_CODES = ["000"]          # e.g. ["000", "022", "111"]; any BIN_Bentheimer* code
DOWNSCALE_FACTOR = 2            # f: 5 -> 50^3 (fast), 2 -> 125^3, 1 -> 250^3 (slow)
SCALE = 1.0                     # isotropic voxel size
PREDICTOR = "explicit"         # complex media: explicit is the correct/fast choice
TARGET_ERROR = 1e-8            # below the floor on purpose; stagnation stop ends it
VISCOSITY = 1.0
DENSITY = 1.0
CHECKPOINT_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]   # of N_total
RUN_TRANSIENT = True           # the intermediate-iteration sweep (slow); off to iterate faster
N_EDT_BINS = 8
N_FOOTPRINT_BINS = 6
SLICE_PRINT_STRIDE = None      # None -> auto (~25 rows); or an int
SAVE_NC = True
OUTDIR = os.path.join("scripts", "diagnostics_output")

BASE = os.path.join("tests", "unit", "resources", "numerical_solved", "netcdf")
EPS = 1e-30


# ---- geometry / IO -----------------------------------------------------------
def block_mean(a, f):
    W, H, D = a.shape
    return a.reshape(W // f, f, H // f, f, D // f, f).mean(axis=(1, 3, 5))


def load_pore(code, f):
    bf = netcdf_file(f"{BASE}/BIN_Bentheimer{code}.nc", "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    pore = np.transpose(geom == 0, (1, 2, 0))          # 0 = pore; flow x -> z
    return block_mean(pore.astype(np.float64), f) >= 0.5


def load_openfoam_axial(code, f):
    """Coarsened OpenFOAM axial (flow-direction) velocity, or None if absent."""
    path = f"{BASE}/solved_openfoam/Bentheimer_DRP_{code}_OpenFOAM.nc"
    if h5py is None or not os.path.exists(path):
        return None
    hf = h5py.File(path, "r")
    vel = hf["float"][:].astype(np.float64); hf.close()
    vel = np.transpose(vel, (1, 2, 0))
    bf = netcdf_file(f"{BASE}/BIN_Bentheimer{code}.nc", "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data).astype(np.uint8)
    bf.close()
    pore = np.transpose(geom == 0, (1, 2, 0))
    W, H, D = vel.shape
    vp = (vel * pore).reshape(W // f, f, H // f, f, D // f, f).sum(axis=(1, 3, 5))
    cnt = pore.astype(np.float64).reshape(W // f, f, H // f, f, D // f, f).sum(axis=(1, 3, 5))
    out = np.zeros_like(vp); m = cnt > 0; out[m] = vp[m] / cnt[m]
    return out


def save_nc(path, arrays):
    """Write a dict {name: 3D float array} to a classic netCDF file (float32)."""
    shape = next(iter(arrays.values())).shape
    f = netcdf_file(path, "w")
    f.createDimension("x", shape[0])
    f.createDimension("y", shape[1])
    f.createDimension("z", shape[2])
    for name, arr in arrays.items():
        v = f.createVariable(name, "f", ("x", "y", "z"))
        v[:] = np.ascontiguousarray(arr, dtype=np.float32)
    f.close()


# ---- enhanced FL model: fields + the EDT / footprint maps it uses -------------
def edt_and_footprint(pore_bool, scale):
    """EDT (distance to wall) and footprint (local thickness), exactly as the
    enhanced Arns model computes them (closed border)."""
    spacing = (float(scale), float(scale), float(scale))
    w, h, d = pore_bool.shape
    bordered = np.zeros((w + 2, h + 2, d), dtype=pore_bool.dtype)
    bordered[1:-1, 1:-1, :] = pore_bool
    edt_map = edt(bordered, scale=spacing, force_method="cpu")[1:-1, 1:-1, :].astype(np.float32)
    footprint = np.zeros_like(edt_map, dtype=np.float32)
    _calculate_footprint(edt_map, footprint, spacing=spacing)
    return edt_map, footprint


def harmonic_face(c_lo, c_hi):
    both = (c_lo > 0) & (c_hi > 0)
    k = np.zeros_like(c_lo)
    k[both] = 2.0 / (1.0 / c_lo[both] + 1.0 / c_hi[both])
    return k


def darcy_mac_velocity(pressure, conductivity, scale):
    """Raw Darcy-flux MAC velocity from the FL solve (no magnitude tuning).

    The inlet/outlet z-faces (k=0 and k=D) are copied from the adjacent interior
    face for fluid columns -- the same open-boundary rule StokesSolver applies.
    Without this the first/last cell-centered axial velocity is halved (a pure
    edge artifact that used to show up as flux_FL[0] ~= flux_FL[1]/2).
    """
    p = np.asarray(pressure, np.float64); c = np.asarray(conductivity, np.float64)
    W, H, D = p.shape
    u = np.zeros((W + 1, H, D)); v = np.zeros((W, H + 1, D)); w = np.zeros((W, H, D + 1))
    u[1:W] = harmonic_face(c[:-1], c[1:]) * (p[:-1] - p[1:]) / scale
    v[:, 1:H] = harmonic_face(c[:, :-1], c[:, 1:]) * (p[:, :-1] - p[:, 1:]) / scale
    w[:, :, 1:D] = harmonic_face(c[:, :, :-1], c[:, :, 1:]) * (p[:, :, :-1] - p[:, :, 1:]) / scale
    fluid_cells = c > 0
    w[:, :, 0] = w[:, :, 1] * fluid_cells[:, :, 0]
    w[:, :, D] = w[:, :, D - 1] * fluid_cells[:, :, D - 1]
    return u, v, w


def fast_laplacian(pore, scale):
    """Enhanced FL: returns (p_field, conductivity_field, (u,v,w) MAC, vm)."""
    poremap = (pore * 100.0).astype(np.float32)
    vm = VolumeManager(poremap, scale=scale)
    vm.convert_pore_volume_to_laplacian_conductivity(enhanced_model=True)
    a, b = vm.get_sparse_system_jit()
    ds = DarcySolver(target_error=1e-9); ds.set_linear_system(a, b)
    ds.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = ds.solve_pcg()
    p = np.asarray(vm.ravel_sparse_solution(x), np.float64)
    uvw = darcy_mac_velocity(p, np.asarray(vm.volume, np.float64), scale)
    return p, np.asarray(vm.volume, np.float64), uvw, vm


# ---- field helpers -----------------------------------------------------------
def cell_speed(u, v, w):
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(uc ** 2 + vc ** 2 + wc ** 2), wc


def axial_flux_per_slice(wc, mask):
    """Sum of axial cell velocity over each z-slice (proportional to slice flux)."""
    return (wc * mask).sum(axis=(0, 1)) * (SCALE * SCALE)


def rel_l2(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    d = a - b
    denom = np.sqrt((b * b).sum())
    return np.sqrt((d * d).sum()) / denom if denom > 0 else np.inf


def optimal_scale(a, b, mask):
    """s minimizing ||s*a - b|| over masked entries (L2 projection)."""
    A, B = a[mask], b[mask]
    aa = float(np.dot(A, A))
    return float(np.dot(A, B) / aa) if aa > 0 else 1.0


def make_stokes(pore, max_iterations, seed=None, stagnation_window=25):
    kw = dict(volume=pore, scale=SCALE, viscosity=VISCOSITY, density=DENSITY,
              target_error=TARGET_ERROR, max_iterations=max_iterations,
              predictor=PREDICTOR, stagnation_window=stagnation_window)
    if seed is not None:
        p0, (u0, v0, w0) = seed
        kw.update(initial_pressure=p0, initial_velocity=(u0, v0, w0))
    return StokesSolver(**kw)


# ---- binned consolidation ----------------------------------------------------
def profile_by_bin(values, sfl, sst, n_bins, label):
    """FL vs Stokes speed compared across quantile bins of `values` (1D arrays,
    already restricted to the cells of interest).

    Columns: bin center, count, mean speed_FL, mean speed_Stokes, ratio FL/St,
    mean SIGNED (FL-St), mean |FL-St|. The signed column is the key one: a
    consistent sign that flips between low and high `values` means the FL profile
    shape is off (too flat vs too peaked); a uniform sign means a magnitude bias.
    """
    if values.size == 0:
        print(f"\n  (no cells for {label})")
        return
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    print(f"\n  FL vs Stokes speed by {label}:")
    print(f"    {'center':>8} {'n':>8} {'mean_FL':>11} {'mean_St':>11} "
          f"{'FL/St':>6} {'signedΔ':>11} {'|Δ|':>10}")
    for i in range(n_bins):
        sel = (values >= edges[i]) & (values < edges[i + 1])
        if not sel.any():
            continue
        af = float(sfl[sel].mean()); bf = float(sst[sel].mean())
        ratio = af / bf if bf > 0 else np.nan
        signed = float((sfl[sel] - sst[sel]).mean())
        absd = float(np.abs(sfl[sel] - sst[sel]).mean())
        c = 0.5 * (edges[i] + edges[i + 1])
        print(f"    {c:8.3f} {int(sel.sum()):8d} {af:11.4e} {bf:11.4e} "
              f"{ratio:6.3f} {signed:+11.3e} {absd:10.4e}")


def binned_median(bins, ratio, kstar, kfl, n_bins, label):
    """Median effective conductivity k* and Arns k_fl per quantile bin of `bins`.
    ratio = k*/k_Arns < 1 means Arns overestimates the local conductivity."""
    if bins.size == 0:
        print(f"\n  (no faces for {label})")
        return
    edges = np.quantile(bins, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    print(f"\n  k*/k_Arns by {label}:")
    print(f"    {'center':>8} {'n':>9} {'med k*':>11} {'med k_Arns':>11} {'med k*/k':>9}")
    for i in range(n_bins):
        sel = (bins >= edges[i]) & (bins < edges[i + 1])
        if not sel.any():
            continue
        c = 0.5 * (edges[i] + edges[i + 1])
        print(f"    {c:8.3f} {int(sel.sum()):9d} {np.median(kstar[sel]):11.4e} "
              f"{np.median(kfl[sel]):11.4e} {np.median(ratio[sel]):9.3f}")


# ---- per-volume diagnostic ---------------------------------------------------
def diagnose(code):
    print("\n" + "#" * 74)
    print(f"# Bentheimer {code}  (f={DOWNSCALE_FACTOR})")
    print("#" * 74)
    pore = load_pore(code, DOWNSCALE_FACTOR)
    shape = pore.shape
    Wd, Hd, Dd = shape
    print(f"shape={shape} fluid={int(pore.sum())} porosity={pore.mean():.3f}")

    # --- enhanced FL + its EDT/footprint maps --------------------------------
    p_fl, cond, uvw_fl, vm = fast_laplacian(pore, SCALE)
    edt_map, fp_map = edt_and_footprint(pore, SCALE)
    speed_fl, wc_fl = cell_speed(*uvw_fl)
    fl_flow, fl_cond = vm.get_conductivity(p_fl)

    # --- Stokes reference (converged) ----------------------------------------
    ref = make_stokes(pore, max_iterations=20000)
    r_ref = ref.solve()
    n_total = r_ref["iterations"]
    p_st = ref.p.copy()
    speed_st, wc_st = cell_speed(ref.u, ref.v, ref.w)
    fluid = ref.pressure_mask_bool & (cond > 0)     # common fluid cells
    print(f"Stokes reference: N_total={n_total} stop={ref.stop_reason} "
          f"converged={r_ref['converged']}")

    of_axial = load_openfoam_axial(code, DOWNSCALE_FACTOR)

    # --- conductance summary --------------------------------------------------
    mid = Dd // 2
    def slice_cond(wc):
        return (wc[:, :, mid] * fluid[:, :, mid]).sum() * SCALE * SCALE * Dd / (Wd * Hd)
    print("\n--- conductance (mid-plane) ---")
    print(f"  fast-Laplacian : {slice_cond(wc_fl):.6e}  (get_conductivity: {fl_cond:.6e})")
    print(f"  Stokes         : {slice_cond(wc_st):.6e}")
    if of_axial is not None:
        of_cond = (of_axial[:, :, mid] * fluid[:, :, mid]).sum() * SCALE * SCALE * Dd / (Wd * Hd)
        print(f"  OpenFOAM       : {of_cond:.6e}")

    # --- global field agreement ----------------------------------------------
    s_star = optimal_scale(speed_fl, speed_st, fluid)
    print("\n--- global FL vs Stokes agreement (fluid cells) ---")
    print(f"  pressure relL2 (raw)      : {rel_l2(p_fl, p_st, fluid):.3f}")

    # Detrend by the per-slice (cross-sectional) mean pressure to separate the
    # trivial shared inlet->outlet drop from the interior structure:
    #   axial profile  = how well FL matches the p(z) drop shape
    #   fluctuations   = how well FL matches the lateral (in-plane) pressure field
    zmean_fl = np.array([p_fl[:, :, k][fluid[:, :, k]].mean() if fluid[:, :, k].any()
                         else 0.0 for k in range(Dd)])
    zmean_st = np.array([p_st[:, :, k][fluid[:, :, k]].mean() if fluid[:, :, k].any()
                         else 0.0 for k in range(Dd)])
    axial_rel = (np.linalg.norm(zmean_fl - zmean_st) / np.linalg.norm(zmean_st)
                 if np.linalg.norm(zmean_st) > 0 else np.inf)
    fluct_rel = rel_l2(p_fl - zmean_fl[None, None, :],
                       p_st - zmean_st[None, None, :], fluid)
    print(f"  pressure relL2 (axial p(z)): {axial_rel:.3f}   "
          f"(fluctuations about it: {fluct_rel:.3f})")
    print(f"  speed    relL2 (raw)      : {rel_l2(speed_fl, speed_st, fluid):.3f}")
    print(f"  speed    relL2 (scaled s*): {rel_l2(s_star * speed_fl, speed_st, fluid):.3f}")
    print(f"  optimal speed scale s*    : {s_star:.3f} (FL overshoots ~{1/s_star:.2f}x)")

    # --- intermediate Stokes steps vs FL and vs converged --------------------
    if RUN_TRANSIENT:
        print("\n--- transient: Stokes at intermediate iterations ---")
        print("   iters   relL2(step,conv)  relL2(step,FL)   flux(mid)")
        caps = sorted(set(max(1, int(fr * n_total)) for fr in CHECKPOINT_FRACTIONS))
        prev = 0
        seed = None
        for cap in caps:
            step_solver = make_stokes(pore, max_iterations=cap - prev, seed=seed,
                                      stagnation_window=0)
            step_solver.solve()
            seed = (step_solver.p.copy(),
                    (step_solver.u.copy(), step_solver.v.copy(), step_solver.w.copy()))
            sp, wc = cell_speed(step_solver.u, step_solver.v, step_solver.w)
            print(f"   {cap:6d}   {rel_l2(sp, speed_st, fluid):.4e}        "
                  f"{rel_l2(sp, speed_fl, fluid):.4e}       {slice_cond(wc):.4e}")
            prev = cap

    # --- per-z-slice metrics --------------------------------------------------
    flux_fl = axial_flux_per_slice(wc_fl, fluid)
    flux_st = axial_flux_per_slice(wc_st, fluid)
    flux_of = axial_flux_per_slice(of_axial[:, :, :], fluid) if of_axial is not None else None
    dp_slice = np.zeros(Dd); dsp_slice = np.zeros(Dd); corr_slice = np.full(Dd, np.nan)
    for k in range(Dd):
        m = fluid[:, :, k]
        if m.sum() < 2:
            continue
        a = p_fl[:, :, k][m]; b = p_st[:, :, k][m]
        dp_slice[k] = np.abs(a - b).mean()
        sa = speed_fl[:, :, k][m]; sb = speed_st[:, :, k][m]
        dsp_slice[k] = np.abs(sa - sb).mean()
        if sa.std() > 0 and sb.std() > 0:
            corr_slice[k] = np.corrcoef(sa, sb)[0, 1]

    stride = SLICE_PRINT_STRIDE or max(1, Dd // 25)
    print("\n--- per-z-slice (stride {}) ---".format(stride))
    hdr = "    z   poro    flux_FL     flux_Stokes" + ("   flux_OF   " if flux_of is not None else "")
    print(hdr + "   mean|Δp|    mean|Δspeed|  corr(speed)")
    poro_z = fluid.mean(axis=(0, 1))
    for k in range(0, Dd, stride):
        line = f"  {k:3d}  {poro_z[k]:.3f}  {flux_fl[k]:.4e}  {flux_st[k]:.4e}"
        if flux_of is not None:
            line += f"  {flux_of[k]:.4e}"
        line += f"   {dp_slice[k]:.4e}   {dsp_slice[k]:.4e}    {corr_slice[k]:+.3f}"
        print(line)

    # slices of best / worst agreement (by mean speed diff, fluid-weighted)
    valid = np.where(fluid.any(axis=(0, 1)))[0]
    order = valid[np.argsort(dsp_slice[valid])]
    print("\n  best-agreement z-slices (smallest mean|Δspeed|):",
          ", ".join(f"{k}({dsp_slice[k]:.2e})" for k in order[:5]))
    print("  worst-agreement z-slices (largest  mean|Δspeed|):",
          ", ".join(f"{k}({dsp_slice[k]:.2e})" for k in order[-5:][::-1]))

    # --- consolidation: FL vs Stokes speed profiles by geometry ---------------
    # The enhanced model's conductance is ALREADY parabolic in EDT (Poiseuille
    # with half-width R=footprint+alpha), so the question is whether that built-in
    # profile is too flat or too peaked vs Stokes -> read the signed column.
    edt_f = edt_map[fluid]; fp_f = fp_map[fluid]
    sfl_f = speed_fl[fluid]; sst_f = speed_st[fluid]
    print("\n--- FL vs Stokes speed consolidation (fluid cells) ---")
    profile_by_bin(edt_f, sfl_f, sst_f, N_EDT_BINS, "EDT (wall distance)")
    profile_by_bin(fp_f, sfl_f, sst_f, N_FOOTPRINT_BINS, "footprint (pore size)")
    # The error grows with pore size, so inspect the in-pore profile specifically
    # in the largest pores (top footprint tercile): does FL over- or under-shoot
    # near the wall vs the center there?
    if fp_f.size:
        big = fp_f > np.quantile(fp_f, 2.0 / 3.0)
        profile_by_bin(edt_f[big], sfl_f[big], sst_f[big], N_EDT_BINS,
                       "EDT within large pores (top footprint tercile)")

    # --- inverse problem: what conductivity reproduces the Stokes velocity? ----
    # On each face between two fluid cells, Darcy says v_face = k_face*(p_lo-p_hi)/h.
    # Invert with the Stokes face velocity and the FL pressure to get the effective
    # k*_face, and compare to the FL face conductivity harmonic_face(k_Arns).
    #   k*/k_Arns < 1  => Arns OVERestimates the conductivity there (velocity too high)
    #   k*/k_Arns > 1  => Arns underestimates
    print("\n--- effective conductivity k* (v_Stokes = -k* grad p_FL) vs Arns ---")
    acc = np.zeros(shape); cnt = np.zeros(shape)
    ks_all = []; kf_all = []; ed_all = []; fp_all = []
    dp_frac = 0.05

    def face_kstar(clo, chi, dp, vf):
        kfl = harmonic_face(clo, chi)
        act = (clo > 0) & (chi > 0)
        nz = act & (np.abs(dp) > 0)
        thr = dp_frac * np.median(np.abs(dp[nz])) if nz.any() else 0.0
        act = act & (np.abs(dp) > thr)
        kstar = np.zeros_like(dp)
        kstar[act] = vf[act] * SCALE / dp[act]
        keep = act & (kstar > 0) & (kfl > 0)     # down-gradient faces only
        return kfl, kstar, keep

    # x-faces (cells i-1,i); y-faces (j-1,j); z-faces (k-1,k)
    kfl, ks, kp = face_kstar(cond[:-1], cond[1:], p_fl[:-1] - p_fl[1:], ref.u[1:-1])
    acc[:-1] += np.where(kp, ks, 0.0); cnt[:-1] += kp
    acc[1:] += np.where(kp, ks, 0.0); cnt[1:] += kp
    ks_all.append(ks[kp]); kf_all.append(kfl[kp])
    ed_all.append(np.minimum(edt_map[:-1], edt_map[1:])[kp])
    fp_all.append((0.5 * (fp_map[:-1] + fp_map[1:]))[kp])

    kfl, ks, kp = face_kstar(cond[:, :-1], cond[:, 1:], p_fl[:, :-1] - p_fl[:, 1:], ref.v[:, 1:-1])
    acc[:, :-1] += np.where(kp, ks, 0.0); cnt[:, :-1] += kp
    acc[:, 1:] += np.where(kp, ks, 0.0); cnt[:, 1:] += kp
    ks_all.append(ks[kp]); kf_all.append(kfl[kp])
    ed_all.append(np.minimum(edt_map[:, :-1], edt_map[:, 1:])[kp])
    fp_all.append((0.5 * (fp_map[:, :-1] + fp_map[:, 1:]))[kp])

    kfl, ks, kp = face_kstar(cond[:, :, :-1], cond[:, :, 1:], p_fl[:, :, :-1] - p_fl[:, :, 1:], ref.w[:, :, 1:-1])
    acc[:, :, :-1] += np.where(kp, ks, 0.0); cnt[:, :, :-1] += kp
    acc[:, :, 1:] += np.where(kp, ks, 0.0); cnt[:, :, 1:] += kp
    ks_all.append(ks[kp]); kf_all.append(kfl[kp])
    ed_all.append(np.minimum(edt_map[:, :, :-1], edt_map[:, :, 1:])[kp])
    fp_all.append((0.5 * (fp_map[:, :, :-1] + fp_map[:, :, 1:]))[kp])

    ks_all = np.concatenate(ks_all); kf_all = np.concatenate(kf_all)
    ed_all = np.concatenate(ed_all); fp_all = np.concatenate(fp_all)
    ratio_all = ks_all / kf_all
    print(f"  faces used: {ks_all.size}   global median k*/k_Arns: {np.median(ratio_all):.3f}"
          f"   (<1 => Arns overestimates conductivity / velocity)")
    binned_median(ed_all, ratio_all, ks_all, kf_all, N_EDT_BINS, "EDT (wall distance)")
    binned_median(fp_all, ratio_all, ks_all, kf_all, N_FOOTPRINT_BINS, "footprint (pore size)")
    k_cell = np.zeros(shape); m = cnt > 0; k_cell[m] = acc[m] / cnt[m]
    ratio_cell = np.zeros(shape); mm = m & (cond > 0); ratio_cell[mm] = k_cell[mm] / cond[mm]

    # --- save fields ----------------------------------------------------------
    if SAVE_NC:
        os.makedirs(OUTDIR, exist_ok=True)
        p_diff = np.where(fluid, p_fl - p_st, 0.0)
        sp_diff = np.where(fluid, speed_fl - speed_st, 0.0)
        out = os.path.join(OUTDIR, f"diag_{code}_f{DOWNSCALE_FACTOR}.nc")
        save_nc(out, {
            "pressure_fl": p_fl, "pressure_stokes": p_st, "pressure_diff": p_diff,
            "speed_fl": speed_fl, "speed_stokes": speed_st, "speed_diff": sp_diff,
            "edt": edt_map, "footprint": fp_map, "conductivity": cond,
            "k_effective": k_cell, "k_eff_over_arns": ratio_cell,
            "fluid": fluid.astype(np.float32),
        })
        print(f"\n  saved fields -> {out}")


def main():
    for code in VOLUME_CODES:
        diagnose(code)


if __name__ == "__main__":
    main()
