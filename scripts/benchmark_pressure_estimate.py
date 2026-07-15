"""Benchmark: compare PCG solve times with zero vs estimated initial guess.

Creates a 100³ porous volume, solves the Darcy system twice:
  1. With X0 = zeros (standard)
  2. With X0 = pressure estimate from pseudo-network

Then compares results to verify both solutions match.
"""

import time

import numpy as np
import scipy as sc
import porespy as ps
from tifffile import imwrite

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.pressureEstimator import estimate_pressure_distribution

SIZE = 200
SEGMENT = 2
SIGMA = 2
RADIUS = 5

TARGET_TOL = 1e-7
MAX_ITER = 10000

# ---------------------------------------------------------------------------
# 0. Generate volume and build sparse system
# ---------------------------------------------------------------------------
print(f"Generating dummy volume...")
image = ps.generators.blobs(shape=(10, 10, 10), porosity=0.60, seed=42)
labeled_image, _ = sc.ndimage.label(image)

# Keep only the largest connected component that spans inlet to outlet
inlet_labels = set(np.unique(labeled_image[:, :, 0])) - {0}
outlet_labels = set(np.unique(labeled_image[:, :, -1])) - {0}
connected_labels = inlet_labels & outlet_labels
if not connected_labels:
    raise RuntimeError("No connected path from inlet to outlet")

# Pick the label with the most voxels among connected ones
best_label = max(connected_labels, key=lambda l: (labeled_image == l).sum())
image = (labeled_image == best_label)

volume_manager = VolumeManager(image)
volume_manager.convert_pore_volume_to_laplacian_conductivity()

sparse_A, dense_b = volume_manager.get_sparse_system_jit()

print("\n--- Warming up JIT ---")
solver = DarcySolver(target_error=TARGET_TOL, max_iterations=MAX_ITER)
print()
solver.set_linear_system(sparse_A, dense_b)
solver.generate_preconditioner(preconditioner="inverse_diagonal")

x_zero, error_zero, iters_zero = DarcySolver._solve_pcg(
    A_val=sparse_A["val"],
    A_col_idx=sparse_A["col_idx"],
    A_row_ptr=sparse_A["row_ptr"],
    P_val=solver.preconditioner["val"],
    P_col_idx=solver.preconditioner["col_idx"],
    P_row_ptr=solver.preconditioner["row_ptr"],
    b=dense_b,
    max_iterations=solver.params["max_iterations"],
    target_error=solver.params["target_error"],
    X0=np.zeros_like(dense_b),
    threads=1,
)

# ---------------------------------------------------------------------------
# 1. Generate volume and build sparse system
# ---------------------------------------------------------------------------
print(f"Generating {SIZE}³ blob volume...")
image = ps.generators.blobs(shape=(SIZE, SIZE, SIZE), porosity=0.38, seed=42)
labeled_image, _ = sc.ndimage.label(image)

# Keep only the largest connected component that spans inlet to outlet
inlet_labels = set(np.unique(labeled_image[:, :, 0])) - {0}
outlet_labels = set(np.unique(labeled_image[:, :, -1])) - {0}
connected_labels = inlet_labels & outlet_labels
if not connected_labels:
    raise RuntimeError("No connected path from inlet to outlet")

# Pick the label with the most voxels among connected ones
best_label = max(connected_labels, key=lambda l: (labeled_image == l).sum())
image = (labeled_image == best_label)
print(f"  Pore voxels: {image.sum()} / {image.size}")

volume_manager = VolumeManager(image)
volume_manager.convert_pore_volume_to_laplacian_conductivity()

print("Assembling sparse system...")
sparse_A, dense_b = volume_manager.get_sparse_system_jit()
print(f"  System size N={dense_b.size}, nnz={sparse_A['val'].size}")

# ---------------------------------------------------------------------------
# 2. Solve with X0 = zeros
# ---------------------------------------------------------------------------
print("\n--- Solve with X0 = zeros ---")
solver = DarcySolver(target_error=TARGET_TOL, max_iterations=MAX_ITER)
print()
solver.set_linear_system(sparse_A, dense_b)
solver.generate_preconditioner(preconditioner="inverse_diagonal")

t0 = time.perf_counter()
x_zero, error_zero, iters_zero = DarcySolver._solve_pcg(
    A_val=sparse_A["val"],
    A_col_idx=sparse_A["col_idx"],
    A_row_ptr=sparse_A["row_ptr"],
    P_val=solver.preconditioner["val"],
    P_col_idx=solver.preconditioner["col_idx"],
    P_row_ptr=solver.preconditioner["row_ptr"],
    b=dense_b,
    max_iterations=solver.params["max_iterations"],
    target_error=solver.params["target_error"],
    X0=np.zeros_like(dense_b),
    threads=1,
)
t_zero = time.perf_counter() - t0
zero_volume = volume_manager.ravel_sparse_solution(x_zero)
print(f"  Time:       {t_zero:.3f} s")
print(f"  Iterations: {iters_zero}")
print(f"  Error:      {error_zero:.2e}")

# ---------------------------------------------------------------------------
# 3. Build pressure estimate
# ---------------------------------------------------------------------------
print("\nBuilding pressure estimate from pseudo-network...")
pore_volume = (volume_manager.volume > 0).astype(np.uint8)
conductivity_volume = volume_manager.volume.astype(np.float64)
scale = volume_manager.scale

t0_est = time.perf_counter()
pressure_est_vol = estimate_pressure_distribution(
    pore_volume, 
    conductivity_volume, 
    scale,
    subsegment_size=SEGMENT,
    sigma=SIGMA,
    radius=RADIUS,
)
t_est = time.perf_counter() - t0_est
print(f"  Estimate time: {t_est:.3f} s")

# Extract 1D initial guess in condensed ordering (matching sparse system rows)
w, h, d = volume_manager.volume.shape
X0_est = np.zeros(dense_b.size, dtype=np.float64)
i = 0
for x in range(w):
    for y in range(h):
        for z in range(d):
            if volume_manager.volume[x, y, z] > 0:
                X0_est[i] = pressure_est_vol[x, y, z]
                i += 1

# ---------------------------------------------------------------------------
# 4. Solve with X0 = pressure estimate
# ---------------------------------------------------------------------------
print("\n--- Solve with X0 = pressure estimate ---")

solver = DarcySolver(target_error=TARGET_TOL, max_iterations=MAX_ITER)
print()
sparse_A, dense_b = volume_manager.get_sparse_system_jit()
solver.set_linear_system(sparse_A, dense_b)
solver.generate_preconditioner(preconditioner="inverse_diagonal")

t0 = time.perf_counter()
x_est, error_est, iters_est = solver._solve_pcg(
    A_val=sparse_A["val"],
    A_col_idx=sparse_A["col_idx"],
    A_row_ptr=sparse_A["row_ptr"],
    P_val=solver.preconditioner["val"],
    P_col_idx=solver.preconditioner["col_idx"],
    P_row_ptr=solver.preconditioner["row_ptr"],
    b=dense_b,
    max_iterations=solver.params["max_iterations"],
    target_error=solver.params["target_error"],
    X0=X0_est,
    threads=1,
)
t_est_solve = time.perf_counter() - t0
print(f"  Time:       {t_est_solve:.3f} s")
print(f"  Iterations: {iters_est}")
print(f"  Error:      {error_est:.2e}")

# ---------------------------------------------------------------------------
# 5. Compare results
# ---------------------------------------------------------------------------
print("\n--- Comparison ---")
print(f"  Zero-init:     {iters_zero} iters in {t_zero:.3f} s")
print(f"  Estimate-init: {iters_est} iters in {t_est_solve:.3f} s  (+ {t_est:.3f} s estimate)")
print(f"  Speedup (solve only):  {t_zero / t_est_solve:.2f}x")
print(f"  Speedup (total):       {t_zero / (t_est_solve + t_est):.2f}x")

max_diff = np.max(np.abs(x_zero - x_est))
print(f"  Max |x_zero - x_est|: {max_diff:.2e}")
try:
    np.testing.assert_allclose(x_zero, x_est, rtol=1e-4)
    print("  Solutions match (rtol=1e-4).")
except AssertionError as e:
    print("  Solutions do NOT match within rtol=1e-4.")
    print(str(e))


# ---------------------------------------------------------------------------
# 6. Save diff as TIFF stack
# ---------------------------------------------------------------------------
output_path = f"pressure_estimate_diff_{SEGMENT}_{SIGMA}_{RADIUS}.tiff"
# TIFF stack axes: (z, y, x) — transpose from (x, y, z)
diff = zero_volume - pressure_est_vol
tiff_data = np.transpose(diff, (2, 1, 0)).astype(np.float32)
imwrite(output_path, tiff_data)
imwrite(output_path+"_zero.tiff", np.transpose(zero_volume, (2, 1, 0)).astype(np.float32))
imwrite(output_path+"_est.tiff", np.transpose(pressure_est_vol, (2, 1, 0)).astype(np.float32))
print(f"\nSaved pressure volume to {output_path}")
print(f"  Shape (z, y, x): {tiff_data.shape}, dtype: {tiff_data.dtype}")

try:
    np.testing.assert_allclose(
        np.transpose(zero_volume, (2, 1, 0)).astype(np.float32),
        np.transpose(pressure_est_vol, (2, 1, 0)).astype(np.float32),
        rtol=1e-4,
        )
    print("  Solutions match (rtol=1e-4).")
except AssertionError as e:
    print("  Solutions do NOT match within rtol=1e-4.")
    print(str(e))


# ---------------------------------------------------------------------------
# 7. Signed pressure / gradient analysis (sign is meaningful — do NOT abs)
# ---------------------------------------------------------------------------
# Convention for every signed field below: value = zero - est.
#   > 0  -> the PCG-converged field (zero_volume) is higher/steeper than the estimate
#   < 0  -> the pseudo-network estimate (pressure_est_vol) is higher/steeper
# Keeping the sign tells us which side the estimator is biased toward.
print("\n--- Signed analysis (zero_volume - pressure_est_vol) ---")

pore_mask = volume_manager.volume > 0
n_pore = int(pore_mask.sum())
conductivity = volume_manager.volume.astype(np.float32)
spacing = tuple(float(s) for s in volume_manager.scale)

zv = zero_volume.astype(np.float32)
ev = pressure_est_vol.astype(np.float32)

# --- Signed pressure difference -------------------------------------------
signed_pressure_diff = zv - ev  # already present as `diff` above; recomputed locally for clarity

# --- Gradient magnitudes --------------------------------------------------
# np.gradient on the full volume uses the 0s in solid voxels, which inflates
# the gradient at pore/solid interfaces. That bias applies to BOTH fields
# equally, so the signed difference grad_zero_mag - grad_est_mag is still
# meaningful as a relative measure.
gz = np.gradient(zv, *spacing)
ge = np.gradient(ev, *spacing)
grad_zero_mag = np.sqrt(gz[0]**2 + gz[1]**2 + gz[2]**2)
grad_est_mag  = np.sqrt(ge[0]**2 + ge[1]**2 + ge[2]**2)
signed_grad_diff = grad_zero_mag - grad_est_mag

# Mask solid voxels to 0 in the TIFF outputs so viewers show a clean void.
signed_pressure_diff_m = np.where(pore_mask, signed_pressure_diff, 0.0)
grad_zero_mag_m        = np.where(pore_mask, grad_zero_mag, 0.0)
grad_est_mag_m         = np.where(pore_mask, grad_est_mag, 0.0)
signed_grad_diff_m     = np.where(pore_mask, signed_grad_diff, 0.0)

# --- Save TIFF stacks (z, y, x) -------------------------------------------
def _save_zyx(arr, path):
    imwrite(path, np.transpose(arr, (2, 1, 0)).astype(np.float32))

tag = f"{SEGMENT}_{SIGMA}_{RADIUS}"
_save_zyx(signed_pressure_diff_m, f"pressure_signed_diff_{tag}.tiff")
_save_zyx(grad_zero_mag_m,        f"gradient_mag_zero_{tag}.tiff")
_save_zyx(grad_est_mag_m,         f"gradient_mag_est_{tag}.tiff")
_save_zyx(signed_grad_diff_m,     f"gradient_signed_diff_{tag}.tiff")
print(f"  Saved pressure_signed_diff_{tag}.tiff")
print(f"  Saved gradient_mag_zero_{tag}.tiff, gradient_mag_est_{tag}.tiff")
print(f"  Saved gradient_signed_diff_{tag}.tiff")

# --- Pore-only 1D views for stats / plots ---------------------------------
p_diff_pore = signed_pressure_diff[pore_mask]
g_diff_pore = signed_grad_diff[pore_mask]
cond_pore   = conductivity[pore_mask]

def _summary(name, arr):
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    zer = arr.size - pos - neg
    print(
        f"  {name}: mean={arr.mean():+.3e}  median={np.median(arr):+.3e}  "
        f"min={arr.min():+.3e}  max={arr.max():+.3e}"
    )
    print(
        f"    zero>est: {pos:>7d} ({100*pos/arr.size:5.1f}%)   "
        f"est>zero: {neg:>7d} ({100*neg/arr.size:5.1f}%)   "
        f"equal: {zer}"
    )

_summary("pressure diff (zero - est)   ", p_diff_pore)
_summary("grad-mag diff (zero - est)   ", g_diff_pore)

# ---------------------------------------------------------------------------
# 8. Correlate signed quantities with local conductivity
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Bin by conductivity and report signed medians so the sign direction is visible.
n_bins = 30
bin_edges = np.linspace(cond_pore.min(), cond_pore.max(), n_bins + 1)
bin_idx = np.clip(np.digitize(cond_pore, bin_edges) - 1, 0, n_bins - 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

def _binned_median(values):
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.any():
            out[b] = np.median(values[sel])
    return out

p_diff_median = _binned_median(p_diff_pore)
g_diff_median = _binned_median(g_diff_pore)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Row 0: signed quantity vs local conductivity (hexbin + binned median)
ax = axes[0, 0]
hb = ax.hexbin(cond_pore, p_diff_pore, gridsize=60, mincnt=1, cmap="Blues",
               bins="log")
ax.plot(bin_centers, p_diff_median, "r-", lw=2, label="binned median")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("local conductivity")
ax.set_ylabel("p_zero − p_est (signed)")
ax.set_title("Signed pressure diff vs conductivity")
fig.colorbar(hb, ax=ax, label="log10 count")
ax.legend(loc="best")

ax = axes[0, 1]
hb = ax.hexbin(cond_pore, g_diff_pore, gridsize=60, mincnt=1, cmap="Oranges",
               bins="log")
ax.plot(bin_centers, g_diff_median, "r-", lw=2, label="binned median")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("local conductivity")
ax.set_ylabel("|∇p|_zero − |∇p|_est (signed)")
ax.set_title("Signed gradient-magnitude diff vs conductivity")
fig.colorbar(hb, ax=ax, label="log10 count")
ax.legend(loc="best")

# Row 1: histograms of each signed quantity (sign visible at zero line)
ax = axes[1, 0]
ax.hist(p_diff_pore, bins=80, color="steelblue", edgecolor="black")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("p_zero − p_est")
ax.set_ylabel("count")
ax.set_title(
    f"Pressure diff distribution "
    f"(mean {p_diff_pore.mean():+.2e}, median {np.median(p_diff_pore):+.2e})"
)

ax = axes[1, 1]
ax.hist(g_diff_pore, bins=80, color="darkorange", edgecolor="black")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("|∇p|_zero − |∇p|_est")
ax.set_ylabel("count")
ax.set_title(
    f"Grad-mag diff distribution "
    f"(mean {g_diff_pore.mean():+.2e}, median {np.median(g_diff_pore):+.2e})"
)

fig.suptitle(
    f"Pressure-estimator error analysis  "
    f"(SEGMENT={SEGMENT}, SIGMA={SIGMA}, RADIUS={RADIUS}, N_pore={n_pore})"
)
fig.tight_layout(rect=(0, 0, 1, 0.97))

plot_path = f"pressure_analysis_{tag}.png"
fig.savefig(plot_path, dpi=120)
print(f"\nSaved analysis plot to {plot_path}")

# Pearson correlation (signed) between conductivity and each signed field.
# Useful single-number summary of which direction the bias trends with c.
if cond_pore.std() > 0:
    r_p = float(np.corrcoef(cond_pore, p_diff_pore)[0, 1])
    r_g = float(np.corrcoef(cond_pore, g_diff_pore)[0, 1])
    print(f"  Pearson r(conductivity, p_zero - p_est)         = {r_p:+.3f}")
    print(f"  Pearson r(conductivity, |grad|_zero - |grad|_est) = {r_g:+.3f}")

plt.show()
