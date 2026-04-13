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

SIZE = 100
SEGMENT = 2
SIGMA = 2
RADIUS = 5

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
solver = DarcySolver(target_error=1e-8)
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
solver = DarcySolver(target_error=1e-8)
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

solver = DarcySolver()
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


