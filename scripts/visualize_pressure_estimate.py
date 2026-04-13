"""Visualize pressure estimate from pseudo-network.

Generates a 100³ porous volume, runs estimate_pressure_distribution,
validates the pressure range, prints mean pressure per every 5th XY plane,
and saves the result as a TIFF stack.
"""

import numpy as np
import scipy as sc
import porespy as ps
from tifffile import imwrite

from pyflowsolver.pressureEstimator import estimate_pressure_distribution

SIZE = 150

# ---------------------------------------------------------------------------
# 1. Generate volume
# ---------------------------------------------------------------------------
print(f"Generating {SIZE}³ blob volume...")
image = ps.generators.blobs(shape=(SIZE, SIZE, SIZE), porosity=0.38, seed=42)
labeled_image, _ = sc.ndimage.label(image)

inlet_labels = set(np.unique(labeled_image[:, :, 0])) - {0}
outlet_labels = set(np.unique(labeled_image[:, :, -1])) - {0}
connected_labels = inlet_labels & outlet_labels
if not connected_labels:
    raise RuntimeError("No connected path from inlet to outlet")

best_label = max(connected_labels, key=lambda l: (labeled_image == l).sum())
image = (labeled_image == best_label)
print(f"  Pore voxels: {image.sum()} / {image.size}")

# ---------------------------------------------------------------------------
# 2. Run pressure estimate
# ---------------------------------------------------------------------------
print("\nRunning estimate_pressure_distribution...")
pore_volume = image.astype(np.uint8)
conductivity_volume = image.astype(np.float64)
scale = (1.0, 1.0, 1.0)

pressure_volume = estimate_pressure_distribution(
    pore_volume, 
    conductivity_volume, 
    scale,
    subsegment_size=4,
    )

# ---------------------------------------------------------------------------
# 3. Validate pressure range
# ---------------------------------------------------------------------------
pore_mask = pore_volume > 0
pore_pressures = pressure_volume[pore_mask]
p_min = pore_pressures.min()
p_max = pore_pressures.max()
print(f"\nPressure range in pore voxels: [{p_min:.6f}, {p_max:.6f}]")
assert p_min >= -1e-3, f"Pressure below 0: {p_min}"
assert p_max <= (1. + 1e-3), f"Pressure above 1: {p_max}"
print("  All pressures within [0, 1].")

# ---------------------------------------------------------------------------
# 4. Mean pressure per every 5th XY plane
# ---------------------------------------------------------------------------
d = pressure_volume.shape[2]
print(f"\nMean pore pressure per XY plane (every 5th, z=0..{d-1}):")
print(f"  {'z':>4s}  {'mean_p':>8s}  {'n_pores':>8s}")
for z in range(0, d, 5):
    plane_mask = pore_volume[:, :, z] > 0
    n_pores = plane_mask.sum()
    if n_pores > 0:
        mean_p = pressure_volume[:, :, z][plane_mask].mean()
    else:
        mean_p = float("nan")
    print(f"  {z:4d}  {mean_p:8.4f}  {n_pores:8d}")

# ---------------------------------------------------------------------------
# 5. Save as TIFF stack
# ---------------------------------------------------------------------------
output_path = "pressure_estimate.tiff"
# TIFF stack axes: (z, y, x) — transpose from (x, y, z)
tiff_data = np.transpose(pressure_volume, (2, 1, 0)).astype(np.float32)
imwrite(output_path, tiff_data)
print(f"\nSaved pressure volume to {output_path}")
print(f"  Shape (z, y, x): {tiff_data.shape}, dtype: {tiff_data.dtype}")



