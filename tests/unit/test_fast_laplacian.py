"""Tests for fastLaplacian, focused on the enhanced-Arns footprint pass.

`_calculate_footprint` computes a local-thickness / footprint field:

    footprint[v] = max{ edt[u] : ||u - v|| <= edt[u] }   for pore voxels (edt[v] > 0)

i.e. the largest radius of any inscribed ball (centred anywhere) that still
covers voxel v. It must be **deterministic** -- the result cannot depend on how
numba schedules the parallel loop.
"""

import math

import numpy as np
import pytest
import scipy.ndimage as ndi

from pyflowsolver.fastLaplacian import (
    _calculate_footprint,
    fast_laplacian_volume_generator,
)


def _fragmented_pore_mask(shape=(50, 50, 50), quantile=0.55, seed=42):
    """A fragmented pore space: many competing medium-radius blobs (no single
    dominant source). This is the regime that exposed the scatter write race --
    unlike one big open ball, where the largest source always wins regardless of
    thread ordering and the race stays invisible."""
    rng = np.random.default_rng(seed)
    noise = ndi.gaussian_filter(rng.random(shape), sigma=1.5)
    return noise > np.quantile(noise, quantile)


def _run_footprint(edt, spacing):
    out = np.zeros_like(edt, dtype=np.float32)
    _calculate_footprint(edt, out, tuple(float(s) for s in spacing))
    return out


def _footprint_reference(edt, spacing):
    """Independent O(N^2) ground truth: for each pore voxel, scan the WHOLE array
    (no window assumption) and take the max covering radius. Deliberately not the
    windowed algorithm under test, so it also validates the window bound."""
    nx, ny, nz = edt.shape
    sx, sy, sz = (float(s) for s in spacing)
    out = np.zeros_like(edt, dtype=np.float32)
    coords = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
    for (i, j, k) in coords:
        if edt[i, j, k] <= 0:
            continue
        best = 0.0
        for (ii, jj, kk) in coords:
            r = float(edt[ii, jj, kk])
            if r > best:
                dist = math.sqrt(((ii - i) * sx) ** 2
                                 + ((jj - j) * sy) ** 2
                                 + ((kk - k) * sz) ** 2)
                if dist <= r:
                    best = r
        out[i, j, k] = best
    return out


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (1.0, 2.0, 1.5)])
def test_footprint_matches_reference(spacing):
    rng = np.random.default_rng(0)
    edt = rng.integers(0, 4, size=(6, 6, 6)).astype(np.float32)
    got = _run_footprint(edt, spacing)
    ref = _footprint_reference(edt, spacing)
    np.testing.assert_array_equal(got, ref)


def test_footprint_is_deterministic():
    # Fragmented geometry with many competing sources whose coverage windows
    # overlap across prange (i-axis) thread slabs -- the exact condition under
    # which the old scatter form raced (empirically 8/8 runs diverged, up to
    # ~0.8). The result must be bit-identical across repeated runs.
    edt = ndi.distance_transform_edt(_fragmented_pore_mask()).astype(np.float32)
    spacing = (1.0, 1.0, 1.0)
    first = _run_footprint(edt, spacing)
    for _ in range(9):
        np.testing.assert_array_equal(_run_footprint(edt, spacing), first)


def test_footprint_at_least_edt_on_pores():
    rng = np.random.default_rng(2)
    edt = rng.integers(0, 5, size=(10, 10, 10)).astype(np.float32)
    out = _run_footprint(edt, (1.0, 1.0, 1.0))
    pore = edt > 0
    assert np.all(out[pore] >= edt[pore])   # a voxel always covers itself
    assert np.all(out[~pore] == 0.0)        # non-pore voxels never get a footprint


def test_enhanced_generator_end_to_end_deterministic():
    # The full enhanced-Arns conductivity path must be reproducible run to run,
    # on the fragmented geometry that previously diverged.
    por = (_fragmented_pore_mask((40, 40, 40)) * 100).astype(np.uint8)
    scale = np.array([1.0, 1.0, 1.0])
    a = fast_laplacian_volume_generator(
        por, scale, closed_border=False, enhanced_model=True)
    b = fast_laplacian_volume_generator(
        por, scale, closed_border=False, enhanced_model=True)
    np.testing.assert_array_equal(a, b)
