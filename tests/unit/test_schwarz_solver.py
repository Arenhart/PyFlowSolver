"""Unit tests for the distributed-ready iterative Schwarz Poisson solver.

The core guarantees: per-slab assembly (no global matrix -- the RAM win)
reproduces the single-domain system exactly, and additive Schwarz converges to
the same solution as the single-domain multigrid solve, on both channel and
complex geometry, for any number of slabs.
"""

import numpy as np
import pytest

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.multigridSolver import MultigridSolver
from pyflowsolver.schwarzSolver import (
    SchwarzSolver, global_fluid_index, slab_ranges, assemble_slab)
from tests.unit.resources.ducts import make_circular_duct


def _single_domain(vol):
    vm = VolumeManager((vol > 0).astype(np.float64), scale=1.0)
    a, b = vm.get_sparse_system_jit()
    mg = MultigridSolver(backend="native", target_error=1e-10)
    mg.set_linear_system(a, b)
    mg.generate_preconditioner()
    x, _, _ = mg.solve_pcg()
    return x, a, b


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_global_fluid_index_matches_c_order_count():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = mask[2, 2, 2] = True
    gidx, n = global_fluid_index(mask)
    assert n == 3
    assert gidx[0, 0, 0] == 0 and gidx[1, 1, 1] == 1 and gidx[2, 2, 2] == 2
    assert (gidx[~mask] == -1).all()


def test_slab_ranges_partition_cover():
    rngs = slab_ranges(10, 3)
    assert rngs[0][0] == 0 and rngs[-1][1] == 10
    # contiguous, non-overlapping, covering
    for (a, b), (c, d) in zip(rngs, rngs[1:]):
        assert b == c


def test_single_slab_matrix_matches_volume_manager():
    """A single slab spanning the whole volume must reproduce VolumeManager's
    bare-Laplacian system row-for-row (same solution)."""
    vol = make_circular_duct(4, 6)
    mask = vol > 0
    gidx, n = global_fluid_index(mask)
    slab = assemble_slab(mask, gidx, 0, vol.shape[2])

    _, a_ref, b_ref = _single_domain(vol)
    assert slab["local_n"] == b_ref.size
    assert slab["halo_rows"].size == 0          # whole domain -> no halo
    # Solve the local system and compare to the reference solution.
    mg = MultigridSolver(backend="native", target_error=1e-10)
    mg.set_linear_system(slab["a_sparse"], slab["b_base"].copy())
    mg.generate_preconditioner()
    x, _, _ = mg.solve_pcg()
    x_ref, _, _ = _single_domain(vol)
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------- #
# Schwarz convergence to the single-domain solution
# --------------------------------------------------------------------------- #
def test_single_partition_is_exact():
    vol = make_circular_duct(4, 8)
    x_ref, _, _ = _single_domain(vol)
    x, rounds, _ = SchwarzSolver(vol, n_partitions=1, target_error=1e-8).solve_serial()
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-7)
    assert rounds <= 3                          # exact block solve, no coupling


@pytest.mark.parametrize("parts", [2, 4])
def test_multi_slab_converges_to_single_domain_duct(parts):
    vol = make_circular_duct(5, 16)
    x_ref, _, _ = _single_domain(vol)
    solver = SchwarzSolver(vol, n_partitions=parts, target_error=1e-8,
                           max_rounds=3000)
    x, rounds, change = solver.solve_serial()
    assert x.size == x_ref.size
    assert change < 1e-8
    rel = np.linalg.norm(x - x_ref) / (np.linalg.norm(x_ref) or 1)
    assert rel < 1e-5


def test_multi_slab_converges_on_complex_geometry():
    import porespy as ps
    import scipy.ndimage as ndi
    im = ps.generators.blobs(shape=(20, 20, 20), porosity=0.5,
                             blobiness=1.0, seed=2)
    lab, _ = ndi.label(im)
    im = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    vol = im.astype(np.float64)

    x_ref, _, _ = _single_domain(vol)
    solver = SchwarzSolver(vol, n_partitions=3, target_error=1e-8, max_rounds=3000)
    x, rounds, change = solver.solve_serial()
    rel = np.linalg.norm(x - x_ref) / (np.linalg.norm(x_ref) or 1)
    assert rel < 1e-5


def test_coarse_correction_reduces_rounds():
    """The two-level (coarse-grid) correction must converge in far fewer rounds
    than plain one-level Schwarz for a many-slab z-driven decomposition, and to
    the same solution."""
    vol = make_circular_duct(5, 24)
    x_ref, _, _ = _single_domain(vol)
    one = SchwarzSolver(vol, n_partitions=8, target_error=1e-8,
                        max_rounds=5000, use_coarse=False)
    two = SchwarzSolver(vol, n_partitions=8, target_error=1e-8,
                        max_rounds=5000, use_coarse=True)
    _, rounds_one, _ = one.solve_serial()
    x2, rounds_two, _ = two.solve_serial()
    assert rounds_two < rounds_one // 2          # substantial acceleration
    rel = np.linalg.norm(x2 - x_ref) / (np.linalg.norm(x_ref) or 1)
    assert rel < 1e-5


def test_warm_start_reduces_rounds():
    """Seeding from the converged field must converge almost immediately -- the
    distributed driver relies on warm starts between rounds."""
    vol = make_circular_duct(5, 16)
    solver = SchwarzSolver(vol, n_partitions=4, target_error=1e-8, max_rounds=3000)
    x, rounds_cold, _ = solver.solve_serial()
    _, rounds_warm, _ = solver.solve_serial(initial=x)
    assert rounds_warm < rounds_cold
    assert rounds_warm <= 3
