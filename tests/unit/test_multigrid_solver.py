"""Unit tests for MultigridSolver (algebraic multigrid Poisson/Darcy solver).

The solver is a drop-in sibling of DarcySolver, so the core checks are: it
recovers the same solution as diagonal-PCG on the project's CSR systems (both
VolumeManager voxels and NetworkManager pore networks), the native V-cycle and
MG-PCG paths both converge, and the DarcySolver interface (set_linear_system /
generate_preconditioner / solve_pcg) is honoured -- including the scipy-format
row_ptr and the symmetric-negative-definite Laplacian sign flip.
"""

import numpy as np
import pytest

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.networkManager import NetworkManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.multigridSolver import (
    MultigridSolver,
    _standard_aggregation,
    _tentative_prolongator,
)
from tests.unit.resources.ducts import make_circular_duct


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _duct_poisson_system(radius=6, length=16):
    """Bare-Laplacian condensed CSR + RHS, as StokesSolver assembles it."""
    volume = make_circular_duct(radius, length)
    vm = VolumeManager((volume > 0).astype(np.float64), scale=1.0)
    a_sparse, b = vm.get_sparse_system_jit()
    return a_sparse, b


def _cubic_network(nx, ny, nz):
    """A cubic pore lattice: pores on a grid, throats to +x/+y/+z neighbours.

    Inlets on the x=0 face, outlets on the x=nx-1 face -> a percolating,
    Laplacian pore-network system for NetworkManager.
    """
    def idx(x, y, z):
        return (x * ny + y) * nz + z

    n_pores = nx * ny * nz
    conn, cond = [], []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = idx(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if xx < nx and yy < ny and zz < nz:
                        j = idx(xx, yy, zz)
                        conn.append((i, j))
                        cond.append(1.0 + 0.1 * ((i * 7 + j) % 5))  # mild variation
    inlets = np.zeros(n_pores, dtype=bool)
    outlets = np.zeros(n_pores, dtype=bool)
    for y in range(ny):
        for z in range(nz):
            inlets[idx(0, y, z)] = True
            outlets[idx(nx - 1, y, z)] = True
    return (np.array(conn, dtype=np.int64), np.array(cond, dtype=np.float64),
            inlets, outlets)


def _reference_pcg(a_sparse, b, tol=1e-9):
    solver = DarcySolver(target_error=tol)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = solver.solve_pcg()
    return x


# --------------------------------------------------------------------------- #
# Aggregation building blocks
# --------------------------------------------------------------------------- #
def test_aggregation_covers_every_node_once():
    # Small path graph: 0-1-2-3-4 (scipy-CSR indptr/indices, diagonal included).
    n = 5
    rows, cols = [], []
    for i in range(n):
        rows.append(i); cols.append(i)               # diagonal
        if i > 0:
            rows.append(i); cols.append(i - 1)
        if i < n - 1:
            rows.append(i); cols.append(i + 1)
    order = np.lexsort((np.array(cols), np.array(rows)))
    cols = np.array(cols)[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    for r in np.array(rows)[order]:
        indptr[r + 1] += 1
    indptr = np.cumsum(indptr)

    agg, n_coarse = _standard_aggregation(indptr, cols.astype(np.int64))
    assert agg.min() >= 0                # every node aggregated
    assert n_coarse == agg.max() + 1
    assert n_coarse < n                  # genuine coarsening
    assert set(agg.tolist()) == set(range(n_coarse))


def test_tentative_prolongator_column_normalized():
    agg = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    P = _tentative_prolongator(agg, 2).toarray()
    # Each column is the normalized indicator of its aggregate (unit 2-norm).
    np.testing.assert_allclose(np.linalg.norm(P, axis=0), 1.0)
    assert np.isclose(P[0, 0], 1 / np.sqrt(2))
    assert np.isclose(P[2, 1], 1 / np.sqrt(3))


# --------------------------------------------------------------------------- #
# Native backend: correctness vs diagonal-PCG
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("aggregation", ["smoothed", "unsmoothed"])
def test_native_matches_pcg_on_duct(aggregation):
    a_sparse, b = _duct_poisson_system()
    x_ref = _reference_pcg(a_sparse, b)

    solver = MultigridSolver(backend="native", aggregation=aggregation,
                             target_error=1e-9)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()
    x, err, iters = solver.solve_pcg()

    assert err <= 1e-8
    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-6)


def test_native_manufactured_solution():
    """A x_exact = b must solve back to x_exact (matrix is non-singular)."""
    a_sparse, _ = _duct_poisson_system()
    n = a_sparse["row_ptr"].size
    rng = np.random.default_rng(0)
    x_exact = rng.standard_normal(n)

    # b = A x_exact via the same condensed-CSR convention.
    val, col, rp = a_sparse["val"], a_sparse["col_idx"], a_sparse["row_ptr"]
    b = np.zeros(n)
    for row in range(n):
        stop = rp[row + 1] if row + 1 < n else val.size
        for k in range(rp[row], stop):
            b[row] += val[k] * x_exact[col[k]]

    solver = MultigridSolver(backend="native", target_error=1e-10)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()
    x, err, _ = solver.solve_pcg()
    np.testing.assert_allclose(x, x_exact, rtol=1e-5, atol=1e-6)


def test_native_standalone_vcycle_converges():
    a_sparse, b = _duct_poisson_system()
    x_ref = _reference_pcg(a_sparse, b)

    solver = MultigridSolver(backend="native", target_error=1e-9)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()
    x, err, iters = solver.solve_mg()

    assert err <= 1e-8
    assert iters >= 1
    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-6)


def test_native_matches_pcg_on_network():
    """MultigridSolver works on a NetworkManager (pore-network) CSR system."""
    conn, cond, inlets, outlets = _cubic_network(8, 5, 5)
    nm = NetworkManager(conn, cond, inlets, outlets)
    nm.generate_sparse_system()
    a_sparse, b = nm.get_sparse_system()

    x_ref = _reference_pcg(a_sparse, b)

    solver = MultigridSolver(backend="native", target_error=1e-9)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()
    x, err, _ = solver.solve_pcg()

    assert err <= 1e-8
    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Interface parity with DarcySolver
# --------------------------------------------------------------------------- #
def test_accepts_scipy_format_row_ptr():
    """set_linear_system must accept the scipy N+1 row_ptr convention too."""
    a_sparse, b = _duct_poisson_system()
    n = b.size
    scipy_rp = np.empty(n + 1, dtype=np.int64)
    scipy_rp[:-1] = a_sparse["row_ptr"]
    scipy_rp[-1] = a_sparse["val"].size
    a_scipy = {"val": a_sparse["val"], "col_idx": a_sparse["col_idx"],
               "row_ptr": scipy_rp}

    solver = MultigridSolver(backend="native", target_error=1e-9)
    solver.set_linear_system(a_scipy, b)
    solver.generate_preconditioner()
    x, err, _ = solver.solve_pcg()

    np.testing.assert_allclose(x, _reference_pcg(a_sparse, b), rtol=1e-5, atol=1e-6)


def test_negative_definite_sign_flip_detected():
    a_sparse, b = _duct_poisson_system()
    solver = MultigridSolver(backend="native")
    solver.set_linear_system(a_sparse, b)
    # Assembled Laplacian has a negative diagonal -> solver flips to SPD.
    assert solver._spd_sign == -1.0
    assert solver._A_spd.diagonal().sum() > 0


def test_solve_lazily_builds_hierarchy():
    a_sparse, b = _duct_poisson_system()
    solver = MultigridSolver(backend="native", target_error=1e-9)
    solver.set_linear_system(a_sparse, b)
    assert solver.hierarchy is None
    solver.solve_pcg()                 # builds hierarchy on first solve
    assert solver.hierarchy is not None


def test_invalid_backend_and_aggregation_raise():
    with pytest.raises(Exception):
        MultigridSolver(backend="bogus")
    with pytest.raises(Exception):
        MultigridSolver(aggregation="bogus")


def test_warm_start_reduces_iterations():
    """Warm-starting MG-PCG from a near-solution cuts the iteration count."""
    a_sparse, b = _duct_poisson_system()
    solver = MultigridSolver(backend="native", target_error=1e-9)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()

    x_cold, _, it_cold = solver.solve_pcg()
    x_warm, _, it_warm = solver.solve_pcg(X0=x_cold.copy())
    assert it_warm < it_cold


# --------------------------------------------------------------------------- #
# pyamg benchmark backend (optional dependency)
# --------------------------------------------------------------------------- #
def test_pyamg_backend_matches_native_when_available():
    pyamg = pytest.importorskip("pyamg")
    a_sparse, b = _duct_poisson_system()
    x_ref = _reference_pcg(a_sparse, b)

    solver = MultigridSolver(backend="pyamg", target_error=1e-9)
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner()
    x, err, _ = solver.solve_pcg()

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-6)
