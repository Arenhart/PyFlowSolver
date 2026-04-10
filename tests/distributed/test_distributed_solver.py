import numpy as np
import pytest

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.distributedDarcySolver import (
    DistributedDarcySolver,
    partition_csr_system,
    BOUNDARY_ESTIMATE,
)
from pyflowsolver.constants import PORE, INLET, OUTLET

# ---------------------------------------------------------------------------
# Module-level setup: build a system using convert_pore_volume_to_laplacian_conductivity
# ---------------------------------------------------------------------------

boundary_volume = np.zeros((5, 5, 5), dtype=np.uint8)
boundary_volume[1:-1, :, 1:-1] = PORE
boundary_volume[1:-1, 0, 1:-1] = INLET
boundary_volume[1:-1, -1, 1:-1] = OUTLET

porosity_volume = (boundary_volume >= 1) * 100
scale = (0.02,) * 3

volume_manager = VolumeManager(
    porosity_volume,
    scale=scale,
    boundary_volume=boundary_volume,
)
volume_manager.convert_pore_volume_to_laplacian_conductivity()

sparse_A, dense_b = volume_manager.get_sparse_system_jit()

# Reference solution using standard DarcySolver
ref_solver = DarcySolver()
ref_solver.set_linear_system(sparse_A, dense_b)
ref_solver.generate_preconditioner(preconditioner="inverse_diagonal")
ref_solution, ref_error, ref_iteration = ref_solver.solve_pcg()


# ---------------------------------------------------------------------------
# Partitioning tests (no Docker required)
# ---------------------------------------------------------------------------

def test_partition_sizes_sum_to_n():
    """Partition sizes must sum to original system size N."""
    partitions = partition_csr_system(sparse_A, dense_b, n_partitions=2)
    total_size = sum(p["b_array"].size for p in partitions)
    assert total_size == dense_b.size


def test_single_partition_reproduces_original():
    """With n_partitions=1, the partitioned system must equal the original."""
    partitions = partition_csr_system(sparse_A, dense_b, n_partitions=1)
    assert len(partitions) == 1
    p = partitions[0]
    np.testing.assert_array_equal(p["a_sparse_array"]["val"], sparse_A["val"])
    np.testing.assert_array_equal(p["a_sparse_array"]["col_idx"], sparse_A["col_idx"])
    np.testing.assert_array_equal(p["a_sparse_array"]["row_ptr"], sparse_A["row_ptr"])
    np.testing.assert_array_equal(p["b_array"], dense_b)
    assert p["global_row_start"] == 0
    assert p["global_row_end"] == dense_b.size


def test_partition_local_csr_is_valid():
    """Each partition's CSR must be structurally valid."""
    partitions = partition_csr_system(sparse_A, dense_b, n_partitions=2)
    for p in partitions:
        sub_A = p["a_sparse_array"]
        sub_b = p["b_array"]
        local_n = sub_b.size
        # row_ptr has local_n elements (N-element format)
        assert sub_A["row_ptr"].size == local_n
        # val and col_idx same size
        assert sub_A["val"].size == sub_A["col_idx"].size
        # All column indices within local range
        if sub_A["col_idx"].size > 0:
            assert np.all(sub_A["col_idx"] >= 0)
            assert np.all(sub_A["col_idx"] < local_n)
        # row_ptr values are non-decreasing
        for i in range(1, local_n):
            assert sub_A["row_ptr"][i] >= sub_A["row_ptr"][i - 1]


def test_cross_partition_entries_modify_b():
    """Cross-partition column references must be moved into the b vector."""
    partitions = partition_csr_system(sparse_A, dense_b, n_partitions=2)
    # For a connected system, the first partition's b must differ from
    # the original slice because cross-partition entries contribute
    # -val * BOUNDARY_ESTIMATE to b.
    first_size = partitions[0]["b_array"].size
    original_b_slice = dense_b[:first_size]
    assert not np.allclose(
        partitions[0]["b_array"], original_b_slice
    ), "Expected cross-partition entries to modify b"


def test_single_partition_solve_matches_reference():
    """With n_partitions=1, the solve result must match the reference solution."""
    partitions = partition_csr_system(sparse_A, dense_b, n_partitions=1)
    p = partitions[0]
    solver = DarcySolver()
    solver.set_linear_system(p["a_sparse_array"], p["b_array"])
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    solution, _, _ = solver.solve_pcg()
    np.testing.assert_allclose(solution, ref_solution, rtol=1e-06)


def test_partition_boundary_values():
    """Verify the BOUNDARY_ESTIMATE constant is 0.5."""
    assert BOUNDARY_ESTIMATE == 0.5


def test_partitions_cover_all_rows():
    """Partition row ranges must be contiguous and cover [0, N)."""
    partitions = partition_csr_system(sparse_A, dense_b, n_partitions=3)
    starts = [p["global_row_start"] for p in partitions]
    ends = [p["global_row_end"] for p in partitions]
    assert starts[0] == 0
    assert ends[-1] == dense_b.size
    for i in range(1, len(partitions)):
        assert starts[i] == ends[i - 1]


# ---------------------------------------------------------------------------
# Distributed solve test (requires Docker)
# ---------------------------------------------------------------------------

@pytest.mark.docker
def test_distributed_solver():
    """End-to-end distributed solve versus single-node reference.

    Requires the Docker cluster to be available. Run with:
        pytest -m docker tests/distributed/
    """
    dist_solver = DistributedDarcySolver(n_partitions=2)
    dist_solver.set_linear_system(sparse_A, dense_b)
    dist_x, dist_error, dist_iteration = dist_solver.solve_distributed()

    assert dist_x.size == ref_solution.size
    # Relaxed tolerance due to BOUNDARY_ESTIMATE approximation at partition edges
    np.testing.assert_allclose(dist_x, ref_solution, rtol=0.5, atol=0.15)
