import numpy as np
import pytest

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.distributedDarcySolver import (
    DistributedDarcySolver,
    partition_csr_system,
    BOUNDARY_ESTIMATE,
)
from pyflowsolver.pressureEstimator import (
    segment_pore_space,
    create_pseudo_network,
    solve_pseudo_network,
    create_pressure_volume,
    estimate_pressure_distribution,
)
from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET

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
# Step 1: Segment pore space
# ---------------------------------------------------------------------------

def test_segment_pore_space_returns_correct_shape():
    """Segmented volume must have the same shape as the input."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    assert segmented.shape == boundary_volume.shape


def test_segment_pore_space_labels_are_positive_integers():
    """All segment labels must be non-negative; 0 = solid, >0 = segment."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    assert segmented.dtype in (np.int32, np.int64)
    assert np.all(segmented >= 0)


def test_segment_pore_space_solid_voxels_unlabeled():
    """SOLID voxels must have segment label 0."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    solid_mask = boundary_volume == SOLID
    assert np.all(segmented[solid_mask] == 0)


def test_segment_pore_space_covers_all_pore_voxels():
    """Every PORE, INLET, and OUTLET voxel must belong to a segment (label > 0)."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    nonsolid_mask = boundary_volume != SOLID
    assert np.all(segmented[nonsolid_mask] > 0)


def test_segment_pore_space_n_segments_matches_labels():
    """n_segments must equal the number of unique positive labels."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    unique_labels = np.unique(segmented)
    unique_positive = unique_labels[unique_labels > 0]
    assert n_segments == unique_positive.size


# ---------------------------------------------------------------------------
# Step 2: Create pseudo-network
# ---------------------------------------------------------------------------

def test_create_pseudo_network_conn_shape():
    """conn must be (n_throats, 2) with integer dtype."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    assert conn.ndim == 2
    assert conn.shape[1] == 2
    assert np.issubdtype(conn.dtype, np.integer)


def test_create_pseudo_network_cond_shape():
    """cond must have length n_throats with positive values."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    assert cond.shape == (conn.shape[0],)
    assert np.all(cond > 0)


def test_create_pseudo_network_inlets_outlets_shape():
    """inlets and outlets must be boolean arrays of length n_segments."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    assert inlets.size == n_segments
    assert outlets.size == n_segments
    assert inlets.dtype == bool
    assert outlets.dtype == bool


def test_create_pseudo_network_has_inlets_and_outlets():
    """Network must have at least one inlet and one outlet segment."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    assert inlets.sum() >= 1
    assert outlets.sum() >= 1


def test_create_pseudo_network_no_self_connections():
    """No throat should connect a segment to itself."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    assert np.all(conn[:, 0] != conn[:, 1])


# ---------------------------------------------------------------------------
# Step 3a: Solve pseudo-network
# ---------------------------------------------------------------------------

def test_solve_pseudo_network_returns_pressures():
    """Must return a pressure array with length n_segments."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    assert pressures.size == n_segments


def test_solve_pseudo_network_inlet_pressure():
    """Inlet segments must have pressure close to 1.0."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    assert np.allclose(pressures[inlets], 1.0)


def test_solve_pseudo_network_outlet_pressure():
    """Outlet segments must have pressure close to 0.0."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    assert np.allclose(pressures[outlets], 0.0)


def test_solve_pseudo_network_pressures_in_range():
    """All pressures must be between 0 and 1."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    assert np.all(pressures >= 0.0)
    assert np.all(pressures <= 1.0)


# ---------------------------------------------------------------------------
# Step 3b: Create pressure volume
# ---------------------------------------------------------------------------

def test_create_pressure_volume_shape():
    """Pressure volume must have the same shape as the input volume."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    pressure_vol = create_pressure_volume(boundary_volume, segmented, pressures)
    assert pressure_vol.shape == boundary_volume.shape


def test_create_pressure_volume_inlet_values():
    """INLET voxels must have pressure = 1.0."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    pressure_vol = create_pressure_volume(boundary_volume, segmented, pressures)
    inlet_mask = boundary_volume == INLET
    assert np.allclose(pressure_vol[inlet_mask], 1.0)


def test_create_pressure_volume_outlet_values():
    """OUTLET voxels must have pressure = 0.0."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    pressure_vol = create_pressure_volume(boundary_volume, segmented, pressures)
    outlet_mask = boundary_volume == OUTLET
    assert np.allclose(pressure_vol[outlet_mask], 0.0)


def test_create_pressure_volume_solid_values():
    """SOLID voxels must have pressure = 0.0."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    pressure_vol = create_pressure_volume(boundary_volume, segmented, pressures)
    solid_mask = boundary_volume == SOLID
    assert np.allclose(pressure_vol[solid_mask], 0.0)


def test_create_pressure_volume_pore_values_in_range():
    """PORE voxels must have pressures between 0 and 1."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    pressure_vol = create_pressure_volume(boundary_volume, segmented, pressures)
    pore_mask = boundary_volume == PORE
    assert np.all(pressure_vol[pore_mask] >= 0.0)
    assert np.all(pressure_vol[pore_mask] <= 1.0)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def test_estimate_pressure_distribution_matches_steps():
    """Convenience function must produce the same result as running steps manually."""
    segmented, n_segments = segment_pore_space(boundary_volume)
    conn, cond, inlets, outlets = create_pseudo_network(
        boundary_volume, segmented, volume_manager.volume, scale,
    )
    pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    step_by_step = create_pressure_volume(boundary_volume, segmented, pressures)

    convenience = estimate_pressure_distribution(
        boundary_volume, volume_manager.volume, scale,
    )
    np.testing.assert_array_equal(convenience, step_by_step)


# ---------------------------------------------------------------------------
# Integration: pressure estimate improves distributed partitioning
# ---------------------------------------------------------------------------

def test_partition_with_pressure_estimate_differs_from_flat():
    """Using a pressure estimate must produce different b vectors than flat 0.5."""
    # Create a synthetic pressure estimate (linear gradient from 1 to 0)
    N = dense_b.size
    pressure_estimate = np.linspace(1.0, 0.0, N)

    partitions_flat = partition_csr_system(sparse_A, dense_b, n_partitions=2)
    partitions_est = partition_csr_system(
        sparse_A, dense_b, n_partitions=2, pressure_estimate=pressure_estimate,
    )

    # The b vectors should differ because the boundary values differ
    assert not np.allclose(
        partitions_flat[0]["b_array"],
        partitions_est[0]["b_array"],
    )


def test_partition_with_pressure_estimate_same_structure():
    """Pressure estimate must not change the CSR structure, only b values."""
    N = dense_b.size
    pressure_estimate = np.linspace(1.0, 0.0, N)

    partitions_flat = partition_csr_system(sparse_A, dense_b, n_partitions=2)
    partitions_est = partition_csr_system(
        sparse_A, dense_b, n_partitions=2, pressure_estimate=pressure_estimate,
    )

    for p_flat, p_est in zip(partitions_flat, partitions_est):
        # Same A matrix structure
        np.testing.assert_array_equal(
            p_flat["a_sparse_array"]["val"],
            p_est["a_sparse_array"]["val"],
        )
        np.testing.assert_array_equal(
            p_flat["a_sparse_array"]["col_idx"],
            p_est["a_sparse_array"]["col_idx"],
        )
        np.testing.assert_array_equal(
            p_flat["a_sparse_array"]["row_ptr"],
            p_est["a_sparse_array"]["row_ptr"],
        )
        # Same row ranges
        assert p_flat["global_row_start"] == p_est["global_row_start"]
        assert p_flat["global_row_end"] == p_est["global_row_end"]


def test_partition_with_none_pressure_estimate_uses_flat():
    """pressure_estimate=None must produce identical results to the default."""
    partitions_default = partition_csr_system(sparse_A, dense_b, n_partitions=2)
    partitions_none = partition_csr_system(
        sparse_A, dense_b, n_partitions=2, pressure_estimate=None,
    )

    for p_def, p_none in zip(partitions_default, partitions_none):
        np.testing.assert_array_equal(p_def["b_array"], p_none["b_array"])
