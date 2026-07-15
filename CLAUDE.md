# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyFlowSolver is a CFD laminar flow solver for porous media and pore networks. It solves Darcy flow and Stokes flow using custom sparse linear algebra with Numba JIT compilation.

## Commands

```bash
pip install -e .                                      # Install in dev mode
pip install -r requirements.txt                       # Install pinned deps
pytest tests/unit/                                    # Run all tests
pytest tests/unit/test_flow_solver.py                 # Run one test file
pytest tests/unit/test_flow_solver.py::test_darcy_solver  # Run single test
python -m pyflowsolver                                # Run as module
```

## Architecture

### Solving Pipeline: Geometry -> Sparse Linear System -> Solve

1. **Geometry**: A 3D voxel volume (`VolumeManager`) or pore network (`NetworkManager`) defines the domain
2. **Assembly**: Geometry is converted to a sparse CSR linear system `Ax = b`
3. **Solve**: `DarcySolver` solves via Conjugate Gradient (CG) or Preconditioned CG (PCG)

### Custom CSR Sparse Format

Used throughout as Python dicts, **not** scipy sparse matrices:
```python
{"val": np.array, "col_idx": np.array, "row_ptr": np.array}
```
`row_ptr` has N elements (not N+1 like scipy). The last row's values run from `row_ptr[-1]` to `val.size`. Scipy-format (N+1) is accepted via a flag in `DarcySolver.set_linear_system()`.

### Key Modules (`pyflowsolver/`)

- **`darcySolver.py`** - CG/PCG solvers. All linear algebra helpers (`_scalar_product`, `_square_sum_vector`, etc.) are module-level `@njit` functions, not methods.
- **`volumeManager.py`** - Converts 3D volumes into sparse systems. Has both Python (`get_sparse_system`) and JIT (`get_sparse_system_jit`) assembly paths. Filters disconnected pore regions via `scipy.ndimage.label`.
- **`networkManager.py`** - Assembles sparse systems from pore connectivity (`conn`) and conductance (`cond`) arrays. Only mid-pores (not inlet/outlet) enter the linear system.
- **`fastLaplacian.py`** - Estimates voxel conductivity from porosity maps using Euclidean Distance Transform (`pyedt`).
- **`sparseArray.py`** - CSR wrapper class with row iteration and element access.
- **`constants.py`** - `SOLID=0, PORE=1, INLET=2, OUTLET=3`
- **`stokesSolver.py`** - Stokes flow (partially implemented).

### Boundary Conditions

- **Regular volumes** (no `boundary_volume`): z=0 is inlet (pressure=1), z=max is outlet (pressure=0)
- **Irregular boundaries**: separate `boundary_volume` uint8 array marks SOLID/PORE/INLET/OUTLET per voxel; only PORE voxels enter the system
- Neighbor conductivity uses harmonic mean: `2 / (1/c1 + 1/c2)`

### Numba Patterns

- Performance-critical code uses `@njit` and `@njit(parallel=True)` with `prange`
- JIT functions must be module-level due to Numba constraints; classes wrap them via `@staticmethod @njit`
- First invocation triggers compilation (slow); subsequent calls are fast

### HPC (`docker/`)

Docker Compose cluster (1 master + 2 workers) with SLURM and Dask for distributed computation.

## Key Dependencies

numpy, numba, scipy, porespy (test data generation), pyedt (EDT), pytest
