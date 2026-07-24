# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyFlowSolver is a CFD laminar-flow solver for porous media and pore networks, built on custom Numba-JIT sparse linear algebra (no scipy sparse matrices in the hot paths). It provides **two independent flow models**:

- **Fast-Laplacian (Darcy)** — solves `∇·(k∇p)=0` on the voxel grid with EDT-estimated conductivities (Arns approximation). Cheap and robust; gives a pressure field and an effective permeability.
- **Stokes** — pore-resolved steady viscous flow via a projection / fractional-step method on a MAC staggered grid. Gives velocity and pressure fields.

For large volumes there are also RAM-scaling **distributed** paths (z-slab Schwarz / Dask).

## Commands

```bash
pip install -e .                                          # install in dev mode
pip install -r requirements.txt                           # pinned deps
pytest tests/unit/                                        # run all tests
pytest tests/unit/test_stokes_solver.py                   # one file
pytest tests/unit/test_stokes_solver.py -k seed_from_pressure   # one test (by keyword)
python scripts/run_stokes_duct.py                         # Stokes duct example + HP check
python scripts/compare_solver.py                          # Darcy vs Stokes vs OpenFOAM, cold vs warm
```

Note: `test_flow_solver.py::test_darcy_solver` xfails on a tight rtol (fails on base too) — unrelated pre-existing issue.

## Architecture

### Two solving pipelines

**Darcy:** Geometry (`VolumeManager` from a voxel volume, or `NetworkManager` from pore connectivity) → condensed sparse CSR system `Ax=b` → solve with `DarcySolver` (CG/PCG) or `MultigridSolver` (AMG-PCG). `VolumeManager.get_conductivity(pressure)` returns the effective permeability.

**Stokes:** raw voxel array → `StokesSolver` assembles its *own* MAC operators and iterates a pseudo-transient projection (predictor → pressure-Poisson → corrector) to steady state. It is **model-independent of the Darcy code** by design (see below), with one bridge: it can warm-start from a fast-Laplacian pressure.

### Custom CSR sparse format

Used throughout as Python dicts, **not** scipy sparse matrices:
```python
{"val": np.array, "col_idx": np.array, "row_ptr": np.array}
```
`row_ptr` has N elements (not N+1 like scipy). The last row's values run from `row_ptr[-1]` to `val.size`. Scipy-format (N+1) is accepted via a flag in `DarcySolver.set_linear_system()`. `MultigridSolver` shares this interface (`set_linear_system` / `generate_preconditioner` / `solve_pcg(X0)`), so it drops in wherever `DarcySolver` is used.

### Key modules (`pyflowsolver/`)

- **`solver.py`** — trivial `Solver` base class; `DarcySolver`, `MultigridSolver`, `StokesSolver` all subclass it.
- **`darcySolver.py`** — CG/PCG solvers. Linear-algebra helpers (`_scalar_product`, `_square_sum_vector`, …) are module-level `@njit` functions, not methods.
- **`multigridSolver.py`** — algebraic-multigrid PCG. Two backends: `"native"` (self-contained smoothed-aggregation AMG with `@njit` V-cycle kernels — the deliverable) and `"pyamg"` (optional benchmark). Size-independent iteration count; ~18–30× faster than diagonal-PCG on large Laplacians. **Sign convention:** the assembled Laplacian is symmetric *negative*-definite; MG/CG need SPD, so it auto-detects (`diag.sum() < 0`) and solves `(-A)x = (-b)`.
- **`volumeManager.py`** — converts a 3D voxel volume into the condensed Darcy CSR system (see "VolumeManager internals" below). Python (`get_sparse_system`) and JIT (`get_sparse_system_jit`) assembly paths. Filters disconnected pore regions via `scipy.ndimage.label`.
- **`networkManager.py`** — assembles CSR systems from pore connectivity (`conn`) and conductance (`cond`) arrays; only mid-pores (not inlet/outlet) become unknowns.
- **`fastLaplacian.py`** — estimates per-voxel conductivity from a porosity map using the Euclidean Distance Transform (`pyedt`); supports the enhanced (footprint/local-thickness) Arns model **and subresolution porosity** (see below).
- **`pressurePoisson.py`** — model-neutral unit-MAC-Laplacian (incompressibility operator) assembly used by `StokesSolver` and `schwarzSolver`. **No dependency on the Darcy/fast-Laplacian code**; builds the unit-conductivity Laplacian directly from a boolean fluid mask. `filter_percolating`, `assemble_poisson`, `assemble_slab` (distributed), `ravel`.
- **`stokesSolver.py`** — Stokes projection solver (fully implemented; details below).
- **`schwarzSolver.py`** — distributed-ready additive-Schwarz pressure-Poisson solver: splits geometry into z-slabs, each slab assembles/solves only its subdomain, so peak RAM/node scales as global/n_slabs. Breaks the single-node RAM ceiling at the cost of Schwarz rounds. `solve_serial`, `solve_streaming`.
- **`distributedDarcySolver.py`** — earlier distributed Darcy path (Docker/Dask workers, frozen-boundary block solve); `schwarzSolver` is its convergent evolution.
- **`pressureEstimator.py`** — coarse pseudo-network pressure estimate (segment pore space → pseudo-network solve → smoothed pressure volume); used to seed distributed solves.
- **`sparseArray.py` / `velocityArray.py`** — CSR wrapper classes with row iteration / element access.
- **`constants.py`** — `SOLID=0, PORE=1, INLET=2, OUTLET=3`.

### Boundary conditions

- **Regular volumes** (no `boundary_volume`): z=0 is inlet (pressure=1), z=max is outlet (pressure=0).
- **Irregular boundaries**: a separate `boundary_volume` uint8 array marks SOLID/PORE/INLET/OUTLET per voxel; only PORE voxels enter the system. (Darcy path only — `StokesSolver` is regular-only.)
- Neighbour conductivity uses the **harmonic mean** `2/(1/c1 + 1/c2)`.

### Numba patterns

- Performance-critical code uses `@njit` and `@njit(parallel=True)` with `prange`.
- JIT functions must be module-level; classes wrap them via `@staticmethod @njit`.
- First invocation compiles (slow); subsequent calls are fast. Keep this in mind when timing.

### HPC (`docker/`)

Docker Compose cluster (1 master + 2 workers) with SLURM and Dask for distributed computation.

## StokesSolver details

`StokesSolver(volume, scale=1.0, ...)` — regular (z-driven) geometry only; `volume > 0` is pore. Iterates predictor → pressure-Poisson → corrector to steady state. Key options:

- **`predictor`** — `"explicit"` (forward-Euler diffusion, viscous-dt-limited; the correct/robust choice on **complex media**) or `"implicit"` (backward-Euler per component via multigrid, large-dt; ~size-independent on **channels** but stalls on complex media, where it hands off to explicit). Explicit iteration count scales ~O(N²) with domain size — the main large-volume cost.
- **Stopping** — converges when the step-residual drops below `target_error` **or** it plateaus (stagnation stop: `stagnation_window`, `stagnation_tol`). The plateau stop is a robust backstop because the residual bottoms out at a noise floor (~1e-7) above tight targets like 1e-8; without it the loop would grind to `max_iterations`. Reports `stop_reason` and `stagnated`.
- **Warm start** — `initial_pressure` / `initial_velocity` (a MAC `(u,v,w)` tuple, or the string `"from_pressure"`). `"from_pressure"` derives a no-slip-correct seed velocity by solving the steady viscous momentum equation for the given pressure (`-ν∇²u=-∇p/ρ`), which cuts the iteration count on complex media.
- **`fast_laplacian_guess=True` (default)** — when no explicit guess is given, `solve()` computes an enhanced fast-Laplacian pressure internally and warm-starts from it (via the `"from_pressure"` seed). This **couples the default solve to `VolumeManager`/`fastLaplacian` (pulls `pyedt`), imported lazily** in `_compute_fast_laplacian_guess`. Pass `fast_laplacian_guess=False` for a pure cold start with no Darcy/EDT dependency. Cold-baseline / predictor-comparison tests set it `False`.
- **`backend`** — `"native"` or `"pyamg"` for the internal multigrid solves.

The hot loops (`_diffuse_jit`, `_divergence_jit`, `_apply_pressure_gradient_jit`, diffusion CSR assembly) are module-level `@njit`. The pressure-Poisson operator comes from `pressurePoisson.assemble_poisson` (not VolumeManager) — this is the deliberate Darcy-independence.

## VolumeManager internals & subresolution porosity (for the next step)

**Next step: a `MultiscaleVolumeManager(VolumeManager)` for a Stokes-Brinkman solver**, to handle images with **subresolution porosity** (voxels that are neither fully solid nor fully open). Below is what already exists to build on. *(This section documents current code only — it does not cover the Brinkman model.)*

### How VolumeManager builds the Darcy system

- `self.volume` is a **per-voxel scalar conductivity** field (float). `0` = solid/excluded; any `>0` value is an unknown (a matrix row).
- `filter_connected_volume()` (in `__init__`) zeroes pore clusters not connected inlet↔outlet (via `scipy.ndimage.label`).
- `_calc_null_counts` builds `nulls_count`, a running count of skipped (zero) voxels used by `_unravel` to map a 3D index to its **condensed** row index — so only `>0` voxels occupy rows (`self.nonzeros` of them). Low-porosity samples stay cheap.
- Assembly (`get_sparse_system` / `get_sparse_system_jit`): for each fluid voxel, off-diagonal entries are the **harmonic-mean face conductivity** `2/(1/c_center + 1/c_neighbour)` to each fluid neighbour; the diagonal is `-Σ face_c` (plus the inlet/outlet Dirichlet contribution). z=0 inlet (`p=1`) is folded into `b` as `-2·c_center` on those rows; z=max outlet is `p=0`.
- `ravel_sparse_solution(x)` scatters the condensed solution back to a `(w,h,d)` field. `get_conductivity(pressure)` integrates the inlet/outlet flux → effective permeability.

### Where conductivity comes from — and the subresolution hook

`convert_pore_volume_to_laplacian_conductivity(porosity_map=False, enhanced_model=True)` replaces `self.volume` with per-voxel conductivity via `fastLaplacian.fast_laplacian_volume_generator`. **Currently it binarizes** the input: `(self.volume >= 1) * 100`, i.e. every non-empty voxel is treated as a fully-open pore (porosity 100) and **subresolution values are discarded**. The `porosity_map=True` branch is stubbed **"Not implemented yet"** — this is the extension point.

`fast_laplacian_volume_generator(porosity_volume, pore_scale, subresolution_function=None, enhanced_model=...)` **already supports subresolution porosity** under this convention:

- `porosity_volume`: uint-ish map where `0` = solid, `100` = fully open pore, and `1–99` = **subresolution** voxels with that % porosity.
- It splits the map: `stokes_pores = (porosity == 100)` get an **EDT/footprint-based** conductance (the Arns model); `darcy_pores = 0 < porosity < 100` get a conductance from `subresolution_function(porosity)` (default `(porosity/100)·(min(scale)/10)²`). The two contributions are summed into one conductance field.

So the infrastructure for mixed resolved/subresolution media is present in `fastLaplacian`; `VolumeManager` simply doesn't route it through yet. A `MultiscaleVolumeManager` would:

1. **Preserve the porosity map** (do not binarize) — keep the `1–99` values.
2. Enable the `porosity_map=True` path in `convert_pore_volume_to_laplacian_conductivity`, forwarding the real porosity map and a `subresolution_function` to `fast_laplacian_volume_generator`.
3. Reuse the inherited condensation/assembly/`get_conductivity` machinery unchanged (it already operates on an arbitrary per-voxel conductivity field).

Follow the `SOLID=0 / PORE=100 / 1–99=subresolution` convention and the harmonic-mean face rule already used throughout.

## Key dependencies

numpy, numba, scipy, pyedt (EDT), porespy (test geometry), h5py + netCDF (OpenFOAM validation data); optional: pyamg (MG benchmark backend), dask/docker (distributed). See `requirements.txt`.
