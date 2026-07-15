# Stokes (Creeping) Flow Solver (`stokesSolver.py`)

This document describes the algorithm we intend to implement in
`pyflowsolver/stokesSolver.py`. It solves **steady, incompressible, laminar
Stokes flow** on the same voxel geometry used by the Darcy solver.

The design goal is **simplicity**: a small, readable baseline we can use as a
starting point to explore improvement strategies. We deliberately solve the
*Stokes* equations (no inertia / no advection), not the full Navier–Stokes
equations. Dropping the nonlinear advection term `(u·∇)u` makes the problem
**linear**, which removes CFL/upwinding/divergence-retry machinery and lets us
lean heavily on the existing `DarcySolver`.

The module mirrors the structure of `darcySolver.py`: `DEFAULT_PARAMS` +
`**params` validation, a public `solve` driver, and module-level `@njit`
kernels for the hot loops.

---

## 1. Governing equations

Steady, incompressible, inertia-free (creeping) flow:

```
momentum:    0 = -(1/ρ)∇p + ν ∇²u + f
continuity:  ∇·u = 0
```

- `u = (u, v, w)` — velocity field
- `p` — pressure
- `ρ` — density
- `ν = μ/ρ` — kinematic viscosity
- `f` — body force per unit mass (optional driving force)

There is **no `(u·∇)u` term** and **no `∂u/∂t` term** — the flow is quasi-static
and linear in `(u, p)`. This is the correct regime for low-Reynolds pore-scale
flow, and it is the natural next step up in fidelity from Darcy's law.

---

## 2. Spatial discretization — MAC staggered grid

We discretize on a **Marker-And-Cell (MAC) staggered grid** over the voxels:

- **Pressure `p`** at cell (voxel) centers → shape `(w, h, d)`.
- **Velocity components on cell faces** (staggering avoids the checkerboard
  pressure decoupling of a collocated grid):
  - `u` on x-faces → shape `(w+1, h, d)`
  - `v` on y-faces → shape `(w, h+1, d)`
  - `w` on z-faces → shape `(w, h, d+1)`

Operators use **second-order central differences**:

- Diffusion `ν∇²u`: standard 7-point Laplacian per component.
- Divergence `∇·u` / gradient `∇p`: natural on the staggered layout.

Grid spacing comes from `volume_manager.scale` (`dx, dy, dz`).

### Geometry / masks

Reusing `constants.py` (`SOLID=0, PORE=1, INLET=2, OUTLET=3`):

- **Solid voxels** are no-slip walls: any velocity face touching a solid voxel
  is clamped to `0`.
- A face is an unknown only if **both** adjacent voxels are fluid.

---

## 3. Boundary conditions

Following the Darcy convention:

- **Regular volume** (no `boundary_volume`): `z = 0` is the inlet, `z = max`
  the outlet; default driving is a pressure drop (`p_inlet = 1`, `p_outlet = 0`).
- **Irregular boundary**: `boundary_volume` marks `INLET`/`OUTLET`/`PORE`; only
  `PORE` faces/cells are unknowns.
- **Walls**: no-slip (`u = 0`) on velocity, Neumann (`∂p/∂n = 0`) on pressure.
- **Inlet/Outlet**: Dirichlet on pressure, Neumann on velocity.

---

## 4. Solution method — pseudo-transient projection

Because Stokes flow is linear, we obtain the steady solution with a simple
**pseudo-transient projection (fractional-step) iteration**. Each iteration
advances a pseudo-time `Δt` and maps one-to-one onto the existing method stubs.
Since there is no advection, the "predictor" is pure diffusion:

**Stage 1 — Predictor (`predictor_step`)** — diffusion + body force only:

```
u* = uⁿ + Δt · ( ν∇²uⁿ + f )
```

**Stage 2 — Pressure Poisson (`poisson_step`)** — enforce incompressibility:

```
∇²p = (ρ/Δt) ∇·u*
```

This is a **Laplacian identical in structure to the Darcy system**, so we
assemble it once with `VolumeManager` and solve it every iteration with the
existing `DarcySolver` (`set_linear_system` → `generate_preconditioner` →
`solve_pcg`), warm-started from the previous pressure.

**Stage 3 — Corrector (`corrector_step`)** — project to divergence-free:

```
uⁿ⁺¹ = u* - (Δt/ρ) ∇p
```

Iterating stages 1–3 drives the field to the steady Stokes solution. (This is
a pseudo-transient / Uzawa-like iteration for the linear Stokes system; the
pseudo-time is purely a relaxation parameter, not physical time.)

---

## 5. Pseudo-time step

Because there is no advection, the only stability constraint is **viscous**:

```
Δt = time_step_factor · min(dx_i²) / (2 · ndim · ν)
```

This `Δt` is constant throughout the iteration — no adaptive/retry logic — which
keeps the driver trivial. `time_step_factor < 1` is a safety factor.

---

## 6. Convergence to steady state

Iterate until the velocity change drops below a threshold:

```
max|uⁿ⁺¹ − uⁿ| / max|uⁿ⁺¹|  <  target_error
```

or until `max_iterations` iterations are taken.

---

## 7. Reuse of `DarcySolver`

The only linear solve per iteration is the pressure-Poisson equation, whose
matrix is a Laplacian in exactly the CSR convention `DarcySolver` consumes. So:

1. Build the Poisson matrix **once** (geometry is fixed) via `VolumeManager`.
2. Each iteration rebuild only the RHS `b = (ρ/Δt) ∇·u*`.
3. Call `DarcySolver.solve_pcg` warm-started with the previous pressure.

---

## 8. Method / kernel map

| Concept                     | Function                                   |
|-----------------------------|--------------------------------------------|
| Allocate MAC fields & masks | `StokesSolver.create_velocity_arrays`      |
| Diffusion predictor `u*`    | `StokesSolver.predictor_step`              |
| Pressure Poisson solve      | `StokesSolver.poisson_step`                |
| Velocity projection         | `StokesSolver.corrector_step`              |
| Iteration driver            | `StokesSolver.solve`                       |
| Steady-state residual       | `StokesSolver._velocity_residual`          |
| Diffusion kernel            | `_diffuse_jit`                             |
| Divergence kernel           | `_divergence_jit`                          |
| Pressure-gradient corrector | `_apply_pressure_gradient_jit`             |
| Max relative change kernel  | `_max_relative_change_jit`                 |

All numerical kernels are module-level `@njit`/`@njit(parallel=True)`
functions, following the same pattern as `darcySolver.py`.

---

## 9. Performance strategy (time & RAM)

Simplicity is a *modeling* goal, not a licence to be wasteful — this solver must
scale to large voxel volumes. Concrete rules the implementation follows:

**Allocate once, never in the loop.**
- All MAC fields (`u, v, w, p`), the two velocity buffers used for ping-pong,
  the divergence/RHS vector, and the masks are allocated a single time in
  `create_velocity_arrays`. The iteration reuses them in place.
- The pressure-Poisson **matrix and preconditioner are built once** in
  `_build_pressure_poisson_system` (geometry is fixed). Each iteration rebuilds
  only the RHS vector `b`, in place.

**In-place, buffer-passing kernels.**
- Kernels take preallocated `*_out` arrays and write into them (see
  `_diffuse_jit`, `_apply_pressure_gradient_jit`) — no per-call allocation,
  mirroring `darcySolver.py`'s `_recalc_residuals_jit` / `_add_product`.
- Predictor uses ping-pong buffers: swap references, don't copy.

**Parallelism.** `@njit(parallel=True)` with `prange` and the same
thread-partitioning reduction pattern as `_square_sum_vector`
(`thread_start/thread_end` per thread, partial sums combined at the end).

**Warm-started Poisson.** The Poisson solve dominates per-iteration cost. We
pass the previous pressure as `X0` to `solve_pcg`, so once the field settles
each solve converges in a handful of CG iterations. (The Poisson preconditioner
is also the natural future place to plug in algebraic multigrid for
size-independent iteration counts, but that is out of scope here.)

**Memory footprint & knob.** Baseline stores full dense arrays (float64 for the
fields fed to `DarcySolver`, 1-byte masks). Rough cost for an `N`-voxel volume:
~4 float64 volumes for `u,v,w,p` + ~4 byte-masks + the Poisson CSR matrix
(~7·N nonzeros). Two future RAM levers, noted but *not* implemented in the
baseline to keep it readable:
1. **float32 fields** — halves velocity/pressure RAM (the Poisson solve still
   needs float64 vectors; convert at the boundary).
2. **Condensed storage** — like the Darcy path, store only fluid faces/cells
   instead of full dense arrays. Big win for low-porosity volumes, at the cost
   of index bookkeeping. Deferred.

---

## 10. References

The three components we actually implement, each traceable to its primary
source:

| Component in this doc                                  | Source              |
|--------------------------------------------------------|---------------------|
| MAC staggered grid — pressure at centers, velocity on faces (§2) | Harlow & Welch 1965 |
| Projection / fractional-step method: predictor, pressure-Poisson, divergence-free corrector (§4) | Chorin 1968 |
| Preconditioned conjugate gradient for the Poisson solve (§7) | Saad 2003 |

- F. H. Harlow, J. E. Welch, "Numerical calculation of time-dependent viscous
  incompressible flow of fluid with free surface," *Phys. Fluids* **8**(12),
  2182–2189 (1965).
- A. J. Chorin, "Numerical solution of the Navier–Stokes equations,"
  *Math. Comp.* **22**(104), 745–762 (1968).
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003).
