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

## 10. Multigrid acceleration (planned — next major work)

The baseline solver is correct (verified against Hagen–Poiseuille and OpenFOAM
Stokes data) but slow. This section is the design brief for the next work
package: replacing the current relaxation-limited solve with a **multigrid**
approach so iteration counts become (nearly) independent of volume size. It is
written to be picked up in a fresh context.

### 10.1 Why — the two measured bottlenecks

Benchmarking (see `scripts/benchmark_stokes_tolerance.py`,
`scripts/benchmark_enhanced_arns_duct.py`, `scripts/benchmark_arns_vs_openfoam.py`)
pinned the cost to two places, both of which multigrid targets directly:

1. **The pressure-Poisson solve scales poorly.** `DarcySolver`'s
   diagonal-preconditioned CG needs `O(N^{1/3})`–ish iterations that in practice
   blow up on real rock: a full `250³` Bentheimer Poisson solve took ~2 600 CG
   iterations and ~330 s *per solve*. This dominates per-iteration cost at scale.
2. **The outer pseudo-transient loop is relaxation-limited.** Because the
   viscous predictor is explicit, the velocity error in the *smooth* (low-
   frequency) modes decays on the diffusion timescale, so the number of
   projection iterations grows like `O((L/dx)²)`. A good initial guess barely
   helps here: an *exact* velocity seed converges in 1 iteration, but a
   realistic guess (enhanced-Arns, cosine ≈ 0.95 to the true field) only gives
   ~1.3–2.7× because the slow modes still must relax. **This is precisely the
   error spectrum multigrid removes** — hence the expectation that the Arns
   first-guess will pay off much more once the solver itself is fast.

### 10.2 Goal

Near-`O(N)` work for both (a) each pressure-Poisson solve and (b) the steady
velocity solve, i.e. V-cycle counts that stay flat as the volume grows. Keep the
existing condensed-CSR convention and Numba/`@njit` style; keep the HP and
OpenFOAM comparisons as accuracy regressions.

### 10.3 Phase 1 — Multigrid pressure-Poisson solve (biggest single win) — **DELIVERED**

**Status:** implemented in `pyflowsolver/multigridSolver.py` as
`MultigridSolver(Solver)`, a drop-in sibling of `DarcySolver`
(`set_linear_system` / `generate_preconditioner` / `solve_pcg(X0)`). Two
backends: `backend="native"` (self-contained smoothed-aggregation AMG with
module-level `@njit` V-cycle kernels — the deliverable) and `backend="pyamg"`
(permanent benchmark reference). Selected from `StokesSolver` via
`poisson_backend="pcg" | "mgpcg"` (default `pcg`, unchanged). Benchmark harness:
`scripts/prototype_amg_poisson.py`. Notes for the implementation as built:

- The assembled Laplacian is symmetric **negative**-definite; the solver
  auto-flips to `(-A)x = (-b)` for the SPD CG/AMG requirement.
- Setup (aggregation, prolongator smoothing, Galerkin `A_c = R A P`) runs once in
  Python via `scipy.sparse`; only the V-cycle hot loops are `@njit`.
- **Measured (native MG-PCG vs diagonal-PCG), tol 1e-8:** iteration count is flat
  (10→23 across N=3.3k→177k) vs diagonal-PCG's 61→1006. On an 80³ blob
  (N=177k): ~1006 iters/7.3s → 23 iters/~0.4s (setup+solve), ~18×. Solutions
  agree to ~1e-7. HP-duct and mgpcg/pcg field equality covered by tests.

Original design brief follows (still the reference for intent). Replace/augment
the diagonal-PCG Poisson solve with multigrid. The Poisson matrix is **fixed by
geometry**, so the hierarchy is built once in `_build_pressure_poisson_system`
and reused (warm-started) every iteration.

- **Method:** algebraic multigrid (**AMG**, smoothed aggregation) on the
  condensed CSR system. AMG is preferred over geometric MG here because the
  unknowns are only the *fluid* cells (irregular, low-porosity, disconnected
  clusters filtered by `VolumeManager`); geometric coarsening of a masked voxel
  grid is fiddly, whereas AMG coarsens the graph of the matrix directly.
- **Smoother:** reuse `_jacobi_sweeps` (already in `darcySolver.py`); weighted
  Jacobi (ω≈0.6–0.8) or a red/black Gauss–Seidel port.
- **Use as a preconditioner** inside the existing CG (`MG-PCG`) for robustness,
  not as a standalone iteration.
- **Prototype then port:** validate the hierarchy and cycle against `pyamg`
  first (throwaway, off the critical path), then implement the hot loops
  (smoother sweep, residual, restriction `R`, prolongation `P = Rᵀ`, coarse
  matvec) as module-level `@njit` kernels; store `R`, `P`, and the Galerkin
  coarse operators `A_c = R A P` in the same `{"val","col_idx","row_ptr"}` CSR
  convention. Setup runs in Python (once); cycles run in Numba.
- **Target:** replace ~10³ CG iterations with ~5–10 V-cycles, size-independent.

### 10.4 Phase 2 — Accelerate the steady velocity solve

Phase 1 makes each iteration cheap but leaves the `O((L/dx)²)` outer count. Two
routes, in increasing order of ambition:

- **Option A (incremental, reuses the projection) — DELIVERED (with a caveat).**
  The viscous predictor can be made **implicit** (backward-Euler diffusion
  `(I − Δt·ν∇²)u* = uⁿ + Δt·f − (Δt/ρ)∇pⁿ`, solved per component with the native
  multigrid) via `StokesSolver(predictor="implicit")` (default `"explicit"`,
  regular volumes only). Implemented as **incremental** pressure-correction
  (predict with ∇pⁿ, solve the homogeneous-BC increment φ, `p += φ`) so the
  projection splitting error vanishes at steady state; run in the large-Δt limit
  (`implicit_dt_factor≈1e6`) the outer loop becomes a **size-independent ~25-step
  Uzawa iteration**. Measured on ducts: explicit 1345→1908 outer iters (r8L12→
  r10L16) collapse to **24–26**, ~14–28× wall-clock, correct to ~1e-4.

  **Caveat / finding:** the pseudo-transient projection reaches the true steady
  state on channel-like geometry but can **stall short of it on complex pore
  media** — a projection-consistency limit (the reused VolumeManager Laplacian's
  inlet/outlet ghost term is not exactly the MAC `div·grad`). The momentum
  residual `R = ν∇²u − ∇p/ρ + f` diagnoses this cleanly (≈1e-6 when truly
  converged vs ≈1 when stalled). So the driver runs implicit until it settles
  then **hands off to the always-correct explicit predictor to verify/finish**
  under the same step criterion: channels confirm in ~1 step (`fell_back=False`),
  complex media get corrected to the exact explicit solution (`fell_back=True`).
  Net: safe everywhere, big win on channels, degrades to explicit cost on complex
  rock. The clean fix for size-independence on complex media is a MAC-consistent
  projection Laplacian or **Option B**.
- **Option B (target, removes pseudo-time):** solve the coupled **steady Stokes
  saddle-point system** for `(u, p)` directly with a block-preconditioned Krylov
  method (MINRES/GMRES). The natural block preconditioner is the textbook Stokes
  one: a velocity-block diffusion-MG (Phase-1 machinery applied per component)
  plus a pressure Schur-complement approximation — **which is exactly the
  pressure-Poisson operator we already assemble.** This gives size-independent
  convergence with no relaxation loop. Heavier lift (assemble/act on the coupled
  operator; MAC-consistent `div`/`grad` blocks) but the endpoint.
- **Recommendation:** ship Option A, measure iters-vs-size, then decide whether
  Option B's extra complexity is warranted.

### 10.5 Phase 3 — First-guess + multigrid

Re-run the Arns warm-start study once the solver is fast. Use the **enhanced**
Arns velocity, magnitude-normalized (empirically `v_stokes ≈ v_arns/(4ν)`, since
the enhanced conductivity is the Hagen–Poiseuille conductance). Expect a larger
payoff than in the baseline, where slow-mode relaxation capped it.

### 10.6 Integration constraints

- **Keep the condensed CSR** (`row_ptr` has `N` entries, not `N+1`); all
  transfer/coarse operators live in that format so `DarcySolver` and
  `VolumeManager` interop is preserved.
- **Numba:** setup (aggregation, Galerkin products) in Python, run once; per-
  cycle kernels module-level `@njit`/`@njit(parallel=True)` with the
  `thread_start/thread_end` reduction pattern used elsewhere.
- **Build once:** hierarchy setup belongs in `_build_pressure_poisson_system`;
  memory overhead is ~2× the fine-grid nnz (geometric series) — acceptable.
- **Isotropy:** the current Poisson assembly assumes isotropic voxels (§ guard
  in `_build_pressure_poisson_system`); keep that assumption for v1.

### 10.7 Verification

- **Correctness:** the duct HP profile and the OpenFOAM correlation must be
  unchanged versus the baseline (same solution, fewer iterations). Reuse
  `scripts/run_stokes_duct.py` and `scripts/benchmark_arns_vs_openfoam.py`.
- **MG signature:** produce iterations-vs-`N` and wall-clock-vs-`N` curves on a
  size sweep (duct and Bentheimer subvolumes); the multigrid curve should be
  flat where the baseline grows. Cross-check the MG Poisson solution against the
  current PCG result to machine tolerance.

### 10.8 Open questions / risks

- **Disconnected pore clusters:** `VolumeManager` filters non-percolating
  regions, but AMG aggregation must still cope with weakly/near-singular
  components; verify the coarse operators stay well-posed.
- **Saddle-point smoother (Option B):** full Vanka is memory/compute heavy;
  block-preconditioned MINRES with the existing pressure-Poisson as the Schur
  approximation is the lighter, recommended first cut.
- **Numba AMG setup:** classical/aggregation setup is pointer-chasing and awkward
  in `nopython` mode — keep setup in Python; only cycle kernels need `@njit`.

---

## 11. References

The three components we actually implement, each traceable to its primary
source:

| Component in this doc                                  | Source              |
|--------------------------------------------------------|---------------------|
| MAC staggered grid — pressure at centers, velocity on faces (§2) | Harlow & Welch 1965 |
| Projection / fractional-step method: predictor, pressure-Poisson, divergence-free corrector (§4) | Chorin 1968 |
| Preconditioned conjugate gradient for the Poisson solve (§7) | Saad 2003 |
| Multigrid / algebraic multigrid, saddle-point preconditioning (§10) | Brandt 1977; Trottenberg et al. 2001; Elman et al. 2014 |

- F. H. Harlow, J. E. Welch, "Numerical calculation of time-dependent viscous
  incompressible flow of fluid with free surface," *Phys. Fluids* **8**(12),
  2182–2189 (1965).
- A. J. Chorin, "Numerical solution of the Navier–Stokes equations,"
  *Math. Comp.* **22**(104), 745–762 (1968).
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003).
- A. Brandt, "Multi-Level Adaptive Solutions to Boundary-Value Problems,"
  *Math. Comp.* **31**(138), 333–390 (1977).
- U. Trottenberg, C. Oosterlee, A. Schüller, *Multigrid*, Academic Press (2001).
- H. Elman, D. Silvester, A. Wathen, *Finite Elements and Fast Iterative
  Solvers*, 2nd ed., Oxford (2014). (Block preconditioning for Stokes.)
