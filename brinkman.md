# Darcy–Stokes–Brinkman model

Design document for the multiscale (subresolution-porosity) flow capability in
PyFlowSolver. **Status: single-phase scoped for implementation; two-phase
(Pc/kr/saturation) documented as a future step. All code shipped so far is
skeleton — see the `STATUS` notes and `TODO`s in each module.**

## Motivation

The existing models are binary: `VolumeManager`/`fastLaplacian` (Darcy) and
`StokesSolver` both treat every non-empty voxel as fully open pore. Real digital
rocks (carbonates, tight sandstones) contain **subresolution porosity** —
voxels imaged below the pore scale that are neither solid nor fully open. The
Darcy–Stokes–Brinkman model represents these directly:

- **open pores** → Stokes flow;
- **subresolution voxels** → Brinkman flow through an effective local
  permeability `K`;
- **near-solid / very low K** → the Darcy limit.

## New / changed components

| File | Role | Status |
|------|------|--------|
| `pyflowsolver/tubeBundle.py` | tube-radius distribution interface + 3 distributions + bundle-of-tubes physics + pluggable K-models | distributions + single-phase k + homogeneous K-model implemented; Pc/kr stubs |
| `pyflowsolver/multiscaleVolumeManager.py` | `MultiscaleVolumeManager(VolumeManager)` — graded porosity → conductivity + K field | conductivity path + `get_permeability_field` (homogeneous) implemented + tested |
| `pyflowsolver/stokesSolver.py` | `permeability=` Brinkman drag term | implemented + tested (Stokes byte-identical when `None`) |
| `pyflowsolver/brinkmanSolver.py` | `BrinkmanSolver(StokesSolver)` | functional + tested; multiscale first-guess still TODO |

Reused unchanged: `fastLaplacian.fast_laplacian_volume_generator` (already
supports the `0 / 1–99 / 100` percent map + `subresolution_function` hook),
`VolumeManager` assembly/flux, `pressurePoisson.assemble_poisson`,
`MultigridSolver`.

---

## 1. Physics

The solver integrates the **ρ-normalized** momentum equation to steady state
(the form `StokesSolver` already uses):

```
Du/Dt = ν ∇²u − ∇p/ρ + f          (Stokes)
Du/Dt = ν ∇²u − (ν/K) u − ∇p/ρ + f (Brinkman)
```

The Brinkman drag `−(μ/K)u` divided by ρ becomes **`−(ν/K)·u`**, so it **reuses
the existing `viscosity` (ν)** — no separate dynamic-viscosity parameter. `K`
has units length²; `ν/K` is a rate (1/time), so `dt·(ν/K)` is dimensionless.

**Regime recovery:**
- `K → ∞` ⟹ drag → 0 ⟹ **exactly today's Stokes code** (this is why
  `permeability=None` must stay byte-identical, and why open voxels carry
  `K = ∞`).
- small `K` ⟹ drag-dominated Darcy limit (stiff; solve implicitly).

Incompressibility (`∇·u = 0`) is unaffected by the drag term, so the entire
projection/pressure-Poisson machinery (`pressurePoisson.assemble_poisson`,
`poisson_step`, `corrector_step`) is untouched.

---

## 2. MultiscaleVolumeManager

`MultiscaleVolumeManager(porosity_map, scale, labelmap=None, distributions=None,
boundary_volume=None, enhanced_model=True)`.

**Porosity input.** Accepts `[0..1]` or `[0..100]` (auto-detected by `max > 1`),
stored internally as `[0..1]` float. Converted to the `0 / 1–99 / 100` uint
percent convention only at the `fastLaplacian` boundary (`_to_percent_map`).

**Region labelmap** (`0 = solid, 1 = resolved/open, 2+ = subresolution regions`).
If not supplied, auto-generated: `0` where porosity is 0, `1` where porosity is
maximal, `2` for all intermediate porosities (one subresolution region). Each
`2+` region **must** have a `TubeRadiusDistribution` in `distributions`.

**Ordering fix (critical).** The base `VolumeManager.__init__` computes
`filter_connected_volume()` / `_calc_null_counts` / `self.nonzeros` from
`self.volume > 0` **before** any conversion. If the raw porosity map were passed
in, the condensed unknown set would be derived from `porosity > 0` and could then
**desync** from the final conductivity field — the Arns model can yield ~0
conductivity at some open-voxel edges, silently turning an "unknown" into a
"solid" and breaking `ravel_sparse_solution`'s index walk. So the subclass
**builds the conductivity field first** (`_build_conductivity_field`) and hands
the already-converted field to `super().__init__`. A guard (TODO) asserts no
non-solid voxel became 0 conductivity.

**Conductivity routing (unified).** `_build_conductivity_field` = **Arns(open)
+ subresolution K**, on disjoint voxel sets:
- open (region 1): `fast_laplacian_volume_generator` on a `0/100` map (no `1–99`,
  so the generator's own subresolution path is inert) → EDT/footprint Arns
  conductance;
- subresolution (2+): `_subresolution_permeability_field` — the **shared**
  per-region helper that applies the chosen `subresolution_model` to each
  region's *exact* porosity (see below).

This is the single point where the open-voxel Arns model and the subresolution
model meet. Because the subresolution part goes through the same helper as the K
field, **the fast-Laplacian conductance and the Brinkman K field always use the
same subresolution model + seed** — verified by test. This also removes the old
percent-LUT path's two limitations: multi-region is handled correctly (per-region
loop), and subresolution conductance uses exact porosity rather than the integer
`1–99` quantization.

**Inherited unchanged:** `get_sparse_system` / `get_sparse_system_jit`
(harmonic-mean faces, Dirichlet folding), `ravel_sparse_solution`,
`get_conductivity`, filtering, `_unravel` — all operate on `self.volume` as an
arbitrary float conductivity field.

**Bridge:** `get_permeability_field() → K` (length²) = `_subresolution_permeability_field`
with open voxels overwritten to `∞`. So `∞` for open voxels, the
bundle-of-tubes `k` for subresolution voxels, `0` for solid. This is what
`BrinkmanSolver` consumes; it is **separate** from `self.volume` (a Laplacian
*conductance* field, not a Darcy `K`).

---

## 3. Bundle-of-tubes (`tubeBundle.py`)

A subresolution voxel is a bundle of parallel capillary tubes whose radii follow
a per-region **`TubeRadiusDistribution`** (ABC):

- `min()`, `max()` — radius bounds;
- `sample(n=None)` — random draw(s);
- `cdf(r_low, r_high)` — probability mass of radii in `[r_low, r_high]`.

Concrete distributions: `ConstantTubeDistribution(radius)`,
`TruncatedGaussianTubeDistribution(mean, sigma, r_min, r_max)`,
`TruncatedLognormalTubeDistribution(mean, sigma, r_min, r_max)` (mean/sigma in
log-space), and `EmpiricalTubeRadiusDistribution(samples)`. `min`/`max`/`sample`/
`cdf` are implemented (inverse-CDF sampling, `scipy.special.ndtr`/`ndtri`).

**Production input** is a throat-radius distribution from mercury-injection
(MICP), fitted to a Gaussian or log-Gaussian. The parametric distributions take
data-driven constructors that compute the fitted statistics:
`from_samples(radii, r_min=, r_max=)` (raw radii) and
`from_density(radii, density, r_min=, r_max=)` (a radius↔density curve, weights
normalised) — the Gaussian fits linear-space moments, the log-Gaussian log-space.
Pass `r_max = F` (the downsample factor / resolution) to truncate: a pore of
radius ≳ F would be resolved, so it is excluded from the subresolution set.

**Single-phase permeability (implemented).** A bundle of parallel
Hagen–Poiseuille tubes filling fraction φ of the voxel, radii ~ number-pdf
`f(r)`, has effective permeability

```
K = φ · (1/8) · ⟨r⁴⟩/⟨r²⟩          (length²)
```

The `⟨r⁴⟩/⟨r²⟩` ratio is the area/flow-weighted mean of the per-tube permeability
`r²/8` (each tube weighted by its cross-section `πr²` and probability) — the `r⁴`
flow weighting pulls the effective radius toward the larger tubes. Derivation:
total flow `Q ∝ Σrᵢ⁴`, porosity `φ ∝ Σrᵢ²`; eliminating the tube count gives the
ratio. `darcy_permeability(distribution, porosity)` computes this, taking the
moments from the `cdf` (mass-normalisation-invariant since it's a ratio).
`subresolution_conductance_function(distribution)` wraps it into the
**vectorized** closure `fastLaplacian` needs — called once on the whole `0..99`
percent array, **maps 0 → 0** (0–99 lookup table indexed by the array).

**Subresolution permeability models — pluggable (`SUBRESOLUTION_MODELS`).**
The per-voxel subresolution `K` is computed by a pluggable strategy (default
`"homogeneous"`); the manager takes `subresolution_model=` (+ `subresolution_seed=`
for stochastic ones). **Both consumers use it via the shared
`_subresolution_permeability_field`**: the fast-Laplacian conductance field
(`self.volume`, adding Arns on open voxels) and the Brinkman K field
(`get_permeability_field`, open voxels → ∞). Registering a new model therefore
upgrades both at once. This deliberately keeps only the
tubes-≪-voxel (REV) limit for now. The homogeneous model fails near resolution
for two distinct reasons — (i) *statistical*: too few tubes per voxel, so the
ensemble mean isn't the voxel's actual `K`; (ii) *structural*: near-resolution
features span voxels, so independent parallel bundles miss series bottlenecks
and connected fast paths. Implemented models:
- **`"homogeneous"`** (default) — `K = φ·(1/8)⟨r⁴⟩/⟨r²⟩`, the REV limit.
- **`"sampled"`** — finite-bundle near-resolution model. Per voxel, draw
  `N = round(A·φ/(π⟨r²⟩))` tubes (`A = dx·dy`, the number that physically fit)
  and set `K = (π/8)·Σrᵢ⁴/A`. Since `E[Σrᵢ⁴] = N⟨r⁴⟩`, the expectation is exactly
  the homogeneous `K`, so it **converges** to homogeneous as N→∞ (tubes ≪ voxel)
  and injects real voxel-to-voxel **heterogeneity** when N is small; the global
  solve then resolves connectivity/bottlenecking across the heterogeneous field
  via the existing harmonic-mean faces. Reproducible (single seeded RNG in fixed
  voxel order; `subresolution_seed=`); when `N > max_tubes` (converged) it uses
  the exact deterministic homogeneous value. **This is a first alternative — the
  per-voxel conductivity calculator is expected to be revisited and compared
  against other models.**

Future strategies (register in `SUBRESOLUTION_MODELS`, no assembly changes):
- **critical-path / EMT** — deterministic bottleneck-controlled `K ∝ r_c²`
  (`r_c` from the `cdf` percolation threshold) or self-consistent effective
  medium; accounts for distribution broadness without stochasticity.
- **spatially-correlated realization** — like sampling but with a correlation
  length ≈ feature size so large pores connect across voxels (addresses
  connectivity properly).

**Two-phase (STUBS — future step).**
- `capillary_pressure(distribution, saturation, σ, θ)`: order tubes by radius;
  at wetting saturation `S`, the non-wetting phase fills the largest tubes down
  to radius `r*` where the cumulative (volume-weighted) tube fraction equals
  `1 − S` (invert `cdf`); then Young–Laplace `Pc = 2σcosθ / r*`.
- `relative_permeability(distribution, saturation) → (krw, krnw)`: each phase's
  sub-bundle Hagen–Poiseuille conductance (radii `≤ r*` vs `> r*`) over the
  full-bundle conductance — the `darcy_permeability` moment integral restricted
  to each sub-range.

These give `Pc(S)` and `kr(S)` per subresolution region for an eventual
two-phase solver (out of scope this session).

---

## 4. Solver integration (`stokesSolver.py` + `brinkmanSolver.py`)

`StokesSolver.__init__` gains `permeability=None`. When set, `_build_drag_fields`
samples the face drag `ν/K` (mean of the two adjacent cells' `ν/K` = the
harmonic-K face rule; open voxels ⟹ 0) onto the three MAC grids and sets
`_has_drag=1`. The drag is threaded through the kernels via a `has_drag` flag +
the drag arrays; when `permeability is None` a shared `(1,1,1)` placeholder is
passed with `has_drag=0`, so **plain Stokes allocates no drag arrays and is
byte-identical** (verified: the physics-level Stokes tests pass unchanged).
Implemented insertion points:

| Location | Change |
|----------|--------|
| `_diffuse_component_jit` / `_diffuse_jit` (explicit) | `out = c + dt*(ν·lap + f − drag·c)` |
| `_assemble_diffusion_csr` / `_build_diffusion_systems` (implicit) | per-row diagonal `+= dt·drag` (stays SPD) |
| `_compute_timestep` | explicit reaction cap `dt ≤ tsf·2/max(drag) = tsf·2·min(K)/ν` |
| `_momentum_residual` | drag flows in via `_diffuse_jit` ⟹ `R = ν·lap(u) − drag·u − ∇p/ρ + f` |
| `_seed_velocity_from_pressure` | auto-inherits drag (reuses `_build_diffusion_systems`) |

Because the implicit diffusion operator gains only a **positive diagonal**
reaction term, it stays SPD — multigrid and the seed path are unaffected. The
Darcy limit is stiff, so `BrinkmanSolver` defaults `predictor="implicit"`.

`BrinkmanSolver(volume|porosity_map, permeability|…)`:
- **precomputed-K path** — model-independent, pass `permeability` + `volume`;
- **porosity-map path** — lazily builds `MultiscaleVolumeManager`, derives `K`
  via `get_permeability_field()` and the pore mask, feeds both to the base.
- **output** — `effective_permeability()` returns the sample-scale Darcy `K_eff`
  = `(Q/A)·μ·L/Δp` from the converged MAC through-flux (Darcy limit ⟹ input `K`;
  open limit ⟹ duct geometric permeability).
- **first guess (TODO)** — the multiscale-Darcy warm start via
  `MultiscaleVolumeManager` is not wired yet, so `fast_laplacian_guess` defaults
  to `False` (cold start); pass `initial_pressure` to warm-start.

---

## 5. Testing / validation plan

**This session (skeletons):**
- All new modules import cleanly.
- `pytest tests/unit/test_stokes_solver.py` unchanged (drag no-op when `None`).
- `MultiscaleVolumeManager` on a small graded volume: `nonzeros` consistent
  between construction and `ravel_sparse_solution` (no unknown-set desync).
- Distributions round-trip `min/max/sample/cdf`;
  `subresolution_conductance_function` maps a `0..100` array with `0 → 0`.
  `darcy_permeability` cross-checked: cdf moment integral `(1/8)⟨r⁴⟩/⟨r²⟩`
  matches an independent Monte-Carlo estimate from the sampling path.
- **`MultiscaleVolumeManager` on the downscaled Bentheimer image** (250³ →
  block-mean ÷10 → 25³ porosity map, no thresholding) builds a conductivity map
  for all three distribution types; verifies solid→0, subresolution→positive,
  finiteness, labelmap auto-gen to spec, `nonzeros` consistency (ordering fix),
  and `[0..1]` vs `[0..100]` normalization equivalence.
- **`get_permeability_field`**: open→∞, solid→0, subresolution finite/positive;
  homogeneous `K` linear in porosity; multi-region uses each region's own
  distribution; unknown `subresolution_model` raises.
- **`"sampled"` model**: `sample(rng=)` reproducible; the field is reproducible
  given `subresolution_seed`, changes with seed, is heterogeneous over uniform
  porosity (vs constant homogeneous), its spatial mean matches the homogeneous
  `K`, and it collapses to the exact homogeneous value for tubes ≪ voxel.
- **Shared model**: on a uniform-porosity block, `self.volume` (fast-Laplacian
  conductance) equals `get_permeability_field` (Brinkman K) on subresolution
  voxels, for both `"homogeneous"` and `"sampled"` — confirming both consumers
  use the same model. See `tests/unit/test_multiscale_volume_manager.py`.
- **`BrinkmanSolver`** (`tests/unit/test_brinkman_solver.py`): `K → ∞` reproduces
  `StokesSolver` velocity/pressure to ~machine precision (explicit *and*
  implicit); finite drag reduces flow below the Stokes limit; `effective_permeability`
  is monotone in `K` and hits the **Darcy limit** `K_eff ≈ K` for small `K`
  (`√K ≪` duct width); the porosity-map construction path solves end to end.
- NOTE: `fastLaplacian._calculate_footprint` (enhanced-Arns) previously had a
  `prange` write race making conductivity non-reproducible; it was rewritten as
  a race-free **gather** and is now deterministic (guarded by
  `tests/unit/test_fast_laplacian.py`). pyedt's `edt` was never at fault.

- **Analytical Brinkman duct** (`tests/unit/test_brinkman_analytical.py`): the
  rectangular-duct Brinkman `K_eff` series is the reference. The solver's
  **ratio** `K_eff(K)/K_eff(∞)` matches it (cancelling the O(1/w) area/length
  conventions; the effective domain is `(w+1)dx×(h+1)dy` from the ghost=0 wall);
  covers direct-K and the bundle-of-tubes chain (constant + gaussian), converges
  under refinement, and explicit==implicit. **Multi-region**: series→harmonic and
  parallel→arithmetic in the Darcy limit (self-consistent against the solver's own
  single-region `K_eff`); subres+open mixes are bracketed sanity checks (no clean
  closed form — an open region's `K_eff∝width²` and interface shear break the
  simple rules).
- **Benchmark scripts** (exploration, not CI — high-res Stokes is expensive):
  - `scripts/benchmark_brinkman_multiscale.py` — high-res Stokes vs low-res
    Brinkman on a Bentheimer crop; caches the high-res Stokes and compares two
    subres-radius estimators. **Finding (the key modeling issue):** the
    **footprint** (local-thickness) radius **bleeds adjacent open-pore size into
    subres voxels** (95% of subres footprint radii exceed the voxel half-width),
    and `K∝r⁴` turns that into a ~14× subres-K overestimate → Brinkman flows
    ~1.5–3× too fast (footprint mean speed ×2.96 at full 250³→50³). **The fix**:
    use a *physical* radius (pore radius, not bled local-thickness) AND **truncate
    the radius distribution at the resolution F** (a pore of radius ≳ F would be
    resolved, not subresolution). With that, on the full 250³: EDT gives mean
    speed **×1.08** and a fixed `Gaussian(4.25, 2.86)` truncated to `[1, F]` gives
    **×1.14** — bulk permeability now correct within ~10–15%. The residual
    block-averaged *field* error (~0.7 rel-L2) is the homogenization limit (a
    Brinkman continuum can't reproduce the sharp channelized Stokes velocity), not
    a permeability error. Permeability is ultimately *throat*-controlled, so the
    remaining refinement is a throat-radius distribution (medial-axis minima /
    pore-network throats); bundle-of-tubes also ignores tortuosity by construction.
    `scripts/precompute_stokes.py` caches the high-res Stokes so estimator
    experiments re-run in seconds.
  - `scripts/benchmark_brinkman_tubes.py` — synthetic parallel-tube bundle
    (geometry = the model's assumption). Findings: low-res Brinkman reproduces
    resolved-Stokes `K_eff` within ~15% (`k_lr/k_hr ≈ 0.82–0.86`, footprint radii
    best); the *analytical* bundle underpredicts resolved Stokes ~1.5–1.7× purely
    from tube **voxelisation** (`K∝r⁴`, a reference artifact); clean field
    agreement needs many well-resolved tubes per low-res voxel (HPC scale). Uses a
    measured-gradient `K_eff` to cancel the length convention.
  - `scripts/benchmark_brinkman_bounds.py` — mixing-rule tightness: Darcy-limit
    series=harmonic / parallel=arithmetic of single-region `K_eff` are exact
    (<0.6%); subres+open has no tight closed form (series harmonic off up to ~20%
    for strong drag; parallel arithmetic off ~2× — half-width open channel
    `K∝width²` + interface shear). Motivates the bracketed-only mixed tests.

**Still deferred:**
- **Analytic Brinkman *profile*** (not just integral `K_eff`): finite-`K` slab vs
  the closed-form `cosh` profile.
- **Two-phase**: `Pc(S)`/`kr(S)` curves vs the analytic bundle-of-tubes result.

## Deferred to implementation sessions

- `BrinkmanSolver` multiscale-Darcy first guess (currently cold-start only).
- Further subresolution K-models (critical-path / EMT / correlated) in
  `SUBRESOLUTION_MODELS` (homogeneous + sampled are done; both the conductance and
  K fields pick them up automatically), and comparison/testing of the
  alternatives against reference permeabilities.
- Revisit the per-voxel conductivity calculator itself (the bundle formula and
  the `N`/area convention) — expected to be refined.
- The full two-phase `Pc(S)`/`kr(S)` model and any two-phase solver loop.
