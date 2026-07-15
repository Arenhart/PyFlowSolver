# Pressure-estimator precision improvements

Brainstorm of ways to improve the precision of
`pyflowsolver/pressureEstimator.py`. Speed is not a constraint — the estimator
is already much cheaper than the time it saves in the full PCG. Segmentation
changes are **intentionally excluded** (2×2×2 cubic segments are already fast
enough for this application; watershed is too slow).

## Current pipeline (recap)

1. **Segment** – `segment_pore_space`: regular cubic chunks (`subsegment_size`)
   labeled per-chunk with `scipy.ndimage.label`. *Out of scope here.*
2. **Network** – `create_pseudo_network`: one node per segment + virtual
   inlet/outlet; throat conductance
   `= geom_mean(c) · total_vol / (2 · dist²)`, Euclidean centroid distance.
3. **Back-project** – `create_pressure_volume`: segment-constant pressure,
   linear z-interpolation between 1.0 / 0.0 and segment centroid for
   inlet/outlet-touching segments, then masked isotropic Gaussian blur.

## Relevant facts

- `NetworkManager` accepts arbitrary `(conn, cond)` but does **not** support
  multiple throats per pair; parallel conductances must be pre-summed before
  passing them in.
- `get_pressure_list` returns pressures for all nodes including virtual inlet
  (=1) and outlet (=0); length is `n_pores`.
- Voxel "conductivity" from `fast_laplacian_volume_generator` is
  `(EDT − margin)²` → high `c` = pore interior, low `c` = near solid wall.
  So a segment-wide geometric-mean `c` is dominated by interior voxels and
  **under-weights throats** (narrow constrictions) — the very elements that
  set the true resistance.
- Existing tests (`tests/distributed/test_pressure_estimator.py`) check only
  invariants (bounds, shape, `solid=0`, `inlet=1`, `outlet=0`), not accuracy.
  Any improvement that keeps pressures in `[0, 1]` with solids at 0 won't
  break them.

## Weaknesses

### Network (stage 2)

- **N1.** Shared-face area is discarded. `throat_pairs` is a `set`, so two
  segments touching at 1 voxel-face are treated like two touching at 50
  voxel-faces.
- **N2.** `harmonic_mean` is computed then never used. Series-resistor
  physics wants harmonic combination, and narrow-throat bottlenecks are best
  captured by the harmonic mean of *interface* voxels, not by a geometric
  mean over whole segment bodies.
- **N3.** The `total_vol / dist²` area proxy is a heuristic; true
  cross-sectional area is knowable once shared face voxels are counted.
- **N4.** Conductance uses centroid-to-centroid straight-line distance, which
  understates path length for tortuous segments.

### Back-projection (stage 3)

- **B1.** Every voxel in a segment gets the *same* pressure; the intra-segment
  gradient — often the main thing a good `X0` needs — is zero by construction.
- **B2.** Inter-segment transitions rely entirely on the Gaussian blur, which
  is isotropic and conductivity-agnostic.
- **B3.** Inlet/outlet interpolation uses only segment centroid `z`; x,y
  variation within boundary-touching segments is ignored.
- **B4.** Gaussian `sigma` is global, so tight throats are over-smoothed and
  their true gradient is flattened.

## Candidate improvements

### Network conductance

- **N-fix1. Record shared-face area.** Change the `throat_pairs` `set` into a
  `dict` keyed on `(idx1, idx2)` that counts shared-face voxels per axis.
  Trivial edit; fixes N1 and is a prerequisite for N-fix2.
- **N-fix2. Series-resistor conductance.** Replace
  `geom_mean · total_vol / (2 · dist²)` with
  `G = A_throat · 2 / (L1/c1_interface + L2/c2_interface)`,
  where `A_throat` = shared-face count × face area, `L1`, `L2` = centroid-to-
  interface-centroid distances, and `c_i_interface` is the **harmonic mean of
  conductivities of the face voxels on side *i***. Fixes N2, N3.
- **N-fix3. Conductivity-weighted centroids.** Weight centroid accumulation
  by `c` so the "effective pore location" tracks where flow actually happens
  — low-`c` wall voxels drag the centroid less. Partial fix to N4 without
  needing full geodesic distances.
- **N-fix4. Geodesic distance.** Replace Euclidean centroid distance with
  graph distance over pore voxels. Higher effort, uncertain win given N-fix2
  already dominates. Low priority.

### Back-projection

- **P1. Inverse-distance-weighted (IDW) blending.** For every pore voxel,
  instead of `p = seg_p`, compute

      p = Σ_k w_k · seg_p_k / Σ_k w_k

  where `k` ranges over this voxel's own segment and its face-adjacent
  segments, and `w_k = 1 / (|voxel − centroid_k|² + ε)`. Fixes B1 at ~no
  extra memory cost and can replace (or precede) the Gaussian blur.
- **P2. Per-segment local Laplacian.** For each segment, solve a small
  Dirichlet Poisson problem with BCs = interface pressures from the network
  solution (evaluated at throat mid-points). Embarrassingly parallel; highest
  fidelity but meaningful implementation cost.
- **P3. Conductivity-aware / guided blur.** Replace isotropic Gaussian with
  a diffusion weighted by local `c`, so smoothing respects channel geometry.
  Medium effort; `scipy.ndimage.generic_filter` or an anisotropic-diffusion
  step would work.
- **P4. Extend inlet/outlet interpolation to actual span.** Use z-min / z-max
  of *pore voxels in the segment* rather than the centroid; optionally
  interpolate per (x,y) column rather than per segment. Small, local fix
  to B3.

### Hybrid: cheap smoothing on the real system

- **H1. Jacobi / Gauss-Seidel sweeps on the full `A x = b`.** After
  back-projection, use the estimate as `X0` and run *k* (e.g. 5-20) Jacobi or
  SOR sweeps using the real sparse system. Classic multigrid intuition: the
  estimator supplies good low-frequency modes; sweeps kill the
  high-frequency residual that dominates early PCG iterations.
  `darcySolver.py` already has the `@njit` primitives
  (`_recalc_residuals_jit`, `_add_product`, preconditioner application); a
  new module-level `@njit _jacobi_sweeps` function is straightforward. Speed
  is not a concern, so *k* can be generous. Clamp the final output to
  `[0, 1]` and keep solid voxels zero so invariant tests still pass.

## Recommended order

Use the benchmark's new signed-diff and signed-grad TIFFs plus the
conductivity-correlation plots (from `scripts/benchmark_pressure_estimate.py`)
as the regression harness. Pick the first improvement based on what the
signed-diff data reveals:

- Strong conductivity-correlated pressure bias → **N-fix1 + N-fix2**
  (throat physics wrong).
- Blocky / checkerboard pattern in signed-diff TIFF → **P1**
  (segment-constant artefact).
- Near-inlet / near-outlet halos → **P4** (boundary interpolation).

Suggested sequence, each landing as an independent change:

1. **N-fix1 + N-fix2** together — shared-face area + series-resistor
   conductance with interface harmonic means. Small, mechanical change to
   `create_pseudo_network`; high physical-correctness payoff; almost certain
   to reduce bias. **First code change.**
2. **P1 (IDW blending)** in `create_pressure_volume` to replace segment-
   constant assignment. Addresses the biggest back-projection artefact
   without needing a local solve.
3. **H1 (Jacobi smoothing)** as a post-processing step bolted onto
   `estimate_pressure_distribution`. Likely the single biggest jump in `X0`
   quality given speed is unconstrained. Needs a new `@njit` in
   `darcySolver.py`; gate behind a parameter so pure-estimator mode stays
   available.
4. **N-fix3 (conductivity-weighted centroids)** — easy follow-up once N-fix2
   is in place.
5. **P2 (per-segment local Laplacian)** — only if 1-4 aren't enough. Ideal
   follow-up if precision is still limiting.
6. **P3 (guided blur)**, **P4 (boundary-span interpolation)**,
   **N-fix4 (geodesic distance)** — low-priority refinements, pick up only
   if the signed-diff data points to them.

## Critical files

- `pyflowsolver/pressureEstimator.py` — most edits land in
  `create_pseudo_network` (N-fix1, N-fix2, N-fix3, N-fix4) and
  `create_pressure_volume` (P1, P2, P3, P4).
- `pyflowsolver/darcySolver.py` — add a new module-level `@njit` Jacobi
  sweep (H1), reusing existing matvec / axpy primitives; expose through a
  thin helper on `DarcySolver`.
- `scripts/benchmark_pressure_estimate.py` — regression / sensitivity harness
  (already plots signed diffs and conductivity correlations).
- `tests/distributed/test_pressure_estimator.py` — invariant tests; any
  change must still satisfy `0 ≤ p ≤ 1`, `p(inlet) = 1`, `p(outlet) = 0`,
  `p(solid) = 0`.

## Verification

1. Run `scripts/benchmark_pressure_estimate.py`. Compare:
   - **Iterations** — `iters_est` vs `iters_zero`; improvements should drive
     `iters_est` down monotonically.
   - **Signed pressure diff** (`pressure_signed_diff_*.tiff`) — mean/median
     should move toward zero; the `zero > est` vs `est > zero` balance should
     become more symmetric.
   - **Signed grad-mag diff** (`gradient_signed_diff_*.tiff`) — same.
   - **Pearson `r(conductivity, signed diff)`** — magnitude should drop;
     improvements should *decorrelate* bias from local conductivity.
2. Run `pytest tests/distributed/test_pressure_estimator.py` to confirm
   invariants still hold.
3. Sweep each improvement in isolation (don't stack without measuring) so
   gains are attributable.
