"""Steady incompressible Stokes (creeping) flow solver.

See `stokes_solver.md` for the full algorithm. Stokes flow is linear (no
advection, no inertia), so we obtain the steady field with a pseudo-transient
projection iteration.

This solver is self-contained and independent of the Darcy / fast-Laplacian
model: it takes a voxel volume ndarray, assembles its OWN pressure-Poisson
operator (`pressurePoisson.py`), and solves each projection's pressure step with
the algebraic-multigrid `MultigridSolver`. It never imports `VolumeManager`,
`fastLaplacian`, or `DarcySolver`, and never solves Darcy's law. The only bridge
between the two models is that a caller may pass a fast-Laplacian result as an
initial guess (`initial_velocity` / `initial_pressure`) to warm-start Stokes.

Regular (z-driven) geometry only: z=0 inlet (p=1), z=max outlet (p=0), no-slip
walls, isotropic voxels. Module-level `@njit` kernels handle the hot loops.
"""

import numpy as np
from numba import njit, prange

from pyflowsolver.solver import Solver
from pyflowsolver.multigridSolver import MultigridSolver
from pyflowsolver.pressurePoisson import assemble_poisson, filter_percolating, ravel


class StokesSolver(Solver):
    # Stopping criteria for the projection iteration (see `_velocity_residual`).
    CONVERGENCE_CRITERIA = ("step", "residual")
    # Viscous predictor discretization:
    #   "explicit" -> forward-Euler diffusion, dt viscous-limited (baseline)
    #   "implicit" -> backward-Euler diffusion (I - dt*nu*L)u* = u^n + dt*f,
    #                 solved per component with multigrid. Unconditionally
    #                 stable, so dt is not viscous-capped -> far fewer, larger
    #                 pseudo-time steps to steady state (Phase 2, stokes_solver.md
    #                 s10.4 Option A). The implicit pseudo-transient is size-
    #                 independent (~25 steps) on channel-like geometry but can
    #                 stall short of steady state on complex pore media, so once
    #                 it settles the driver hands off to the explicit predictor
    #                 to verify/finish (see `solve`); `fell_back` reports whether
    #                 that finish had to do real work. Regular (z-driven) volumes
    #                 only -- irregular boundaries raise NotImplementedError.
    PREDICTORS = ("explicit", "implicit")
    # Sentinel for `initial_velocity`: derive the seed velocity from
    # `initial_pressure` via a steady viscous solve (see `_seed_velocity_from_pressure`).
    SEED_FROM_PRESSURE = "from_pressure"
    # Relative-residual tolerance for that seed's per-component diffusion solves.
    # Loose on purpose -- it is only a seed, so an approximate viscous velocity is
    # plenty and keeps the seed cost low.
    SEED_TOLERANCE = 1.0e-4

    DEFAULT_PARAMS = {
        "viscosity": 1.0,          # kinematic viscosity nu = mu / rho
        "density": 1.0,            # rho
        "time_step_factor": 0.5,   # safety factor on the viscous pseudo-dt
        "max_iterations": 5000,    # max projection iterations to steady state
        "target_error": 1e-07,     # steady-state velocity-change tolerance
        "body_force": (0.0, 0.0, 0.0),  # body force per unit mass f = (fx, fy, fz)
        # How to measure convergence:
        #   "step"     -> max|u^{n+1}-u^n| / max|u^{n+1}|   (relative change)
        #   "residual" -> that divided by dt, i.e. the steady-state momentum
        #                 residual max|R| / max|u|, which is independent of the
        #                 pseudo-time step (a property of the field, not of dt).
        "convergence_criterion": "step",
        # Viscous predictor discretization (see PREDICTORS).
        "predictor": "explicit",
        # Pseudo-dt for the implicit predictor: dt = implicit_dt_factor *
        # min(dx_i^2) / nu. Unbounded by stability. Convergence is fastest in the
        # large-dt (Uzawa) limit, where the predictor reduces to a steady
        # diffusion solve given p and the outer loop becomes a size-independent
        # ~20-iteration Schur iteration; the count saturates once dt exceeds the
        # domain diffusion time (~ (L/dx)^2), so the default is deliberately large
        # (overshooting is free -- incremental correction keeps it accurate).
        "implicit_dt_factor": 1.0e6,
        # Stagnation ("plateau") stop -- a robust safety net for when the field is
        # steady but the step-residual has flattened at a noise floor ABOVE
        # target_error (common on complex pore media, where the residual bottoms
        # out ~1e-7 and a 1e-8 target would otherwise grind to max_iterations).
        # The solve also stops when the residual improves by less than
        # `stagnation_tol` (fractional) over the last `stagnation_window`
        # iterations. Set stagnation_window=0 to disable (pure target_error).
        "stagnation_window": 25,
        "stagnation_tol": 1.0e-2,
    }

    def __init__(self, volume, scale=1.0, boundary_volume=None,
                 initial_pressure=None, initial_velocity=None,
                 backend="native", fast_laplacian_guess=True, **params):
        """
        volume: a 3D voxel array `(w, h, d)`; any positive value is pore, 0 is
            solid. Flow is driven along z (z=0 inlet, z=max outlet).
        scale: voxel size, a scalar or (dx, dy, dz). Must be isotropic.
        boundary_volume: reserved for irregular INLET/OUTLET geometries; not yet
            supported here -- pass None (raises NotImplementedError otherwise).
        initial_pressure: optional first guess for the cell-center pressure,
            an ndarray shaped like the volume `(w, h, d)`. Defaults to zeros.
        initial_velocity: optional first guess for the MAC velocity field, a
            tuple/list `(u, v, w)` of the staggered face arrays. Defaults to
            zeros. This is the one place a fast-Laplacian result may be fed in.
            May also be the string "from_pressure": derive the seed velocity by
            solving the steady viscous momentum equation for `initial_pressure`
            (-nu*laplacian(u) = -grad(p)/rho, no-slip). Given a good pressure
            (e.g. a fast-Laplacian solve), this yields a no-slip-correct velocity
            that cuts the projection iteration count on complex media (~-30% on
            Bentheimer). Requires `initial_pressure`.
        fast_laplacian_guess: if True (default) and no explicit `initial_pressure`
            / `initial_velocity` is given, `solve` first computes an enhanced
            fast-Laplacian (Arns) pressure for `volume` and warm-starts from it
            (velocity derived as with "from_pressure"). Recommended default on
            complex media (~-30% iterations, no accuracy cost). It couples the
            solve to `VolumeManager`/`fastLaplacian` (pulls `pyedt`), imported
            lazily so a plain `import` stays free of that dependency. Set False for
            a pure cold start (no Darcy/EDT dependency, no warm-start overhead).
        params: any key in DEFAULT_PARAMS (see module docstring / md file).
        """
        if boundary_volume is not None:
            raise NotImplementedError(
                "StokesSolver supports regular (z-driven) volumes only; irregular "
                "boundary_volume geometries are not yet supported."
            )
        self.volume = np.asarray(volume)
        scale = np.asarray(scale, dtype=np.float64).ravel()
        self.scale = np.repeat(scale, 3) if scale.size == 1 else scale
        self.boundary_volume = None
        self._fluid_bool = None          # filtered percolating mask (set lazily)
        self.backend = backend
        self.fast_laplacian_guess = fast_laplacian_guess
        self.params = self.DEFAULT_PARAMS.copy()

        # Optional warm-start guesses, applied by create_velocity_arrays.
        self.initial_pressure = initial_pressure
        self.initial_velocity = initial_velocity

        # MAC staggered fields (allocated once by create_velocity_arrays)
        self.u = None            # x-face velocity, shape (w+1, h, d)
        self.v = None            # y-face velocity, shape (w, h+1, d)
        self.w = None            # z-face velocity, shape (w, h, d+1)
        self.p = None            # cell-center pressure, shape (w, h, d)

        # Preallocated scratch reused every iteration (no per-step allocation):
        # ping-pong velocity buffers for the predictor and the Poisson RHS.
        self.u_buf = None
        self.v_buf = None
        self.w_buf = None
        self.rhs = None          # pressure-Poisson RHS (condensed fluid cells)

        # Precomputed geometry / boundary masks
        self.fluid_mask = None   # cell-center fluid mask
        self.u_mask = None       # active x-faces
        self.v_mask = None       # active y-faces
        self.w_mask = None       # active z-faces

        # Cached pressure-Poisson solver (MultigridSolver: matrix + hierarchy)
        self.poisson_solver = None

        # Implicit-predictor state: one diffusion solver per velocity component
        # (built once, since dt and geometry are fixed), plus the condensed
        # index maps and reused RHS / warm-start buffers.
        self.diffusion_solvers = None      # [u, v, w] MultigridSolvers
        self.diffusion_index_maps = None   # [u, v, w] condensed face->row maps
        self.diffusion_rhs = None          # [u, v, w] RHS vectors
        self.diffusion_x0 = None           # [u, v, w] warm-start vectors

        for key, value in params.items():
            if key in self.DEFAULT_PARAMS.keys():
                self.params[key] = value
            else:
                raise Exception(f"{key} is not a valid parameter,"
                                f"parameters are {list(self.params.keys())}")

        if self.params["convergence_criterion"] not in self.CONVERGENCE_CRITERIA:
            raise Exception(
                f"convergence_criterion must be one of {self.CONVERGENCE_CRITERIA}, "
                f"got {self.params['convergence_criterion']!r}"
            )

        if self.params["predictor"] not in self.PREDICTORS:
            raise Exception(
                f"predictor must be one of {self.PREDICTORS}, "
                f"got {self.params['predictor']!r}"
            )

        if self.params["stagnation_window"] < 0:
            raise Exception("stagnation_window must be >= 0 (0 disables the plateau stop)")
        if self.params["stagnation_tol"] < 0.0:
            raise Exception("stagnation_tol must be >= 0.0")

        if isinstance(self.initial_velocity, str):
            if self.initial_velocity != self.SEED_FROM_PRESSURE:
                raise ValueError(
                    f"initial_velocity string must be {self.SEED_FROM_PRESSURE!r}, "
                    f"got {self.initial_velocity!r}"
                )
            if self.initial_pressure is None:
                raise ValueError(
                    f"initial_velocity={self.SEED_FROM_PRESSURE!r} requires initial_pressure"
                )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def create_velocity_arrays(self):
        """Allocate MAC velocity/pressure fields, scratch buffers, and masks.

        Sizes derive from `self.volume.shape == (w, h, d)`:
            u: (w+1, h, d)   v: (w, h+1, d)   w: (w, h, d+1)   p: (w, h, d)

        Everything the iteration touches is allocated here exactly once (fields,
        the `*_buf` ping-pong buffers, and `rhs`) so the solve loop performs no
        allocations. Masks are stored as 1-byte arrays to keep RAM down:
        `fluid_mask` (cell-center) plus the per-component face masks marking
        active unknowns (both neighbouring voxels fluid) vs no-slip walls
        (touching a solid voxel).
        """
        w, h, d = self.volume.shape

        # MAC staggered fields on cell faces (velocity) and centers (pressure).
        self.u = np.zeros((w + 1, h, d), dtype=np.float64)
        self.v = np.zeros((w, h + 1, d), dtype=np.float64)
        self.w = np.zeros((w, h, d + 1), dtype=np.float64)
        self.p = np.zeros((w, h, d), dtype=np.float64)

        # Ping-pong buffers reused by the predictor every iteration.
        self.u_buf = np.zeros_like(self.u)
        self.v_buf = np.zeros_like(self.v)
        self.w_buf = np.zeros_like(self.w)

        # Cell-center fluid mask (1 = fluid, 0 = solid).
        fluid = self._compute_fluid_mask()
        self.fluid_mask = fluid

        # A velocity face is an unknown only when BOTH adjacent cells are fluid.
        # Domain-boundary faces touch a single cell, so they stay 0 (no-slip
        # walls) and are handled by the boundary-condition step.
        self.u_mask = np.zeros((w + 1, h, d), dtype=np.uint8)
        self.v_mask = np.zeros((w, h + 1, d), dtype=np.uint8)
        self.w_mask = np.zeros((w, h, d + 1), dtype=np.uint8)
        self.u_mask[1:w, :, :] = np.logical_and(fluid[:-1, :, :], fluid[1:, :, :])
        self.v_mask[:, 1:h, :] = np.logical_and(fluid[:, :-1, :], fluid[:, 1:, :])
        self.w_mask[:, :, 1:d] = np.logical_and(fluid[:, :, :-1], fluid[:, :, 1:])

        # Apply optional warm-start guesses on top of the zero fields.
        if self.initial_pressure is not None:
            self._set_initial_field(self.p, self.initial_pressure, "initial_pressure")
        if isinstance(self.initial_velocity, str):
            # Sentinel (validated in __init__): derive the seed velocity from the
            # just-applied pressure guess via a steady viscous solve. Masks and
            # buffers above are all it needs.
            self._seed_velocity_from_pressure()
        elif self.initial_velocity is not None:
            if len(self.initial_velocity) != 3:
                raise ValueError(
                    "initial_velocity must be a (u, v, w) tuple of 3 arrays, "
                    f"got {len(self.initial_velocity)} entries"
                )
            u0, v0, w0 = self.initial_velocity
            self._set_initial_field(self.u, u0, "initial_velocity[u]")
            self._set_initial_field(self.v, v0, "initial_velocity[v]")
            self._set_initial_field(self.w, w0, "initial_velocity[w]")
            # No-slip invariant: wall faces stay at 0 regardless of the guess.
            self.u[self.u_mask == 0] = 0.0
            self.v[self.v_mask == 0] = 0.0
            self.w[self.w_mask == 0] = 0.0
            # The inlet/outlet faces are masked (mask == 0) but are NOT no-slip
            # walls -- they carry the through-flux. Zeroing them above would
            # clamp the duct shut and force a converged seed to re-develop its
            # end flux over many iterations, so restore them from the interior.
            self._apply_velocity_boundary_conditions()

    @staticmethod
    def _set_initial_field(target, source, name):
        """Copy `source` into the preallocated `target`, validating its shape."""
        source = np.asarray(source, dtype=target.dtype)
        if source.shape != target.shape:
            raise ValueError(
                f"{name} has shape {source.shape}, expected {target.shape}"
            )
        target[...] = source

    def _compute_fast_laplacian_guess(self):
        """Compute an enhanced fast-Laplacian (Arns) pressure for `self.volume` and
        install it as the warm start (velocity derived from it in
        `create_velocity_arrays` via the "from_pressure" path).

        This is the only place StokesSolver touches the Darcy/Arns machinery, so
        `VolumeManager` (which pulls `fastLaplacian`/`pyedt`) is imported lazily --
        a plain `import stokesSolver`, or a solve with `fast_laplacian_guess=False`,
        stays free of that dependency.
        """
        from pyflowsolver.volumeManager import VolumeManager  # lazy: pulls pyedt
        poremap = ((np.asarray(self.volume) > 0).astype(np.float32)) * 100.0
        vm = VolumeManager(poremap, scale=self.scale)
        vm.convert_pore_volume_to_laplacian_conductivity(enhanced_model=True)
        a_sparse, b = vm.get_sparse_system_jit()
        darcy = MultigridSolver(backend=self.backend)
        darcy.set_linear_system(a_sparse, b)
        darcy.generate_preconditioner()
        x, _, _ = darcy.solve_pcg()
        self.initial_pressure = np.asarray(vm.ravel_sparse_solution(x), dtype=np.float64)
        self.initial_velocity = self.SEED_FROM_PRESSURE

    def _seed_velocity_from_pressure(self):
        """Set u, v, w to the steady viscous velocity implied by `self.p`.

        Solves, per component, the steady Stokes momentum balance for the fixed
        pressure guess with no-slip walls,

            -nu * laplacian(u) = -grad(p)/rho  (+ body force),   u = 0 at walls,

        which is what a good pressure (e.g. a fast-Laplacian solve) implies for
        the velocity. Unlike an algebraic k*grad(p) guess it enforces no-slip and
        the true viscous profile, removing the slow-mode error that limits the
        pseudo-transient -- so it reduces the projection iteration count on
        complex media (not just per-iteration cost).

        Implementation: the implicit predictor at large dt from u=0 IS exactly
        this steady solve ((I - dt*nu*L)u* = -(dt/rho)grad(p) -> -nu*L*u* =
        -grad(p)/rho as dt->inf), so we reuse that tested machinery. The
        per-component diffusion solves use a loose tolerance (`SEED_TOLERANCE`);
        the (memory-heavy) diffusion hierarchies are freed afterwards since the
        actual solve rebuilds its own (implicit) or does not need them (explicit).
        Requires masks/buffers allocated and `self.p` set (both done by the
        caller `create_velocity_arrays`).
        """
        dx, dy, dz = (float(s) for s in self.scale[:3])
        nu = self.params["viscosity"]
        min_dx2 = min(dx * dx, dy * dy, dz * dz)
        # Large (Uzawa-limit) dt so the implicit predictor reduces to the steady
        # viscous solve, independent of the predictor chosen for the real solve.
        dt = self.params["implicit_dt_factor"] * min_dx2 / nu

        self._build_diffusion_systems(dt)
        for diff_solver in self.diffusion_solvers:
            diff_solver.params["target_error"] = self.SEED_TOLERANCE
        u_star, v_star, w_star = self._implicit_predictor_step(dt)
        self.u[...] = u_star
        self.v[...] = v_star
        self.w[...] = w_star
        # Wall faces are already 0 (masked kernels); open the inlet/outlet.
        self._apply_velocity_boundary_conditions()

        # Free the seed's diffusion hierarchies (~3x velocity-field RAM). An
        # implicit solve rebuilds them with its own dt; an explicit solve is
        # matrix-free and never uses them.
        self.diffusion_solvers = None
        self.diffusion_index_maps = None
        self.diffusion_rhs = None
        self.diffusion_x0 = None

    def _compute_fluid_mask(self):
        """Cell-center fluid mask as a 1-byte array (1 = fluid, 0 = solid).

        Only the percolating pore cluster (connected to both the z=0 inlet and
        the z=max outlet) is kept, so isolated/dead-end pores never become
        singular rows in the pressure-Poisson. The bool mask is cached in
        `self._fluid_bool` and reused by `_build_pressure_poisson_system`.
        """
        self._fluid_bool = filter_percolating(self.volume > 0)
        return self._fluid_bool.astype(np.uint8)

    def _build_pressure_poisson_system(self):
        """Assemble the (geometry-fixed) pressure-Poisson matrix once.

        Assembles the solver's OWN unit MAC Laplacian directly from the fluid
        mask via `pressurePoisson.assemble_poisson` -- no VolumeManager, no Darcy
        code. One row per fluid cell (condensed); z=0 inlet / z=max outlet
        Dirichlet folded into the RHS. Matrix + AMG hierarchy are built once and
        cached (`self.poisson_solver`); only the RHS changes per iteration. The
        assembled operator is the *bare* Laplacian (unit weights = h**2 times the
        true operator); the h**2 factor is folded into the RHS in `poisson_step`.
        """
        dx, dy, dz = (float(s) for s in self.scale[:3])
        if not (np.isclose(dx, dy) and np.isclose(dy, dz)):
            raise NotImplementedError(
                "Pressure-Poisson assembly assumes isotropic voxels; "
                f"got scale=({dx}, {dy}, {dz})."
            )
        self._h = dx

        mask = self._fluid_bool
        if mask is None:
            mask = filter_percolating(self.volume > 0)
            self._fluid_bool = mask
        poisson = assemble_poisson(mask)      # own unit MAC Laplacian, no Darcy
        a_sparse, b_bc = poisson["a_sparse"], poisson["b"]

        solver = MultigridSolver(backend=self.backend)
        solver.set_linear_system(a_sparse, b_bc)
        solver.generate_preconditioner()

        self.poisson_solver = solver
        self.poisson_bc = b_bc.copy()          # boundary term re-added each step
        self.pressure_mask = mask.astype(np.uint8)
        self.pressure_mask_bool = mask
        self._poisson_n = poisson["n"]

        # Preallocate: dense divergence (pressure-shaped) + condensed RHS vector.
        self.div = np.zeros(self.volume.shape, dtype=np.float64)
        self.rhs = np.zeros(poisson["n"], dtype=np.float64)

        # Warm-start pressure (condensed), seeded from any initial guess.
        if self.p is not None:
            self.p_condensed = self.p[self.pressure_mask_bool].astype(np.float64)
        else:
            self.p_condensed = np.zeros(poisson["n"], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Projection iteration: predictor -> poisson -> corrector
    # ------------------------------------------------------------------ #
    def predictor_step(self, dt):
        """Stage 1: diffusion-only predictor (Stokes has no advection).

        Dispatches on the `predictor` parameter:
          - "explicit": u* = u^n + dt*(nu*laplacian(u^n) + f)   (forward Euler)
          - "implicit": (I - dt*nu*laplacian) u* = u^n + dt*f    (backward Euler)

        Both write into the preallocated `*_buf` ping-pong buffers (no
        allocation) and return them as (u_star, v_star, w_star).
        """
        if self.params["predictor"] == "implicit":
            return self._implicit_predictor_step(dt)

        dx, dy, dz = (float(s) for s in self.scale[:3])
        nu = self.params["viscosity"]
        fx, fy, fz = (float(f) for f in self.params["body_force"])
        _diffuse_jit(
            self.u, self.v, self.w,
            self.u_mask, self.v_mask, self.w_mask,
            nu, dx, dy, dz,
            fx, fy, fz,
            dt,
            self.u_buf, self.v_buf, self.w_buf,
        )
        return self.u_buf, self.v_buf, self.w_buf

    def _build_diffusion_systems(self, dt):
        """Assemble the implicit diffusion operator M = I - dt*nu*L per velocity
        component and build its multigrid hierarchy (once; dt and geometry are
        fixed). Each component's active-face set differs, so there are three
        independent condensed systems.
        """
        dx, dy, dz = (float(s) for s in self.scale[:3])
        inv_dx2, inv_dy2, inv_dz2 = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
        coef = dt * self.params["viscosity"]

        self.diffusion_solvers = []
        self.diffusion_index_maps = []
        self.diffusion_rhs = []
        self.diffusion_x0 = []
        for mask in (self.u_mask, self.v_mask, self.w_mask):
            val, col_idx, row_ptr, index_map, n = _assemble_diffusion_csr(
                mask, inv_dx2, inv_dy2, inv_dz2, coef
            )
            a_sparse = {"val": val, "col_idx": col_idx, "row_ptr": row_ptr}
            solver = MultigridSolver(backend=self.backend)
            solver.set_linear_system(a_sparse, np.zeros(n, dtype=np.float64))
            solver.generate_preconditioner()
            self.diffusion_solvers.append(solver)
            self.diffusion_index_maps.append(index_map)
            self.diffusion_rhs.append(np.zeros(n, dtype=np.float64))
            self.diffusion_x0.append(np.zeros(n, dtype=np.float64))

        # Scratch for the incremental predictor source u^n - (dt/rho) grad p^n.
        self._src_u = np.zeros_like(self.u)
        self._src_v = np.zeros_like(self.v)
        self._src_w = np.zeros_like(self.w)

    def _implicit_predictor_step(self, dt):
        """Backward-Euler diffusion predictor solved with multigrid per component.

            (I - dt*nu*L) u* = u^n + dt*f - (dt/rho) grad p^n  (+ open-boundary)

        This is the *incremental* pressure-correction predictor: it carries the
        previous pressure gradient so the projection error vanishes at steady
        state (see `_poisson_increment_step`), which is what lets dt be large.
        The matrices/hierarchies are built once (`_build_diffusion_systems`);
        each call rebuilds only the RHS and solves warm-started from u^n.
        """
        if self.diffusion_solvers is None:
            self._build_diffusion_systems(dt)

        dx, dy, dz = (float(s) for s in self.scale[:3])
        inv_dx2, inv_dy2, inv_dz2 = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
        nu = self.params["viscosity"]
        coef = dt * nu
        rho = self.params["density"]
        forces = tuple(float(f) for f in self.params["body_force"])
        fields = (self.u, self.v, self.w)
        masks = (self.u_mask, self.v_mask, self.w_mask)
        buffers = (self.u_buf, self.v_buf, self.w_buf)

        # Source = u^n - (dt/rho) grad p^n on active faces (open-boundary faces,
        # untouched by the masked gradient, keep their u^n values for coupling).
        self._src_u[:] = self.u
        self._src_v[:] = self.v
        self._src_w[:] = self.w
        _apply_pressure_gradient_jit(
            self._src_u, self._src_v, self._src_w, self.p,
            self.u_mask, self.v_mask, self.w_mask,
            dt / rho, dx, dy, dz,
        )
        sources = (self._src_u, self._src_v, self._src_w)

        for c in range(3):
            source, field, mask = sources[c], fields[c], masks[c]
            idx = self.diffusion_index_maps[c]
            rhs, x0 = self.diffusion_rhs[c], self.diffusion_x0[c]
            _diffusion_rhs(source, mask, idx, inv_dx2, inv_dy2, inv_dz2,
                           coef, dt * forces[c], rhs)
            _gather_face_to_condensed(field, idx, x0)   # warm-start from u^n
            solver = self.diffusion_solvers[c]
            solver.b_array = rhs
            x, _, _ = solver.solve_pcg(X0=x0)
            _scatter_condensed_to_face(x, idx, buffers[c])
        return buffers

    def poisson_step(self, u_star, v_star, w_star, dt):
        """Stage 2: solve the pressure-Poisson equation for u*.

            laplacian(p) = (rho / dt) * div(u*)

        Builds the RHS from `div(u*)` on fluid cells, then solves with the
        cached multigrid solver (warm-started from the previous `self.p`).
        Returns the new pressure field.

        The assembled pressure-Poisson operator is the *bare* Laplacian
        (unit face weights), i.e. h**2 times the true operator. Multiplying the
        source by h**2 rescales the whole equation, so the boundary term
        `poisson_bc` (already in bare-Laplacian units) is added as-is:

            A_bare p = h**2 * (rho / dt) * div(u*) + poisson_bc
        """
        if self.poisson_solver is None:
            self._build_pressure_poisson_system()

        dx, dy, dz = (float(s) for s in self.scale[:3])
        _divergence_jit(u_star, v_star, w_star, self.pressure_mask,
                        dx, dy, dz, self.div)

        rho = self.params["density"]
        factor = (self._h ** 2) * rho / dt
        self.rhs[:] = self.poisson_bc
        self.rhs += factor * self.div[self.pressure_mask_bool]

        # Reuse the cached matrix + preconditioner; only the RHS changed.
        self.poisson_solver.b_array = self.rhs
        x, self.poisson_error, self.poisson_iterations = \
            self.poisson_solver.solve_pcg(X0=self.p_condensed)

        self.p_condensed = x
        self.p = ravel(x, self.pressure_mask_bool)
        return self.p

    def _seed_incremental_pressure(self):
        """Seed p^0 as the harmonic pressure satisfying the inlet/outlet drive.

        Solves `A p^0 = poisson_bc` (i.e. div-free RHS: only the boundary term).
        The incremental scheme then keeps this drive in `p` and solves only the
        homogeneous-BC increment `phi` each step, so p carries the pressure drop
        while phi -> 0 at steady state.
        """
        n = self._poisson_n
        self.phi_condensed = np.zeros(n, dtype=np.float64)
        self.phi_full = np.zeros(self.volume.shape, dtype=np.float64)
        self.poisson_solver.b_array = self.poisson_bc.copy()
        p0, _, _ = self.poisson_solver.solve_pcg(X0=self.p_condensed)
        self.p_condensed = p0
        self.p = ravel(p0, self.pressure_mask_bool)

    def _poisson_increment_step(self, u_star, v_star, w_star, dt):
        """Incremental pressure-correction Poisson solve.

            lap(phi) = (rho/dt) div(u*),   phi = 0 at inlet/outlet (homogeneous)

        Unlike `poisson_step`, the boundary term `poisson_bc` is NOT added, so
        the solve yields the pressure *increment* phi with homogeneous Dirichlet
        ghosts. The running pressure is updated `p += phi`. Returns the phi field
        (cell-centered) for the corrector. Warm-started from the previous phi
        (which decays to 0 as the field settles).
        """
        dx, dy, dz = (float(s) for s in self.scale[:3])
        _divergence_jit(u_star, v_star, w_star, self.pressure_mask,
                        dx, dy, dz, self.div)

        rho = self.params["density"]
        factor = (self._h ** 2) * rho / dt
        self.rhs[:] = factor * self.div[self.pressure_mask_bool]

        self.poisson_solver.b_array = self.rhs
        phi, self.poisson_error, self.poisson_iterations = \
            self.poisson_solver.solve_pcg(X0=self.phi_condensed)

        self.phi_condensed = phi
        self.p_condensed = self.p_condensed + phi
        self.p = ravel(self.p_condensed, self.pressure_mask_bool)
        self.phi_full[:] = 0.0
        self.phi_full[self.pressure_mask_bool] = phi
        return self.phi_full

    def corrector_step(self, u_star, v_star, w_star, pressure, dt):
        """Stage 3: project the intermediate velocity to divergence-free.

            u^{n+1} = u* - (dt / rho) * grad(p)

        Subtracts the MAC pressure gradient in place on the `u_star` buffers and
        returns them. Because `pressure` solves the Poisson equation assembled
        from the same MAC divergence/gradient operators (`L = div . grad`), the
        result is discretely divergence-free on the fluid cells. Wall faces
        (mask 0) are left untouched (they stay at 0, i.e. no-slip).
        """
        dx, dy, dz = (float(s) for s in self.scale[:3])
        coef = dt / self.params["density"]
        _apply_pressure_gradient_jit(
            u_star, v_star, w_star,
            pressure,
            self.u_mask, self.v_mask, self.w_mask,
            coef, dx, dy, dz,
        )
        return u_star, v_star, w_star

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    def solve(self):
        """Iterate the projection scheme to steady state.

        The pseudo-time step is constant (viscous-limited, see
        `_compute_timestep`). Loop until the relative velocity change drops
        below `params["target_error"]` or `params["max_iterations"]` is hit:

            dt      = self._compute_timestep()
            u_star  = self.predictor_step(dt)
            p       = self.poisson_step(u_star, dt)
            u_new   = self.corrector_step(u_star, p, dt)
            self._apply_velocity_boundary_conditions()
            residual = self._velocity_residual(u_new, u_old)

        Returns a dict with the steady fields (`u, v, w, p`) and diagnostics
        (`iterations`, `residual`, `converged`).
        """
        if self.u is None:
            # Default warm start: compute a fast-Laplacian pressure and seed from
            # it, unless the caller supplied their own guess or opted out.
            if (self.fast_laplacian_guess and self.initial_pressure is None
                    and self.initial_velocity is None):
                self._compute_fast_laplacian_guess()
            self.create_velocity_arrays()
        if self.poisson_solver is None:
            self._build_pressure_poisson_system()

        dt = self._compute_timestep()
        max_iterations = self.params["max_iterations"]
        target_error = self.params["target_error"]

        # Incremental pressure-correction (implicit predictor): seed the driving
        # pressure once, then solve only the homogeneous-BC increment each step.
        # The implicit pseudo-transient converges to the true steady state for
        # channel-like geometry but can stall short of it on complex pore media.
        # So once the implicit phase settles (velocity step below tolerance) we
        # ALWAYS hand off to the always-correct explicit predictor, which either
        # confirms the field in ~1 step (implicit was right) or keeps correcting
        # to the true steady state (implicit had stalled). Same step criterion
        # throughout; the explicit finish guarantees correctness.
        started_implicit = self.params["predictor"] == "implicit"
        incremental = started_implicit
        self.fell_back = False
        self.implicit_iterations = 0
        if started_implicit:
            self._build_diffusion_systems(dt)
            self._seed_incremental_pressure()

        # Stopping bookkeeping. `stop_reason` records why the loop ended:
        #   "target"       -> residual dropped below target_error
        #   "stagnation"   -> residual plateaued (see _residual_stagnated)
        #   "max_iterations" -> ran out of iterations without either
        # `stagnated` is True when the final stop was the plateau safety net
        # rather than the target (the field is steady but never reached target).
        self.stop_reason = "max_iterations"
        self.stagnated = False
        residual_history = []

        residual = np.inf
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            # Predictor writes the ping-pong buffers from the current fields.
            u_star, v_star, w_star = self.predictor_step(dt)
            # Open the inlet/outlet before measuring divergence so the Poisson
            # RHS is not polluted by the predictor zeroing the boundary faces.
            self._apply_velocity_boundary_conditions((u_star, v_star, w_star))

            if incremental:
                phi = self._poisson_increment_step(u_star, v_star, w_star, dt)
                self.corrector_step(u_star, v_star, w_star, phi, dt)
            else:
                p = self.poisson_step(u_star, v_star, w_star, dt)
                self.corrector_step(u_star, v_star, w_star, p, dt)

            # Ping-pong swap: the buffers now hold u^{n+1}.
            self.u, self.u_buf = self.u_buf, self.u
            self.v, self.v_buf = self.v_buf, self.v
            self.w, self.w_buf = self.w_buf, self.w
            self._apply_velocity_boundary_conditions()

            residual = self._velocity_residual(
                (self.u, self.v, self.w),
                (self.u_buf, self.v_buf, self.w_buf),
                dt,
            )
            residual_history.append(residual)

            hit_target = residual < target_error
            stagnated = self._residual_stagnated(residual_history)
            if hit_target or stagnated:
                if incremental:
                    # Implicit phase settled (target reached OR plateaued) -> hand
                    # off to the always-correct explicit predictor to verify /
                    # finish (do not declare convergence yet). Reset the residual
                    # history so the explicit phase gets a fresh plateau baseline:
                    # switching predictors makes the residual jump.
                    incremental = False
                    self.implicit_iterations = iteration
                    self.params["predictor"] = "explicit"
                    dt = self._compute_timestep()   # -> explicit (stable) dt
                    residual_history = []
                    continue
                converged = True
                self.stagnated = stagnated and not hit_target
                self.stop_reason = "target" if hit_target else "stagnation"
                break

        # Restore the requested predictor (the handoff mutated it in place) and
        # flag whether the explicit finish had to do real work (implicit stall).
        if started_implicit:
            self.params["predictor"] = "implicit"
            self.fell_back = (iteration - self.implicit_iterations) > 2

        self.iteration = iteration
        self.residual = residual
        return {
            "u": self.u, "v": self.v, "w": self.w, "p": self.p,
            "iterations": iteration, "residual": residual,
            "converged": converged,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_timestep(self):
        """Constant pseudo-time step.

        Explicit predictor (viscous-stability limited):
            dt = time_step_factor * min(dx_i^2) / (2 * ndim * nu)
        Implicit predictor (unconditionally stable, so uncapped):
            dt = implicit_dt_factor * min(dx_i^2) / nu

        No advection term, so there is no convective CFL constraint.
        """
        dx, dy, dz = (float(s) for s in self.scale[:3])
        nu = self.params["viscosity"]
        min_dx2 = min(dx * dx, dy * dy, dz * dz)
        if self.params["predictor"] == "implicit":
            return self.params["implicit_dt_factor"] * min_dx2 / nu
        ndim = 3
        return self.params["time_step_factor"] * min_dx2 / (2.0 * ndim * nu)

    def _residual_stagnated(self, history):
        """True when the convergence residual has plateaued.

        The pseudo-transient residual decreases by orders of magnitude while the
        field is still developing, then flattens at a noise floor once the field
        is effectively steady. This detects that flat tail so the solve stops
        when converged even if the floor sits above `target_error` (otherwise it
        would grind to `max_iterations`). We compare the latest residual to the
        one `stagnation_window` iterations back: if it improved by less than
        `stagnation_tol` (fractionally) over that window -- including going flat
        or rising -- the field is no longer meaningfully changing.
        """
        window = self.params["stagnation_window"]
        if window <= 0 or len(history) <= window:
            return False
        past = history[-1 - window]
        recent = history[-1]
        if not np.isfinite(past) or past <= 0.0:
            return False
        return (past - recent) < self.params["stagnation_tol"] * past

    def _apply_velocity_boundary_conditions(self, fields=None):
        """Enforce inlet/outlet velocity BCs on u, v, w (in place).

        `fields` is an optional `(u, v, w)` tuple; defaults to the solver's own
        fields. No-slip walls are already held at 0 by the masked kernels, so
        the only active condition is the open (Neumann / fully-developed) inlet
        and outlet: the z-boundary faces copy their adjacent interior face for
        fluid columns, which lets flux enter/leave the duct while keeping the
        normal velocity gradient zero.
        """
        if fields is None:
            u, v, w = self.u, self.v, self.w
        else:
            u, v, w = fields

        if self.boundary_volume is None:
            d = self.volume.shape[2]
            w[:, :, 0] = w[:, :, 1] * self.fluid_mask[:, :, 0]
            w[:, :, d] = w[:, :, d - 1] * self.fluid_mask[:, :, d - 1]

    def _momentum_residual(self):
        """Steady-state momentum residual R = nu*lap(u) - grad(p)/rho + f.

        This is the honest, dt-independent convergence measure: it is what the
        Stokes equation demands be zero at steady state, so unlike the velocity
        *step* it cannot be fooled by a pseudo-transient iteration that has
        stalled short of the solution. Returned as max|R| / max|u| over active
        faces. Uses the `_src_*` buffers as scratch (free at call time).
        """
        nu = self.params["viscosity"]
        rho = self.params["density"]
        dx, dy, dz = (float(s) for s in self.scale[:3])
        fx, fy, fz = (float(f) for f in self.params["body_force"])
        ru, rv, rw = self._src_u, self._src_v, self._src_w

        # ru = u + nu*lap(u)  (dt=1, f=0) -> subtract u to get nu*lap(u).
        _diffuse_jit(self.u, self.v, self.w,
                     self.u_mask, self.v_mask, self.w_mask,
                     nu, dx, dy, dz, 0.0, 0.0, 0.0, 1.0, ru, rv, rw)
        ru -= self.u
        rv -= self.v
        rw -= self.w
        # ru -= grad(p)/rho  on active faces.
        _apply_pressure_gradient_jit(ru, rv, rw, self.p,
                                     self.u_mask, self.v_mask, self.w_mask,
                                     1.0 / rho, dx, dy, dz)
        # + body force on active faces.
        if fx != 0.0:
            ru[self.u_mask == 1] += fx
        if fy != 0.0:
            rv[self.v_mask == 1] += fy
        if fz != 0.0:
            rw[self.w_mask == 1] += fz

        max_r = 0.0
        max_u = 0.0
        for R, field, mask in ((ru, self.u, self.u_mask),
                               (rv, self.v, self.v_mask),
                               (rw, self.w, self.w_mask)):
            active = mask == 1
            if active.any():
                max_r = max(max_r, float(np.abs(R[active]).max()))
                max_u = max(max_u, float(np.abs(field[active]).max()))
        if max_u == 0.0:
            return 0.0 if max_r == 0.0 else np.inf
        return max_r / max_u

    def _velocity_residual(self, new_fields, old_fields, dt):
        """Convergence metric over fluid faces, per `convergence_criterion`.

        Both criteria share the raw quantity max|u_new - u_old| / max|u_new|:
          - "step": returned as-is (relative change per iteration; scales with
            dt, so a smaller pseudo-time step reads as "more converged" even
            when the field is equally far from steady).
          - "residual": divided by dt. Since the scheme gives
            u_new - u_old = dt * (nu*lap(u) + f - grad(p)/rho), this is the
            steady-state momentum residual max|R| / max|u|, a property of the
            field that does NOT depend on the pseudo-time step.
        """
        masks = (self.u_mask, self.v_mask, self.w_mask)
        max_diff = 0.0
        max_val = 0.0
        for new, old, mask in zip(new_fields, old_fields, masks):
            diff, val = _max_relative_change_jit(
                new.ravel(), old.ravel(), mask.ravel(), threads=1
            )
            max_diff = max(max_diff, diff)
            max_val = max(max_val, val)
        if max_val == 0.0:
            return 0.0 if max_diff == 0.0 else np.inf
        metric = max_diff / max_val
        if self.params["convergence_criterion"] == "residual":
            metric /= dt
        return metric


# ====================================================================== #
# Module-level @njit kernels (bodies to be implemented)
# ====================================================================== #

@njit(parallel=True)
def _diffuse_component_jit(field, mask, nu, f, inv_dx2, inv_dy2, inv_dz2, dt, out):
    """Single-component diffusion predictor: out = field + dt*(nu*lap + f).

    Central-difference 7-point Laplacian over the face field. Neighbours that
    fall outside the array are treated as 0 (no-slip); wall faces already hold
    0, so reading them directly gives the correct no-slip contribution. Faces
    where `mask == 0` are set to 0 in the output.
    """
    nx, ny, nz = field.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k] == 0:
                    out[i, j, k] = 0.0
                    continue
                c = field[i, j, k]
                xm = field[i - 1, j, k] if i > 0 else 0.0
                xp = field[i + 1, j, k] if i < nx - 1 else 0.0
                ym = field[i, j - 1, k] if j > 0 else 0.0
                yp = field[i, j + 1, k] if j < ny - 1 else 0.0
                zm = field[i, j, k - 1] if k > 0 else 0.0
                zp = field[i, j, k + 1] if k < nz - 1 else 0.0
                lap = ((xp - 2.0 * c + xm) * inv_dx2
                       + (yp - 2.0 * c + ym) * inv_dy2
                       + (zp - 2.0 * c + zm) * inv_dz2)
                out[i, j, k] = c + dt * (nu * lap + f)


@njit
def _diffuse_jit(
    u, v, w,
    u_mask, v_mask, w_mask,
    nu, dx, dy, dz,
    fx, fy, fz,
    dt,
    u_out, v_out, w_out,
):
    """Compute u* = u + dt * (nu * laplacian(u) + f) per component.

    Stokes flow has no advection, so the predictor is pure diffusion plus the
    (optional) body force. Central-difference 7-point Laplacian. Writes into
    the preallocated `u_out, v_out, w_out`; wall faces (mask 0) are left at 0.
    """
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)
    _diffuse_component_jit(u, u_mask, nu, fx, inv_dx2, inv_dy2, inv_dz2, dt, u_out)
    _diffuse_component_jit(v, v_mask, nu, fy, inv_dx2, inv_dy2, inv_dz2, dt, v_out)
    _diffuse_component_jit(w, w_mask, nu, fz, inv_dx2, inv_dy2, inv_dz2, dt, w_out)


@njit(parallel=True)
def _divergence_jit(
    u, v, w,
    fluid_mask,
    dx, dy, dz,
    div_out,
):
    """Cell-centered divergence div(u) = du/dx + dv/dy + dw/dz.

    On the MAC grid each term differences the two bounding faces of the cell:
    for cell (i,j,k) the x-term is (u[i+1] - u[i]) / dx, etc. Writes `div_out`
    (shape of the pressure field); non-fluid cells are set to 0. Wall faces
    hold 0, so no-flux boundaries fall out naturally.
    """
    W, H, D = fluid_mask.shape
    for i in prange(W):
        for j in range(H):
            for k in range(D):
                if fluid_mask[i, j, k] == 0:
                    div_out[i, j, k] = 0.0
                    continue
                du = (u[i + 1, j, k] - u[i, j, k]) / dx
                dv = (v[i, j + 1, k] - v[i, j, k]) / dy
                dw = (w[i, j, k + 1] - w[i, j, k]) / dz
                div_out[i, j, k] = du + dv + dw


@njit(parallel=True)
def _apply_pressure_gradient_jit(
    u, v, w,
    pressure,
    u_mask, v_mask, w_mask,
    coef, dx, dy, dz,
):
    """In-place velocity projection: u -= coef * grad(p), coef = dt / rho.

    Each face subtracts the pressure difference of its two bounding cells,
    scaled by `coef / d`. Wall faces (mask 0) are left unchanged. Only interior
    faces (index >= 1 along the face normal) can be active, so the two bounding
    cell indices are always in range.
    """
    W, H, D = pressure.shape

    # x-faces: gradient between cells (i-1) and i.
    for i in prange(1, W):
        for j in range(H):
            for k in range(D):
                if u_mask[i, j, k]:
                    u[i, j, k] -= coef * (pressure[i, j, k] - pressure[i - 1, j, k]) / dx

    # y-faces: gradient between cells (j-1) and j.
    for i in prange(W):
        for j in range(1, H):
            for k in range(D):
                if v_mask[i, j, k]:
                    v[i, j, k] -= coef * (pressure[i, j, k] - pressure[i, j - 1, k]) / dy

    # z-faces: gradient between cells (k-1) and k.
    for i in prange(W):
        for j in range(H):
            for k in range(1, D):
                if w_mask[i, j, k]:
                    w[i, j, k] -= coef * (pressure[i, j, k] - pressure[i, j, k - 1]) / dz


@njit
def _assemble_diffusion_csr(mask, inv_dx2, inv_dy2, inv_dz2, coef):
    """Condensed CSR for the implicit diffusion operator M = I - coef*L.

    `L` is the same 7-point face Laplacian the explicit predictor uses
    (diagonal -2*(inv_dx2+inv_dy2+inv_dz2); off-diagonal +inv per neighbour),
    and `coef = dt*nu`. Only active faces (`mask != 0`) are unknowns; a masked
    neighbour is Dirichlet (its value is moved to the RHS by `_diffusion_rhs`),
    so it contributes no column here. The result is a symmetric positive-definite
    M-matrix in the project CSR convention (row_ptr length N), plus an
    `index_map` giving the condensed row of each active face (-1 if inactive).

    The stencil diagonal is the full -2*S regardless of whether neighbours are
    walls / out of bounds, matching `_diffuse_component_jit` exactly, so the
    implicit and explicit predictors share a fixed point.
    """
    nx, ny, nz = mask.shape
    index_map = -np.ones((nx, ny, nz), dtype=np.int64)
    n_active = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k] != 0:
                    index_map[i, j, k] = n_active
                    n_active += 1

    counts = np.zeros(n_active, dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = index_map[i, j, k]
                if r < 0:
                    continue
                c = 1  # diagonal
                if i > 0 and mask[i - 1, j, k]:
                    c += 1
                if i < nx - 1 and mask[i + 1, j, k]:
                    c += 1
                if j > 0 and mask[i, j - 1, k]:
                    c += 1
                if j < ny - 1 and mask[i, j + 1, k]:
                    c += 1
                if k > 0 and mask[i, j, k - 1]:
                    c += 1
                if k < nz - 1 and mask[i, j, k + 1]:
                    c += 1
                counts[r] = c

    nnz = 0
    for r in range(n_active):
        nnz += counts[r]
    val = np.zeros(nnz, dtype=np.float64)
    col_idx = np.zeros(nnz, dtype=np.int64)
    row_ptr = np.zeros(n_active, dtype=np.int64)
    for r in range(1, n_active):
        row_ptr[r] = row_ptr[r - 1] + counts[r - 1]

    diag = 1.0 + 2.0 * coef * (inv_dx2 + inv_dy2 + inv_dz2)
    off_x = -coef * inv_dx2
    off_y = -coef * inv_dy2
    off_z = -coef * inv_dz2
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = index_map[i, j, k]
                if r < 0:
                    continue
                pos = row_ptr[r]
                val[pos] = diag
                col_idx[pos] = r
                pos += 1
                if i > 0 and mask[i - 1, j, k]:
                    val[pos] = off_x; col_idx[pos] = index_map[i - 1, j, k]; pos += 1
                if i < nx - 1 and mask[i + 1, j, k]:
                    val[pos] = off_x; col_idx[pos] = index_map[i + 1, j, k]; pos += 1
                if j > 0 and mask[i, j - 1, k]:
                    val[pos] = off_y; col_idx[pos] = index_map[i, j - 1, k]; pos += 1
                if j < ny - 1 and mask[i, j + 1, k]:
                    val[pos] = off_y; col_idx[pos] = index_map[i, j + 1, k]; pos += 1
                if k > 0 and mask[i, j, k - 1]:
                    val[pos] = off_z; col_idx[pos] = index_map[i, j, k - 1]; pos += 1
                if k < nz - 1 and mask[i, j, k + 1]:
                    val[pos] = off_z; col_idx[pos] = index_map[i, j, k + 1]; pos += 1

    return val, col_idx, row_ptr, index_map, n_active


@njit(parallel=True)
def _diffusion_rhs(field, mask, index_map, inv_dx2, inv_dy2, inv_dz2,
                   coef, dt_f, rhs_out):
    """RHS of the implicit diffusion solve: field + dt*f + boundary coupling.

        rhs[r] = field[face] + dt*f
                 + coef * sum_{inactive neighbours} inv * field[neighbour]

    Inactive neighbours are walls (field == 0, no contribution) or open
    inlet/outlet faces (field == opened value -> a lagged Neumann term). This
    reproduces the explicit stencil's reading of the current boundary field
    while the interior is solved implicitly.
    """
    nx, ny, nz = mask.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                r = index_map[i, j, k]
                if r < 0:
                    continue
                acc = field[i, j, k] + dt_f
                if i > 0 and mask[i - 1, j, k] == 0:
                    acc += coef * inv_dx2 * field[i - 1, j, k]
                if i < nx - 1 and mask[i + 1, j, k] == 0:
                    acc += coef * inv_dx2 * field[i + 1, j, k]
                if j > 0 and mask[i, j - 1, k] == 0:
                    acc += coef * inv_dy2 * field[i, j - 1, k]
                if j < ny - 1 and mask[i, j + 1, k] == 0:
                    acc += coef * inv_dy2 * field[i, j + 1, k]
                if k > 0 and mask[i, j, k - 1] == 0:
                    acc += coef * inv_dz2 * field[i, j, k - 1]
                if k < nz - 1 and mask[i, j, k + 1] == 0:
                    acc += coef * inv_dz2 * field[i, j, k + 1]
                rhs_out[r] = acc


@njit(parallel=True)
def _gather_face_to_condensed(field, index_map, out):
    """out[index_map[face]] = field[face] over active faces (for warm start)."""
    nx, ny, nz = index_map.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                r = index_map[i, j, k]
                if r >= 0:
                    out[r] = field[i, j, k]


@njit(parallel=True)
def _scatter_condensed_to_face(x, index_map, field_out):
    """field_out[face] = x[index_map[face]]; inactive faces set to 0."""
    nx, ny, nz = index_map.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                r = index_map[i, j, k]
                if r >= 0:
                    field_out[i, j, k] = x[r]
                else:
                    field_out[i, j, k] = 0.0


@njit(parallel=True)
def _max_relative_change_jit(new, old, mask, threads):
    """Return (max|new - old|, max|new|) over active (mask != 0) flat entries.

    The caller combines these across the three velocity components into the
    single relative steady-state metric max|dU| / max|U|. Reductions use the
    thread-partitioning style of `_square_sum_vector` in darcySolver.py.
    """
    n = new.size
    partial_diff = np.zeros(threads, dtype=np.float64)
    partial_val = np.zeros(threads, dtype=np.float64)

    for t in prange(threads):
        thread_start = t * n // threads
        thread_end = (t + 1) * n // threads
        local_diff = 0.0
        local_val = 0.0
        for i in range(thread_start, thread_end):
            if mask[i] != 0:
                diff = abs(new[i] - old[i])
                if diff > local_diff:
                    local_diff = diff
                val = abs(new[i])
                if val > local_val:
                    local_val = val
        partial_diff[t] = local_diff
        partial_val[t] = local_val

    return partial_diff.max(), partial_val.max()
