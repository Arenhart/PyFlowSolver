"""Steady incompressible Stokes (creeping) flow solver.

See `stokes_solver.md` for the full algorithm description. The solver is kept
deliberately simple: Stokes flow is linear (no advection, no inertia), so we
obtain the steady field with a pseudo-transient projection iteration and reuse
`DarcySolver` for the pressure-Poisson solve.

The module mirrors the structure of `darcySolver.py`:
- `DEFAULT_PARAMS` + `**params` validation in `__init__`
- a public `solve` driver that iterates to steady state
- module-level `@njit` / `@njit(parallel=True)` kernels for the hot loops

The pressure-Poisson linear system reuses VolumeManager's condensed CSR
assembly (one row per fluid cell) and is solved with `DarcySolver`, which keeps
memory and CPU proportional to the pore space rather than the full voxel grid --
important for the low-porosity (10-20%) cylindrical samples this targets. The
assembly currently assumes isotropic voxels.
"""

import numpy as np
from numba import njit, prange

from pyflowsolver.solver import Solver
from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET


class StokesSolver(Solver):
    # Stopping criteria for the projection iteration (see `_velocity_residual`).
    CONVERGENCE_CRITERIA = ("step", "residual")

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
    }

    def __init__(self, volume_manager, initial_pressure=None,
                 initial_velocity=None, **params):
        """
        volume_manager: a VolumeManager describing the (fixed) voxel geometry.
        initial_pressure: optional first guess for the cell-center pressure,
            an ndarray shaped like the volume `(w, h, d)`. Defaults to zeros.
        initial_velocity: optional first guess for the MAC velocity field, a
            tuple/list `(u, v, w)` of the staggered face arrays shaped
            `(w+1, h, d)`, `(w, h+1, d)`, `(w, h, d+1)`. Defaults to zeros.
            The guess is applied in `create_velocity_arrays`; wall faces are
            re-zeroed there to keep the no-slip invariant.
        params: any key in DEFAULT_PARAMS (see module docstring / md file).
        """
        self.volume_manager = volume_manager
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

        # Cached pressure-Poisson DarcySolver (matrix + preconditioner)
        self.poisson_solver = None

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

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def create_velocity_arrays(self):
        """Allocate MAC velocity/pressure fields, scratch buffers, and masks.

        Sizes derive from `self.volume_manager.volume.shape == (w, h, d)`:
            u: (w+1, h, d)   v: (w, h+1, d)   w: (w, h, d+1)   p: (w, h, d)

        Everything the iteration touches is allocated here exactly once (fields,
        the `*_buf` ping-pong buffers, and `rhs`) so the solve loop performs no
        allocations. Masks are stored as 1-byte arrays to keep RAM down:
        `fluid_mask` (cell-center) plus the per-component face masks marking
        active unknowns (both neighbouring voxels fluid) vs no-slip walls
        (touching a solid voxel).
        """
        w, h, d = self.volume_manager.volume.shape

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
        if self.initial_velocity is not None:
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

    def _compute_fluid_mask(self):
        """Cell-center fluid mask as a 1-byte array (1 = fluid, 0 = solid).

        For an irregular boundary, fluid cells are PORE/INLET/OUTLET voxels;
        otherwise (regular volume) any voxel with positive conductivity.
        """
        vm = self.volume_manager
        if vm.boundary_volume is not None:
            fluid = np.isin(vm.boundary_volume, (PORE, INLET, OUTLET))
        else:
            fluid = vm.volume > 0
        return fluid.astype(np.uint8)

    def _build_pressure_poisson_system(self):
        """Assemble the (geometry-fixed) pressure-Poisson matrix once.

        Uses `self.volume_manager` to build the Laplacian sparse system in the
        CSR convention `DarcySolver` consumes, wraps it in a `DarcySolver`, and
        generates its preconditioner. Only the RHS changes per iteration, so
        the matrix and preconditioner are cached in `self.poisson_solver`.

        The system is the *condensed* CSR Laplacian VolumeManager produces:
        only fluid cells become unknowns (rows). At 10-20% porosity in
        cylindrical samples this is a large RAM/CPU saving over a dense grid.
        We rebuild it from a unit-conductivity copy of the geometry so the
        matrix is the pure geometric Laplacian (the pressure Poisson operator),
        with the same inlet=1 / outlet=0 Dirichlet structure as the Darcy path.
        """
        vm = self.volume_manager
        dx, dy, dz = (float(s) for s in vm.scale[:3])
        if not (np.isclose(dx, dy) and np.isclose(dy, dz)):
            raise NotImplementedError(
                "Pressure-Poisson assembly via VolumeManager assumes isotropic "
                f"voxels; got scale=({dx}, {dy}, {dz})."
            )
        self._h = dx

        # Unit-conductivity geometry so harmonic-mean face weights are all 1 and
        # VolumeManager builds the bare Laplacian (a scalar h**2 factor is
        # folded into the RHS in poisson_step).
        if vm.boundary_volume is not None:
            geom = (vm.boundary_volume == PORE).astype(np.float64)
            poisson_vm = VolumeManager(
                geom, scale=vm.scale, boundary_volume=vm.boundary_volume.copy()
            )
            pmask = poisson_vm.boundary_volume == PORE
        else:
            geom = (vm.volume > 0).astype(np.float64)
            poisson_vm = VolumeManager(geom, scale=vm.scale)
            pmask = poisson_vm.volume > 0

        a_sparse, b_bc = poisson_vm.get_sparse_system_jit()

        solver = DarcySolver()
        solver.set_linear_system(a_sparse, b_bc)
        solver.generate_preconditioner(preconditioner="inverse_diagonal")

        self.poisson_solver = solver
        self.poisson_vm = poisson_vm
        self.poisson_bc = b_bc.copy()          # boundary term re-added each step
        self.pressure_mask = pmask.astype(np.uint8)
        self.pressure_mask_bool = pmask.astype(bool)

        # Preallocate: dense divergence (pressure-shaped) + condensed RHS vector.
        self.div = np.zeros(vm.volume.shape, dtype=np.float64)
        self.rhs = np.zeros(poisson_vm.nonzeros, dtype=np.float64)

        # Warm-start pressure (condensed), seeded from any initial guess.
        if self.p is not None:
            self.p_condensed = self.p[self.pressure_mask_bool].astype(np.float64)
        else:
            self.p_condensed = np.zeros(poisson_vm.nonzeros, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Projection iteration: predictor -> poisson -> corrector
    # ------------------------------------------------------------------ #
    def predictor_step(self, dt):
        """Stage 1: diffusion-only predictor (Stokes has no advection).

            u* = u^n + dt * ( nu * laplacian(u^n) + f )

        Writes into the preallocated `*_buf` ping-pong buffers (no allocation)
        and returns them as (u_star, v_star, w_star).
        """
        dx, dy, dz = (float(s) for s in self.volume_manager.scale[:3])
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

    def poisson_step(self, u_star, v_star, w_star, dt):
        """Stage 2: solve the pressure-Poisson equation for u*.

            laplacian(p) = (rho / dt) * div(u*)

        Builds the RHS from `div(u*)` on fluid cells, then solves with the
        cached `DarcySolver` (warm-started from the previous `self.p`).
        Returns the new pressure field.

        The Poisson operator VolumeManager assembled is the *bare* Laplacian
        (unit face weights), i.e. h**2 times the true operator. Multiplying the
        source by h**2 rescales the whole equation, so the boundary term
        `poisson_bc` (already in bare-Laplacian units) is added as-is:

            A_bare p = h**2 * (rho / dt) * div(u*) + poisson_bc
        """
        if self.poisson_solver is None:
            self._build_pressure_poisson_system()

        dx, dy, dz = (float(s) for s in self.volume_manager.scale[:3])
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
        self.p = self.poisson_vm.ravel_sparse_solution(x)
        return self.p

    def corrector_step(self, u_star, v_star, w_star, pressure, dt):
        """Stage 3: project the intermediate velocity to divergence-free.

            u^{n+1} = u* - (dt / rho) * grad(p)

        Subtracts the MAC pressure gradient in place on the `u_star` buffers and
        returns them. Because `pressure` solves the Poisson equation assembled
        from the same MAC divergence/gradient operators (`L = div . grad`), the
        result is discretely divergence-free on the fluid cells. Wall faces
        (mask 0) are left untouched (they stay at 0, i.e. no-slip).
        """
        dx, dy, dz = (float(s) for s in self.volume_manager.scale[:3])
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
            self.create_velocity_arrays()
        if self.poisson_solver is None:
            self._build_pressure_poisson_system()

        dt = self._compute_timestep()
        max_iterations = self.params["max_iterations"]
        target_error = self.params["target_error"]

        residual = np.inf
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            # Predictor writes the ping-pong buffers from the current fields.
            u_star, v_star, w_star = self.predictor_step(dt)
            # Open the inlet/outlet before measuring divergence so the Poisson
            # RHS is not polluted by the predictor zeroing the boundary faces.
            self._apply_velocity_boundary_conditions((u_star, v_star, w_star))

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
            if residual < target_error:
                converged = True
                break

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
        """Constant viscous-limited pseudo-time step.

            dt = time_step_factor * min(dx_i^2) / (2 * ndim * nu)

        No advection term, so there is no convective CFL constraint.
        """
        dx, dy, dz = (float(s) for s in self.volume_manager.scale[:3])
        ndim = 3
        nu = self.params["viscosity"]
        min_dx2 = min(dx * dx, dy * dy, dz * dz)
        return self.params["time_step_factor"] * min_dx2 / (2.0 * ndim * nu)

    def _apply_velocity_boundary_conditions(self, fields=None):
        """Enforce inlet/outlet velocity BCs on u, v, w (in place).

        `fields` is an optional `(u, v, w)` tuple; defaults to the solver's own
        fields. No-slip walls are already held at 0 by the masked kernels, so
        the only active condition is the open (Neumann / fully-developed) inlet
        and outlet: the z-boundary faces copy their adjacent interior face for
        fluid columns, which lets flux enter/leave the duct while keeping the
        normal velocity gradient zero. Irregular boundaries drive their flow
        through the explicit INLET/OUTLET faces (already active), so no extra
        clamp is applied here.
        """
        if fields is None:
            u, v, w = self.u, self.v, self.w
        else:
            u, v, w = fields

        if self.volume_manager.boundary_volume is None:
            d = self.volume_manager.volume.shape[2]
            w[:, :, 0] = w[:, :, 1] * self.fluid_mask[:, :, 0]
            w[:, :, d] = w[:, :, d - 1] * self.fluid_mask[:, :, d - 1]

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
