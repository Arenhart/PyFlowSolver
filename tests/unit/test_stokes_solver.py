"""Unit tests for the Stokes solver, built up stepwise.

This first batch covers the *predictor* stage only (diffusion + body force),
its supporting setup (`create_velocity_arrays`, masks) and the viscous
pseudo-time step. A circular-duct geometry is also exercised here so the same
fixture can later be compared against the Hagen-Poiseuille profile once the
full projection solver is implemented.
"""

import numpy as np
import pytest

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.stokesSolver import (
    StokesSolver,
    _diffuse_jit,
    _divergence_jit,
    _apply_pressure_gradient_jit,
    _max_relative_change_jit,
)
from tests.unit.resources.ducts import make_circular_duct


def _csr_matvec(a_sparse, x):
    """Dense matrix-vector product for the condensed CSR convention (N-length
    row_ptr; last row runs to val.size)."""
    val = a_sparse["val"]
    col = a_sparse["col_idx"]
    rp = a_sparse["row_ptr"]
    n = x.size
    y = np.zeros(n, dtype=np.float64)
    for row in range(n):
        start = rp[row]
        stop = rp[row + 1] if row + 1 < n else val.size
        for idx in range(start, stop):
            y[row] += val[idx] * x[col[idx]]
    return y


def _fill_quadratic(arr, coeffs):
    """Fill `arr` in place with a*i^2 + b*j^2 + c*k^2 + linear + const.

    The exact discrete second difference of i^2 is 2, so the central-difference
    Laplacian of this field is the constant 2a/dx^2 + 2b/dy^2 + 2c/dz^2 at every
    interior point regardless of dx,dy,dz -> makes an exact analytic check.
    """
    a, b, c = coeffs
    i, j, k = np.indices(arr.shape).astype(np.float64)
    arr[...] = a * i * i + b * j * j + c * k * k + 0.3 * i + 0.2 * j + 0.1 * k + 5.0


# --------------------------------------------------------------------------- #
# Setup: array allocation, masks, timestep
# --------------------------------------------------------------------------- #
def test_create_velocity_arrays_shapes_and_masks():
    volume = np.ones((4, 5, 6), dtype=np.float64)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm)
    solver.create_velocity_arrays()

    w, h, d = volume.shape
    assert solver.u.shape == (w + 1, h, d)
    assert solver.v.shape == (w, h + 1, d)
    assert solver.w.shape == (w, h, d + 1)
    assert solver.p.shape == (w, h, d)
    assert solver.u_buf.shape == solver.u.shape
    assert solver.v_buf.shape == solver.v.shape
    assert solver.w_buf.shape == solver.w.shape

    # All fluid: interior faces active, domain-boundary faces are walls (0).
    assert solver.u_mask[1:w, :, :].all()
    assert not solver.u_mask[0, :, :].any()
    assert not solver.u_mask[w, :, :].any()
    assert solver.v_mask[:, 1:h, :].all()
    assert not solver.v_mask[:, 0, :].any()
    assert not solver.v_mask[:, h, :].any()
    assert solver.w_mask[:, :, 1:d].all()
    assert not solver.w_mask[:, :, 0].any()
    assert not solver.w_mask[:, :, d].any()


def test_face_mask_deactivated_by_solid_cell():
    volume = np.ones((4, 4, 4), dtype=np.float64)
    volume[1, 1, 1] = 0.0  # a single solid cell
    vm = VolumeManager(volume)
    solver = StokesSolver(vm)
    solver.create_velocity_arrays()

    # The two z-faces bounding the solid cell (1,1,1) must be no-slip walls.
    assert solver.w_mask[1, 1, 1] == 0
    assert solver.w_mask[1, 1, 2] == 0
    # A z-face away from the solid cell is still an active unknown.
    assert solver.w_mask[0, 0, 2] == 1


def test_compute_timestep():
    volume = np.ones((3, 3, 3), dtype=np.float64)
    scale = (0.1, 0.2, 0.4)
    vm = VolumeManager(volume, scale=scale)
    nu = 2.0
    factor = 0.5
    solver = StokesSolver(vm, viscosity=nu, time_step_factor=factor)

    dt = solver._compute_timestep()
    min_dx2 = min(0.1 ** 2, 0.2 ** 2, 0.4 ** 2)
    expected = factor * min_dx2 / (2.0 * 3 * nu)
    assert np.isclose(dt, expected)


# --------------------------------------------------------------------------- #
# Warm-start initial guesses
# --------------------------------------------------------------------------- #
def test_initial_guess_applied():
    volume = np.ones((4, 4, 4), dtype=np.float64)
    vm = VolumeManager(volume)
    w, h, d = volume.shape

    p0 = np.full((w, h, d), 3.0)
    u0 = np.full((w + 1, h, d), 1.0)
    v0 = np.full((w, h + 1, d), 2.0)
    w0 = np.full((w, h, d + 1), 5.0)

    solver = StokesSolver(vm, initial_pressure=p0, initial_velocity=(u0, v0, w0))
    solver.create_velocity_arrays()

    np.testing.assert_array_equal(solver.p, p0)
    # Interior active faces carry the guess.
    np.testing.assert_array_equal(solver.u[solver.u_mask == 1], 1.0)
    np.testing.assert_array_equal(solver.w[solver.w_mask == 1], 5.0)
    # No-slip x-boundary walls are forced to zero.
    np.testing.assert_array_equal(solver.u[solver.u_mask == 0], 0.0)
    # The inlet/outlet z-faces are masked but are NOT no-slip walls: they must
    # be reopened to the adjacent interior value so a seeded field keeps its
    # through-flux instead of being clamped shut.
    np.testing.assert_array_equal(solver.w[:, :, 0], 5.0)
    np.testing.assert_array_equal(solver.w[:, :, d], 5.0)

    # The guess must be copied, not aliased, so buffers stay independent.
    assert solver.u is not u0
    u0[...] = 99.0
    assert not np.any(solver.u == 99.0)


def test_seeding_converged_field_converges_immediately():
    """Regression: seeding a converged velocity field must not clamp the open
    inlet/outlet faces to zero. A fresh solver seeded with the steady solution
    should stop in a handful of iterations, not re-develop the whole flow."""
    volume = make_circular_duct(radius=4, length=6)
    base = StokesSolver(VolumeManager(volume.copy()),
                        target_error=1e-6, max_iterations=20000)
    r = base.solve()
    assert r["converged"]

    seeded = StokesSolver(
        VolumeManager(volume.copy()),
        initial_velocity=(r["u"].copy(), r["v"].copy(), r["w"].copy()),
        target_error=1e-6, max_iterations=20000,
    )
    result = seeded.solve()
    assert result["iterations"] <= 5          # essentially already converged
    assert result["iterations"] < r["iterations"] // 10


def test_default_initial_guess_is_zero():
    volume = np.ones((3, 3, 3), dtype=np.float64)
    solver = StokesSolver(VolumeManager(volume))
    solver.create_velocity_arrays()
    for field in (solver.u, solver.v, solver.w, solver.p):
        np.testing.assert_array_equal(field, 0.0)


def test_initial_guess_wrong_shape_raises():
    volume = np.ones((4, 4, 4), dtype=np.float64)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm, initial_pressure=np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        solver.create_velocity_arrays()


# --------------------------------------------------------------------------- #
# Predictor: diffusion kernel
# --------------------------------------------------------------------------- #
def test_diffuse_linear_field_has_zero_laplacian():
    """A field linear in the indices has zero Laplacian -> u* == u."""
    volume = np.ones((6, 6, 6), dtype=np.float64)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm, viscosity=1.5)
    solver.create_velocity_arrays()

    for field in (solver.u, solver.v, solver.w):
        _fill_quadratic(field, coeffs=(0.0, 0.0, 0.0))  # purely linear

    dt = 0.05
    u_star, v_star, w_star = solver.predictor_step(dt)

    for field, out in ((solver.u, u_star), (solver.v, v_star), (solver.w, w_star)):
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        np.testing.assert_allclose(out[interior], field[interior], rtol=1e-10)


def test_diffuse_quadratic_field_constant_laplacian():
    """Quadratic field -> exact constant Laplacian increment on interior faces."""
    volume = np.ones((6, 6, 6), dtype=np.float64)
    scale = (0.1, 0.2, 0.25)
    vm = VolumeManager(volume, scale=scale)
    nu = 1.3
    solver = StokesSolver(vm, viscosity=nu)
    solver.create_velocity_arrays()

    dx, dy, dz = scale
    inv = np.array([1 / dx ** 2, 1 / dy ** 2, 1 / dz ** 2])
    coeffs = {
        "u": (0.5, -0.3, 0.2),
        "v": (0.1, 0.4, -0.2),
        "w": (-0.25, 0.15, 0.35),
    }
    _fill_quadratic(solver.u, coeffs["u"])
    _fill_quadratic(solver.v, coeffs["v"])
    _fill_quadratic(solver.w, coeffs["w"])

    dt = 0.03
    u_star, v_star, w_star = solver.predictor_step(dt)

    for key, field, out in (
        ("u", solver.u, u_star),
        ("v", solver.v, v_star),
        ("w", solver.w, w_star),
    ):
        lap = 2.0 * np.dot(coeffs[key], inv)
        expected_increment = dt * nu * lap
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        increment = out[interior] - field[interior]
        # VolumeManager stores an iterable scale as float32, so dx,dy,dz carry
        # single-precision rounding; tolerance is set accordingly.
        np.testing.assert_allclose(increment, expected_increment, rtol=1e-5, atol=1e-9)


def test_body_force_added_on_active_faces():
    """With zero velocity, u* = dt * f on active faces and 0 on walls."""
    volume = np.ones((5, 5, 5), dtype=np.float64)
    vm = VolumeManager(volume)
    force = (0.7, -0.4, 1.1)
    dt = 0.02
    solver = StokesSolver(vm, viscosity=1.0, body_force=force)
    solver.create_velocity_arrays()

    u_star, v_star, w_star = solver.predictor_step(dt)

    for f, out, mask in (
        (force[0], u_star, solver.u_mask),
        (force[1], v_star, solver.v_mask),
        (force[2], w_star, solver.w_mask),
    ):
        np.testing.assert_allclose(out[mask == 1], dt * f, rtol=1e-12)
        np.testing.assert_array_equal(out[mask == 0], 0.0)


def test_predictor_leaves_walls_at_zero():
    """No-slip walls (mask 0) stay zero even with a non-zero field elsewhere."""
    volume = make_circular_duct(radius=4, length=6)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm, viscosity=1.0)
    solver.create_velocity_arrays()

    solver.u[...] = 1.0
    solver.v[...] = 1.0
    solver.w[...] = 1.0

    u_star, v_star, w_star = solver.predictor_step(0.01)

    np.testing.assert_array_equal(u_star[solver.u_mask == 0], 0.0)
    np.testing.assert_array_equal(v_star[solver.v_mask == 0], 0.0)
    np.testing.assert_array_equal(w_star[solver.w_mask == 0], 0.0)


def test_diffuse_jit_matches_manual_stencil():
    """Directly exercise the kernel against a hand-rolled central difference."""
    rng = np.random.default_rng(0)
    W = H = D = 5
    u = rng.standard_normal((W + 1, H, D))
    v = rng.standard_normal((W, H + 1, D))
    w = rng.standard_normal((W, H, D + 1))
    u_mask = np.ones_like(u, dtype=np.uint8)
    v_mask = np.ones_like(v, dtype=np.uint8)
    w_mask = np.ones_like(w, dtype=np.uint8)
    u_out = np.empty_like(u)
    v_out = np.empty_like(v)
    w_out = np.empty_like(w)

    nu, dx, dy, dz, dt = 1.2, 0.5, 0.5, 0.5, 0.01
    _diffuse_jit(u, v, w, u_mask, v_mask, w_mask,
                 nu, dx, dy, dz, 0.0, 0.0, 0.0, dt,
                 u_out, v_out, w_out)

    # Interior of the u component with all neighbours in bounds.
    i, j, k = 2, 2, 2
    lap = ((u[i + 1, j, k] - 2 * u[i, j, k] + u[i - 1, j, k]) / dx ** 2
           + (u[i, j + 1, k] - 2 * u[i, j, k] + u[i, j - 1, k]) / dy ** 2
           + (u[i, j, k + 1] - 2 * u[i, j, k] + u[i, j, k - 1]) / dz ** 2)
    assert np.isclose(u_out[i, j, k], u[i, j, k] + dt * nu * lap)


# --------------------------------------------------------------------------- #
# Poisson step
# --------------------------------------------------------------------------- #
def test_divergence_jit_matches_manual():
    rng = np.random.default_rng(1)
    W, H, D = 4, 3, 5
    u = rng.standard_normal((W + 1, H, D))
    v = rng.standard_normal((W, H + 1, D))
    w = rng.standard_normal((W, H, D + 1))
    fluid = np.ones((W, H, D), dtype=np.uint8)
    fluid[2, 1, 3] = 0  # a solid cell -> div forced to 0
    div = np.empty((W, H, D))

    dx, dy, dz = 0.5, 0.25, 2.0
    _divergence_jit(u, v, w, fluid, dx, dy, dz, div)

    i, j, k = 1, 1, 2
    expected = ((u[i + 1, j, k] - u[i, j, k]) / dx
                + (v[i, j + 1, k] - v[i, j, k]) / dy
                + (w[i, j, k + 1] - w[i, j, k]) / dz)
    assert np.isclose(div[i, j, k], expected)
    assert div[2, 1, 3] == 0.0


def test_poisson_system_is_condensed():
    """The Poisson matrix has exactly one row per fluid cell, not per voxel."""
    volume = make_circular_duct(radius=4, length=6)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm)
    solver.create_velocity_arrays()
    solver._build_pressure_poisson_system()

    n_fluid = int(np.count_nonzero(solver.pressure_mask))
    assert solver.poisson_solver.a_sparse_array["row_ptr"].size == n_fluid
    assert solver.poisson_solver.b_array.size == n_fluid
    # Condensed system is far smaller than the full dense voxel grid.
    assert n_fluid < volume.size


def test_poisson_recovers_manufactured_pressure():
    """A p = A @ p_exact must solve back to p_exact (matrix is non-singular)."""
    volume = make_circular_duct(radius=4, length=6)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm)
    solver.create_velocity_arrays()
    solver._build_pressure_poisson_system()

    a_sparse = solver.poisson_solver.a_sparse_array
    n = solver.poisson_solver.b_array.size
    rng = np.random.default_rng(2)
    p_exact = rng.standard_normal(n)

    b = _csr_matvec(a_sparse, p_exact)
    solver.poisson_solver.b_array = b
    x, err, _ = solver.poisson_solver.solve_pcg()
    np.testing.assert_allclose(x, p_exact, rtol=1e-5, atol=1e-6)


def test_poisson_step_converges_and_shapes():
    volume = make_circular_duct(radius=4, length=6)
    vm = VolumeManager(volume)
    solver = StokesSolver(vm, density=2.0, target_error=1e-8)
    solver.create_velocity_arrays()

    # Seed a non-trivial (divergent) velocity field on the active faces.
    solver.w[solver.w_mask == 1] = 0.3
    solver.v[solver.v_mask == 1] = -0.1

    dt = 0.01
    p = solver.poisson_step(solver.u, solver.v, solver.w, dt)

    assert p.shape == volume.shape
    # The reported PCG error is the relative residual; confirm it converged and
    # that A p indeed matches the assembled RHS.
    assert solver.poisson_error <= 1e-6
    residual = _csr_matvec(solver.poisson_solver.a_sparse_array, solver.p_condensed) - solver.rhs
    rel = np.linalg.norm(residual) / np.linalg.norm(solver.rhs)
    assert rel < 1e-5


def test_poisson_step_builds_system_lazily():
    volume = make_circular_duct(radius=3, length=5)
    solver = StokesSolver(VolumeManager(volume))
    solver.create_velocity_arrays()
    assert solver.poisson_solver is None
    solver.poisson_step(solver.u, solver.v, solver.w, dt=0.01)
    assert solver.poisson_solver is not None


def test_poisson_anisotropic_scale_raises():
    volume = np.ones((4, 4, 4), dtype=np.float64)
    vm = VolumeManager(volume, scale=(0.1, 0.2, 0.3))
    solver = StokesSolver(vm)
    solver.create_velocity_arrays()
    with pytest.raises(NotImplementedError):
        solver._build_pressure_poisson_system()


# --------------------------------------------------------------------------- #
# Corrector step
# --------------------------------------------------------------------------- #
def test_pressure_gradient_kernel_matches_manual():
    W, H, D = 5, 5, 5
    rng = np.random.default_rng(4)
    u = rng.standard_normal((W + 1, H, D))
    v = rng.standard_normal((W, H + 1, D))
    w = rng.standard_normal((W, H, D + 1))
    u0, v0, w0 = u.copy(), v.copy(), w.copy()
    pressure = rng.standard_normal((W, H, D))
    u_mask = np.ones_like(u, dtype=np.uint8)
    v_mask = np.ones_like(v, dtype=np.uint8)
    w_mask = np.ones_like(w, dtype=np.uint8)
    u_mask[0] = u_mask[W] = 0  # domain-boundary walls
    v_mask[:, 0] = v_mask[:, H] = 0
    w_mask[:, :, 0] = w_mask[:, :, D] = 0

    coef, dx, dy, dz = 0.3, 0.5, 0.5, 0.5
    _apply_pressure_gradient_jit(u, v, w, pressure,
                                 u_mask, v_mask, w_mask, coef, dx, dy, dz)

    i, j, k = 2, 2, 2
    assert np.isclose(u[i, j, k],
                      u0[i, j, k] - coef * (pressure[i, j, k] - pressure[i - 1, j, k]) / dx)
    assert np.isclose(w[i, j, k],
                      w0[i, j, k] - coef * (pressure[i, j, k] - pressure[i, j, k - 1]) / dz)
    # Walls untouched.
    np.testing.assert_array_equal(u[0], u0[0])
    np.testing.assert_array_equal(w[:, :, D], w0[:, :, D])


def test_corrector_leaves_walls_zero():
    volume = make_circular_duct(radius=4, length=6)
    solver = StokesSolver(VolumeManager(volume))
    solver.create_velocity_arrays()
    pressure = np.ones(volume.shape)  # arbitrary non-zero pressure field

    u_star = solver.u_buf
    v_star = solver.v_buf
    w_star = solver.w_buf
    u_star[solver.u_mask == 1] = 0.5
    v_star[solver.v_mask == 1] = 0.5
    w_star[solver.w_mask == 1] = 0.5

    u_new, v_new, w_new = solver.corrector_step(u_star, v_star, w_star, pressure, dt=0.01)
    np.testing.assert_array_equal(u_new[solver.u_mask == 0], 0.0)
    np.testing.assert_array_equal(v_new[solver.v_mask == 0], 0.0)
    np.testing.assert_array_equal(w_new[solver.w_mask == 0], 0.0)


def test_projection_makes_field_divergence_free():
    """Predictor -> Poisson -> corrector must leave a divergence-free field on
    interior fluid cells (away from the z Dirichlet boundaries)."""
    volume = make_circular_duct(radius=5, length=8)
    solver = StokesSolver(vm := VolumeManager(volume), target_error=1e-11)
    solver.create_velocity_arrays()

    # Seed a divergent velocity field on the active faces.
    rng = np.random.default_rng(5)
    solver.u[solver.u_mask == 1] = rng.standard_normal(int((solver.u_mask == 1).sum()))
    solver.v[solver.v_mask == 1] = rng.standard_normal(int((solver.v_mask == 1).sum()))
    solver.w[solver.w_mask == 1] = rng.standard_normal(int((solver.w_mask == 1).sum()))

    dt = solver._compute_timestep()
    u_star, v_star, w_star = solver.predictor_step(dt)
    p = solver.poisson_step(u_star, v_star, w_star, dt)
    u_new, v_new, w_new = solver.corrector_step(u_star, v_star, w_star, p, dt)

    dx, dy, dz = (float(s) for s in vm.scale[:3])
    div = np.zeros(volume.shape)
    _divergence_jit(u_new, v_new, w_new, solver.pressure_mask, dx, dy, dz, div)

    # Exclude the z=0 / z=max cell layers: their faces are masked as walls but
    # the matrix carries the inlet/outlet Dirichlet term, so divergence there is
    # not expected to vanish until the boundary-condition wiring is added.
    d = volume.shape[2]
    interior = solver.pressure_mask_bool.copy()
    interior[:, :, 0] = False
    interior[:, :, d - 1] = False

    assert np.abs(div[interior]).max() < 1e-6


# --------------------------------------------------------------------------- #
# Circular-duct fixture (for the eventual Hagen-Poiseuille comparison)
# --------------------------------------------------------------------------- #
def test_circular_duct_geometry():
    radius, length = 5, 8
    volume = make_circular_duct(radius, length)

    n = volume.shape[0]
    assert volume.shape == (n, n, length)
    # Cross-section is identical along the flow direction.
    for z in range(1, length):
        np.testing.assert_array_equal(volume[:, :, z], volume[:, :, 0])
    # Center is fluid, corners are solid wall.
    center = n // 2
    assert volume[center, center, 0] == 1.0
    assert volume[0, 0, 0] == 0.0
    # Fluid voxel count per slice is close to the disk area pi r^2.
    area = np.count_nonzero(volume[:, :, 0])
    assert np.isclose(area, np.pi * radius ** 2, rtol=0.15)


def test_circular_duct_hagen_poiseuille():
    """Steady z-velocity in a pressure-driven pipe must match the analytic
    Hagen-Poiseuille parabola:

        w(r) = (G / (4 mu)) * (R^2 - r^2),   G = dp/dz,  mu = rho * nu

    With unit spacing and a unit pressure drop over the duct length L, the
    profile w(r) plotted against r^2 is a straight line of slope -G/(4 mu)
    with G = 1 / L. We verify the profile is parabolic (good linear fit of w
    vs r^2) and that the slope matches the analytic value.
    """
    radius, length = 6, 8
    volume = make_circular_duct(radius, length)
    vm = VolumeManager(volume, scale=1.0)
    solver = StokesSolver(vm, viscosity=1.0, density=1.0,
                          max_iterations=20000, target_error=1e-6)
    result = solver.solve()
    assert result["converged"]

    W, H, D = volume.shape
    k = D // 2  # mid-duct cross-section (fully developed, away from ends)
    w_center = 0.5 * (result["w"][:, :, k] + result["w"][:, :, k + 1])
    fluid = volume[:, :, k] > 0

    c = (W - 1) / 2.0
    yy, xx = np.mgrid[0:W, 0:H]
    r2 = ((xx - c) ** 2 + (yy - c) ** 2)[fluid]
    w = w_center[fluid]

    # Flow is driven in +z and (essentially) purely axial.
    assert w.min() > 0
    assert np.abs(result["u"][:, :, k]).max() < 1e-9
    assert np.abs(result["v"][:, :, k]).max() < 1e-9

    # Least-squares fit w = slope * r^2 + intercept.
    A = np.vstack([r2, np.ones_like(r2)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, w, rcond=None)
    pred = A @ np.array([slope, intercept])
    r_squared = 1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum()

    assert r_squared > 0.99                     # profile is parabolic
    analytic_slope = -(1.0 / length) / (4.0 * 1.0)   # -G/(4 mu), mu = rho*nu = 1
    assert np.isclose(slope, analytic_slope, rtol=0.1)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def test_max_relative_change_jit():
    new = np.array([1.0, 2.0, 5.0, -4.0])
    old = np.array([1.0, 1.0, 5.0, 0.0])
    mask = np.array([1, 1, 1, 0], dtype=np.uint8)  # last entry ignored
    diff, val = _max_relative_change_jit(new, old, mask, threads=1)
    assert np.isclose(diff, 1.0)   # |2-1|; the |-4-0| entry is masked out
    assert np.isclose(val, 5.0)    # max|new| over active entries


def test_apply_velocity_bc_opens_ends():
    volume = make_circular_duct(radius=3, length=5)
    solver = StokesSolver(VolumeManager(volume))
    solver.create_velocity_arrays()
    d = volume.shape[2]
    solver.w[:, :, 1] = 0.7
    solver.w[:, :, d - 1] = 0.4

    solver._apply_velocity_boundary_conditions()

    fluid_in = volume[:, :, 0] > 0
    fluid_out = volume[:, :, d - 1] > 0
    np.testing.assert_array_equal(solver.w[:, :, 0][fluid_in], 0.7)
    np.testing.assert_array_equal(solver.w[:, :, d][fluid_out], 0.4)
    # Solid columns stay closed.
    np.testing.assert_array_equal(solver.w[:, :, 0][~fluid_in], 0.0)


def test_solve_returns_fields_and_diagnostics():
    volume = make_circular_duct(radius=3, length=5)
    solver = StokesSolver(VolumeManager(volume), max_iterations=2000, target_error=1e-5)
    result = solver.solve()

    assert set(result) == {"u", "v", "w", "p", "iterations", "residual", "converged"}
    assert result["u"].shape == (volume.shape[0] + 1, volume.shape[1], volume.shape[2])
    assert result["p"].shape == volume.shape
    assert result["converged"]
    assert result["iterations"] >= 1


def test_invalid_convergence_criterion_raises():
    with pytest.raises(Exception):
        StokesSolver(VolumeManager(np.ones((3, 3, 3))), convergence_criterion="bogus")


def test_invalid_poisson_backend_raises():
    with pytest.raises(Exception):
        StokesSolver(VolumeManager(np.ones((3, 3, 3))), poisson_backend="bogus")


def test_invalid_predictor_raises():
    with pytest.raises(Exception):
        StokesSolver(VolumeManager(np.ones((3, 3, 3))), predictor="bogus")


# --------------------------------------------------------------------------- #
# Implicit (backward-Euler) diffusion predictor  (Phase 2, s10.4 Option A)
# --------------------------------------------------------------------------- #
def test_diffusion_matrix_equals_I_minus_coef_laplacian():
    """The assembled implicit operator M must equal I - coef*L, where L is the
    same 7-point face Laplacian the explicit predictor applies."""
    from pyflowsolver.stokesSolver import (
        _assemble_diffusion_csr, _gather_face_to_condensed, _diffuse_jit)
    from pyflowsolver.multigridSolver import _csr_matvec

    volume = make_circular_duct(radius=5, length=8)
    solver = StokesSolver(VolumeManager(volume), viscosity=1.3)
    solver.create_velocity_arrays()

    coef = 0.37
    mask = solver.w_mask
    val, col, rp, idx, n = _assemble_diffusion_csr(mask, 1.0, 1.0, 1.0, coef)

    rng = np.random.default_rng(0)
    field = np.zeros_like(solver.w)
    field[mask == 1] = rng.standard_normal(int((mask == 1).sum()))
    xc = np.zeros(n)
    _gather_face_to_condensed(field, idx, xc)

    Mx = np.zeros(n)
    _csr_matvec(val, col, rp, xc, Mx)

    # explicit kernel gives field + coef*L(field); so I - coef*L applied = 2f - out
    out = np.zeros_like(field)
    vb = np.zeros_like(solver.v); wb = np.zeros_like(solver.w)
    _diffuse_jit(field, solver.v, solver.w, mask, solver.v_mask, solver.w_mask,
                 1.3, 1.0, 1.0, 1.0, 0, 0, 0, coef / 1.3, out, vb, wb)
    expected = np.zeros(n)
    _gather_face_to_condensed(2 * field - out, idx, expected)

    np.testing.assert_allclose(Mx, expected, rtol=1e-10, atol=1e-12)


def test_implicit_matches_explicit_steady_solution():
    """Incremental implicit predictor must reach the same steady field as the
    explicit predictor (different path to the same discrete steady state)."""
    volume = make_circular_duct(radius=5, length=8)
    fields = {}
    for predictor in ("explicit", "implicit"):
        solver = StokesSolver(VolumeManager(volume.copy()), viscosity=1.0,
                              density=1.0, max_iterations=40000,
                              target_error=1e-6, predictor=predictor)
        fields[predictor] = solver.solve()
        assert fields[predictor]["converged"]

    for key in ("w", "p"):
        a, b = fields["explicit"][key], fields["implicit"][key]
        scale = np.abs(a).max()
        assert np.abs(a - b).max() <= 5e-3 * scale


def test_implicit_far_fewer_iterations():
    """The implicit predictor's outer count is ~size-independent, so on a duct
    big enough for the explicit O((L/dx)^2) growth it must use far fewer steps."""
    volume = make_circular_duct(radius=8, length=12)
    expl = StokesSolver(VolumeManager(volume.copy()), max_iterations=40000,
                        target_error=1e-6).solve()
    impl = StokesSolver(VolumeManager(volume.copy()), max_iterations=40000,
                        target_error=1e-6, predictor="implicit").solve()
    assert impl["converged"] and expl["converged"]
    assert impl["iterations"] < expl["iterations"] // 5


def test_implicit_hagen_poiseuille():
    """The HP parabola regression must hold with the implicit predictor."""
    radius, length = 6, 8
    volume = make_circular_duct(radius, length)
    solver = StokesSolver(VolumeManager(volume), viscosity=1.0, density=1.0,
                          max_iterations=40000, target_error=1e-6,
                          predictor="implicit")
    result = solver.solve()
    assert result["converged"]

    W, H, D = volume.shape
    k = D // 2
    w_center = 0.5 * (result["w"][:, :, k] + result["w"][:, :, k + 1])
    fluid = volume[:, :, k] > 0
    c = (W - 1) / 2.0
    yy, xx = np.mgrid[0:W, 0:H]
    r2 = ((xx - c) ** 2 + (yy - c) ** 2)[fluid]
    w = w_center[fluid]

    A = np.vstack([r2, np.ones_like(r2)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, w, rcond=None)
    pred = A @ np.array([slope, intercept])
    r_squared = 1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum()

    assert r_squared > 0.99
    analytic_slope = -(1.0 / length) / (4.0 * 1.0)
    assert np.isclose(slope, analytic_slope, rtol=0.1)


def test_implicit_channel_does_not_fall_back():
    """On a clean channel the implicit pseudo-transient reaches steady state, so
    the explicit finish is a single confirming step (no real fallback work)."""
    volume = make_circular_duct(radius=6, length=10)
    solver = StokesSolver(VolumeManager(volume), max_iterations=40000,
                          target_error=1e-6, predictor="implicit")
    result = solver.solve()
    assert result["converged"]
    assert solver.implicit_iterations >= 1
    assert not solver.fell_back                       # implicit did the work
    assert result["iterations"] - solver.implicit_iterations <= 2


def test_implicit_hybrid_correct_on_complex_geometry():
    """Safety property: on complex geometry where the implicit pseudo-transient
    stalls, the explicit handoff must still land on the correct steady state
    (the same field the pure-explicit solver reaches). This is what makes the
    implicit predictor safe as a default-off accelerator."""
    import porespy as ps
    import scipy.ndimage as ndi
    im = ps.generators.blobs(shape=(16, 16, 16), porosity=0.5,
                             blobiness=1.0, seed=2)
    lab, _ = ndi.label(im)
    im = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)   # percolating cluster
    volume = im.astype(np.float64)

    expl = StokesSolver(VolumeManager(volume.copy()), max_iterations=40000,
                        target_error=1e-6, poisson_backend="mgpcg").solve()
    isolver = StokesSolver(VolumeManager(volume.copy()), max_iterations=40000,
                           target_error=1e-6, predictor="implicit",
                           poisson_backend="mgpcg")
    impl = isolver.solve()

    assert impl["converged"]
    # Implicit stalls here, so the explicit finish does the real work ...
    assert isolver.fell_back
    # ... but the final field must match the pure-explicit solution.
    scale = np.abs(expl["w"]).max()
    assert np.abs(expl["w"] - impl["w"]).max() <= 1e-2 * scale
    # And the momentum residual confirms a genuine steady state (not a stall).
    assert isolver._momentum_residual() < 1e-4


def test_implicit_irregular_boundary_raises():
    from pyflowsolver.constants import PORE, INLET, OUTLET
    bv = np.zeros((6, 6, 6), dtype=np.uint8)
    bv[1:-1, :, 1:-1] = PORE
    bv[1:-1, 0, 1:-1] = INLET
    bv[1:-1, -1, 1:-1] = OUTLET
    vm = VolumeManager((bv >= 1) * 1.0, boundary_volume=bv)
    with pytest.raises(NotImplementedError):
        StokesSolver(vm, predictor="implicit")


# --------------------------------------------------------------------------- #
# Multigrid pressure-Poisson backend
# --------------------------------------------------------------------------- #
def test_mgpcg_backend_builds_multigrid_solver():
    from pyflowsolver.multigridSolver import MultigridSolver
    volume = make_circular_duct(radius=4, length=6)
    solver = StokesSolver(VolumeManager(volume), poisson_backend="mgpcg")
    solver.create_velocity_arrays()
    solver._build_pressure_poisson_system()
    assert isinstance(solver.poisson_solver, MultigridSolver)


def test_mgpcg_matches_pcg_steady_solution():
    """The multigrid backend must reach the same steady Stokes field as the
    baseline diagonal-PCG backend (same equations, faster Poisson solve)."""
    volume = make_circular_duct(radius=5, length=8)
    fields = {}
    for backend in ("pcg", "mgpcg"):
        solver = StokesSolver(VolumeManager(volume.copy()), viscosity=1.0,
                              density=1.0, max_iterations=20000,
                              target_error=1e-6, poisson_backend=backend)
        fields[backend] = solver.solve()
        assert fields[backend]["converged"]

    # Axial velocity and pressure are the physical unknowns; both must agree.
    for key in ("w", "p"):
        a, b = fields["pcg"][key], fields["mgpcg"][key]
        scale = np.abs(a).max()
        assert np.abs(a - b).max() <= 1e-5 * scale


def test_mgpcg_hagen_poiseuille():
    """The HP parabola regression must hold with the multigrid backend too."""
    radius, length = 6, 8
    volume = make_circular_duct(radius, length)
    solver = StokesSolver(VolumeManager(volume), viscosity=1.0, density=1.0,
                          max_iterations=20000, target_error=1e-6,
                          poisson_backend="mgpcg")
    result = solver.solve()
    assert result["converged"]

    W, H, D = volume.shape
    k = D // 2
    w_center = 0.5 * (result["w"][:, :, k] + result["w"][:, :, k + 1])
    fluid = volume[:, :, k] > 0
    c = (W - 1) / 2.0
    yy, xx = np.mgrid[0:W, 0:H]
    r2 = ((xx - c) ** 2 + (yy - c) ** 2)[fluid]
    w = w_center[fluid]

    A = np.vstack([r2, np.ones_like(r2)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, w, rcond=None)
    pred = A @ np.array([slope, intercept])
    r_squared = 1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum()

    assert r_squared > 0.99
    analytic_slope = -(1.0 / length) / (4.0 * 1.0)
    assert np.isclose(slope, analytic_slope, rtol=0.1)


def test_residual_criterion_is_step_over_dt():
    """The 'residual' metric equals the 'step' metric divided by dt."""
    volume = make_circular_duct(radius=3, length=5)
    solver = StokesSolver(VolumeManager(volume))
    solver.create_velocity_arrays()

    rng = np.random.default_rng(7)
    new = (rng.standard_normal(solver.u.shape),
           rng.standard_normal(solver.v.shape),
           rng.standard_normal(solver.w.shape))
    old = (rng.standard_normal(solver.u.shape),
           rng.standard_normal(solver.v.shape),
           rng.standard_normal(solver.w.shape))
    dt = 0.023

    solver.params["convergence_criterion"] = "step"
    step = solver._velocity_residual(new, old, dt)
    solver.params["convergence_criterion"] = "residual"
    residual = solver._velocity_residual(new, old, dt)
    assert np.isclose(residual, step / dt)


def test_solve_converges_with_residual_criterion():
    volume = make_circular_duct(radius=4, length=6)
    solver = StokesSolver(VolumeManager(volume), convergence_criterion="residual",
                          target_error=1e-3, max_iterations=20000)
    result = solver.solve()
    assert result["converged"]
    # A converged parabolic profile should still develop under either criterion.
    W, H, D = volume.shape
    wc = 0.5 * (result["w"][:, :, D // 2] + result["w"][:, :, D // 2 + 1])
    assert wc[volume[:, :, D // 2] > 0].min() > 0
