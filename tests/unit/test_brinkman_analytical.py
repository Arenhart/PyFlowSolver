"""Analytical / semi-analytical validation of the Brinkman solver.

Reference: fully-developed flow in a rectangular duct filled with a uniform
permeability K obeys  nu*lap(u) - (nu/K)*u + G/rho = 0  with no-slip walls, whose
mean gives the closed-form effective permeability

    K_eff(K) = (64/pi^4) * sum_{m,n odd} 1 / [ m^2 n^2 ((m*pi/Lx)^2 + (n*pi/Ly)^2 + 1/K) ]

Two discretisation facts shape the tests:
  * the solver's kernels impose no-slip with a ghost=0 half a cell beyond the
    first/last sample, so the effective continuum domain is (w+1)*dx x (h+1)*dy;
  * absolute K_eff carries O(1/w) area/length convention offsets, but the RATIO
    K_eff(K)/K_eff(inf) cancels them -- so the ratio is compared to the series,
    leaving only the genuine O(dx^2) cross-sectional error (smaller for weaker
    drag / better-resolved boundary layers).
"""

import numpy as np
import pytest

from pyflowsolver.brinkmanSolver import BrinkmanSolver
from pyflowsolver.tubeBundle import (
    ConstantTubeDistribution,
    TruncatedGaussianTubeDistribution,
    darcy_permeability,
)

_OPEN = 1.0e12   # ~infinite permeability (Stokes limit) without literal inf


def brinkman_duct_keff(K, Lx, Ly, n_modes=160):
    """Closed-form K_eff for a rectangular duct uniformly filled with perm. K."""
    m = np.arange(1, n_modes, 2)[:, None]
    n = np.arange(1, n_modes, 2)[None, :]
    km = (m * np.pi / Lx) ** 2
    kn = (n * np.pi / Ly) ** 2
    terms = 1.0 / (m ** 2 * n ** 2 * (km + kn + 1.0 / K))
    return (64.0 / np.pi ** 4) * float(terms.sum())


def _solve_keff(permeability, w, h, d, predictor="implicit", **over):
    vol = np.ones((w, h, d), dtype=np.float64)
    if np.isscalar(permeability):
        permeability = np.full(vol.shape, permeability, dtype=np.float64)
    params = dict(scale=1.0, fast_laplacian_guess=False, predictor=predictor,
                  max_iterations=8000, target_error=1e-10)
    params.update(over)
    solver = BrinkmanSolver(vol, permeability=permeability, **params)
    solver.solve()
    return solver.effective_permeability()


def _analytic_ratio(K, w, h, n_modes=160):
    """Series K_eff(K)/K_eff(open) using the effective (w+1)x(h+1) domain."""
    Lx, Ly = w + 1.0, h + 1.0
    return (brinkman_duct_keff(K, Lx, Ly, n_modes)
            / brinkman_duct_keff(_OPEN, Lx, Ly, n_modes))


# --------------------------------------------------------------------------- #
# Uniform-K duct: solver ratio vs the analytical Brinkman-duct series
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [0.5, 1.0])
def test_uniform_duct_ratio_matches_series(K):
    w = h = 20
    d = 14
    k_open = _solve_keff(_OPEN, w, h, d)
    k_num = _solve_keff(K, w, h, d)
    ratio_num = k_num / k_open
    ratio_an = _analytic_ratio(K, w, h)
    assert ratio_num == pytest.approx(ratio_an, rel=0.06)


def test_bundle_of_tubes_duct_single_radius_and_gaussian():
    # End-to-end: porosity + radius distribution -> bundle K -> Brinkman duct.
    # The bundle K is small (physical sub-voxel radii => sharp boundary layer),
    # so the ratio tolerance is looser than the well-resolved direct-K test.
    w = h = 24
    d = 14
    phi = 0.8
    k_open = _solve_keff(_OPEN, w, h, d)
    for dist in (ConstantTubeDistribution(0.4),
                 TruncatedGaussianTubeDistribution(0.35, 0.08, 0.15, 0.5)):
        K = darcy_permeability(dist, phi)                      # analytical bundle K
        por = np.full((w, h, d), phi)
        solver = BrinkmanSolver(
            porosity_map=por, scale=1.0, distributions={2: dist},
            fast_laplacian_guess=False, predictor="implicit",
            max_iterations=8000, target_error=1e-10)
        solver.solve()
        ratio_num = solver.effective_permeability() / k_open
        ratio_an = _analytic_ratio(K, w, h)
        assert ratio_num == pytest.approx(ratio_an, rel=0.12)


def test_duct_ratio_converges_under_refinement():
    # The ratio error vs the analytical series must shrink as the duct is
    # refined (O(dx^2) cross-sectional discretisation of the boundary layer).
    K = 0.3
    errs = []
    for w in (12, 24):
        d = max(8, w // 2)
        ratio_num = _solve_keff(K, w, w, d) / _solve_keff(_OPEN, w, w, d)
        errs.append(abs(ratio_num - _analytic_ratio(K, w, w)) / _analytic_ratio(K, w, w))
    assert errs[1] < errs[0]         # finer grid is closer
    assert errs[1] < 0.06


def test_explicit_and_implicit_agree_on_brinkman_duct():
    # The two drag implementations (forward-Euler vs multigrid backward-Euler)
    # must reach the same steady effective permeability.
    w = h = 16
    d = 12
    K = 0.5
    k_imp = _solve_keff(K, w, h, d, predictor="implicit")
    k_exp = _solve_keff(K, w, h, d, predictor="explicit")
    assert k_exp == pytest.approx(k_imp, rel=0.02)


# --------------------------------------------------------------------------- #
# Multi-region composition. In the Darcy limit each subresolution region
# conducts at ~its bulk K (thin boundary layers), so a heterogeneous K field
# must compose by the series (harmonic) / parallel (arithmetic) rules. We
# compare against the solver's OWN single-region K_eff, which cancels the duct
# convention factor (both carry the same offset), isolating the composition.
# --------------------------------------------------------------------------- #
_W = _H = 14
_D = 16
_KA, _KB = 0.02, 0.006          # both small => Darcy limit, exact composition


def test_series_darcy_limit_is_harmonic():
    dA = _D // 2
    dB = _D - dA
    s_a = _solve_keff(_KA, _W, _H, _D)
    s_b = _solve_keff(_KB, _W, _H, _D)
    Kf = np.full((_W, _H, _D), _KA)
    Kf[:, :, dA:] = _KB                      # two slabs stacked along the flow (z)
    s_series = _solve_keff(Kf, _W, _H, _D)
    expected = (dA + dB) / (dA / s_a + dB / s_b)     # length-weighted harmonic
    assert s_series == pytest.approx(expected, rel=0.03)


def test_parallel_darcy_limit_is_arithmetic():
    wA = _W // 2
    wB = _W - wA
    s_a = _solve_keff(_KA, _W, _H, _D)
    s_b = _solve_keff(_KB, _W, _H, _D)
    Kf = np.full((_W, _H, _D), _KA)
    Kf[wA:, :, :] = _KB                      # two slabs side by side (across x)
    s_par = _solve_keff(Kf, _W, _H, _D)
    expected = (wA * s_a + wB * s_b) / _W            # area-weighted arithmetic
    assert s_par == pytest.approx(expected, rel=0.03)


def test_series_subres_plus_open_sanity():
    # Subresolution + resolved (open) in series: only semi-analytical (the
    # Brinkman<->Poiseuille profile transition at the interface is not captured
    # by the harmonic rule), so a loose, bracketed sanity check.
    dA = _D // 2
    dB = _D - dA
    s_sub = _solve_keff(_KA, _W, _H, _D)
    s_open = _solve_keff(_OPEN, _W, _H, _D)
    Kf = np.full((_W, _H, _D), _KA)
    Kf[:, :, dA:] = _OPEN
    s_series = _solve_keff(Kf, _W, _H, _D)
    expected = (dA + dB) / (dA / s_sub + dB / s_open)
    assert s_sub < s_series < s_open                 # bracketed by the two media
    assert s_series == pytest.approx(expected, rel=0.30)


def test_parallel_subres_plus_open_sanity():
    # Subresolution + open side by side: NO clean closed form. The area-weighted
    # rule fails here because an open region's permeability is geometry-dependent
    # (Poiseuille K ~ width^2), so a half-width open channel carries far less than
    # the full-width s_open; plus interface shear couples the two. We assert only
    # the robust invariants: the result is bracketed by the two media and exceeds
    # the pure-subresolution duct (the open channel raises the throughput).
    s_sub = _solve_keff(_KA, _W, _H, _D)
    s_open = _solve_keff(_OPEN, _W, _H, _D)
    Kf = np.full((_W, _H, _D), _KA)
    Kf[_W // 2:, :, :] = _OPEN
    s_par = _solve_keff(Kf, _W, _H, _D)
    assert s_sub < s_par < s_open
    assert s_par > 5.0 * s_sub                       # the open half dominates flow
