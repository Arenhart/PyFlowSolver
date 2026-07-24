"""Tests for BrinkmanSolver (Darcy-Stokes-Brinkman on a MAC grid).

Physics anchors:
  - K -> inf   => the drag term vanishes, so the solver must reproduce the pure
    Stokes solver bit-for-bit.
  - decreasing K  => stronger drag => less flow (monotone effective permeability).
  - K small (screening length sqrt(K) << duct width) => Darcy limit, so the
    effective permeability of a duct filled with uniform K approaches K itself.
"""

import numpy as np
import pytest

from pyflowsolver.stokesSolver import StokesSolver
from pyflowsolver.brinkmanSolver import BrinkmanSolver
from pyflowsolver.tubeBundle import ConstantTubeDistribution


def _open_duct(w=6, h=6, d=10):
    return np.ones((w, h, d), dtype=np.float64)


@pytest.mark.parametrize("predictor", ["explicit", "implicit"])
def test_infinite_permeability_recovers_stokes(predictor):
    vol = _open_duct()
    params = dict(scale=1.0, fast_laplacian_guess=False, predictor=predictor,
                  max_iterations=300, target_error=1e-7)
    stokes = StokesSolver(vol, **params).solve()
    brink = BrinkmanSolver(
        vol, permeability=np.full(vol.shape, np.inf), **params).solve()
    for key in ("u", "v", "w", "p"):
        np.testing.assert_allclose(brink[key], stokes[key], rtol=1e-9, atol=1e-12)


def _keff(vol, K, **over):
    params = dict(scale=1.0, fast_laplacian_guess=False, predictor="implicit",
                  max_iterations=4000, target_error=1e-9)
    params.update(over)
    solver = BrinkmanSolver(vol, permeability=np.full(vol.shape, K, dtype=np.float64),
                            **params)
    solver.solve()
    return solver.effective_permeability()


def test_effective_permeability_monotonic_and_darcy_limit():
    vol = _open_duct(10, 10, 8)
    k_lo = _keff(vol, 0.01)
    k_mid = _keff(vol, 0.5)
    k_inf = _keff(vol, np.inf)          # Stokes (open-duct) limit
    # More permeable medium conducts more.
    assert k_lo < k_mid < k_inf
    # Darcy limit: with sqrt(K)=0.1 << half-width 5, K_eff ~ K.
    assert k_lo == pytest.approx(0.01, rel=0.15)


def test_finite_drag_reduces_flow_below_stokes():
    vol = _open_duct()
    st = BrinkmanSolver(vol, permeability=np.full(vol.shape, np.inf),
                        scale=1.0, fast_laplacian_guess=False, predictor="implicit",
                        max_iterations=2000, target_error=1e-8)
    st.solve()
    br = BrinkmanSolver(vol, permeability=np.full(vol.shape, 0.05),
                        scale=1.0, fast_laplacian_guess=False, predictor="implicit",
                        max_iterations=2000, target_error=1e-8)
    br.solve()
    assert br.effective_permeability() < st.effective_permeability()


def test_porosity_map_path_runs():
    # Build K internally from a (uniform subresolution) porosity map.
    por = np.full((8, 8, 8), 0.5)
    br = BrinkmanSolver(
        porosity_map=por, scale=1.0,
        distributions={2: ConstantTubeDistribution(0.25)},
        fast_laplacian_guess=False, predictor="implicit",
        max_iterations=2000, target_error=1e-8)
    res = br.solve()
    assert res["converged"]
    keff = br.effective_permeability()
    assert np.isfinite(keff) and keff > 0.0


def test_requires_exactly_one_of_permeability_or_porosity():
    with pytest.raises(ValueError):
        BrinkmanSolver(_open_duct())                       # neither
    with pytest.raises(ValueError):
        BrinkmanSolver(_open_duct(), permeability=np.ones((6, 6, 10)),
                       porosity_map=np.ones((6, 6, 10)))    # both
