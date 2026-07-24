"""Tests for MultiscaleVolumeManager and the tube-bundle distributions.

Reference geometry: BIN_Bentheimer000.nc (250^3, int8, 0 = pore / 1 = solid),
block-mean downscaled by 10 to a 25^3 *porosity map* WITHOUT thresholding -- the
averaged pore fraction of each 10^3 block IS the subresolution porosity. This
yields a genuinely multiscale image (fully-solid, fully-open, and intermediate
subresolution voxels all present).
"""

import os

import numpy as np
import pytest
from scipy.io import netcdf_file

from pyflowsolver.multiscaleVolumeManager import (
    MultiscaleVolumeManager,
    REGION_SOLID,
    REGION_RESOLVED,
    REGION_SUBRES_FIRST,
)
from pyflowsolver.tubeBundle import (
    ConstantTubeDistribution,
    TruncatedGaussianTubeDistribution,
    TruncatedLognormalTubeDistribution,
    EmpiricalTubeRadiusDistribution,
    darcy_permeability,
    subresolution_conductance_function,
)

_BASE = os.path.join(os.path.dirname(__file__),
                     "resources", "numerical_solved", "netcdf")
_REF = os.path.join(_BASE, "BIN_Bentheimer000.nc")
_DOWNSCALE = 10


def _load_porosity_map():
    """Block-mean the reference binary geometry into a [0..1] porosity map."""
    bf = netcdf_file(_REF, "r", mmap=False)
    geom = np.asarray(bf.variables["__xarray_dataarray_variable__"].data)
    bf.close()
    pore = (geom == 0).astype(np.float64)          # 0 = pore -> pore fraction
    f, (W, H, D) = _DOWNSCALE, pore.shape
    return pore.reshape(W // f, f, H // f, f, D // f, f).mean(axis=(1, 3, 5))


# Loaded once; treated as read-only by the tests.
POROSITY = _load_porosity_map()

# Sub-voxel tube radii (coarse voxel scale == 1.0), one per distribution kind.
_DISTRIBUTIONS = {
    "constant": ConstantTubeDistribution(0.25),
    "trunc_gaussian": TruncatedGaussianTubeDistribution(
        mean=0.25, sigma=0.10, r_min=0.05, r_max=0.45),
    "trunc_lognormal": TruncatedLognormalTubeDistribution(
        mean=np.log(0.25), sigma=0.5, r_min=0.05, r_max=0.45),
}


# --------------------------------------------------------------------------- #
# Sanity of the downscaled reference porosity map
# --------------------------------------------------------------------------- #
def test_porosity_map_is_multiscale():
    assert POROSITY.shape == (25, 25, 25)
    assert POROSITY.min() == 0.0 and POROSITY.max() == 1.0
    assert (POROSITY == 0).any()                    # fully solid blocks
    assert (POROSITY == 1).any()                    # fully open blocks
    assert ((POROSITY > 0) & (POROSITY < 1)).any()  # subresolution blocks


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def test_constant_distribution():
    d = ConstantTubeDistribution(0.3)
    assert d.min() == 0.3 and d.max() == 0.3
    assert d.cdf(0.2, 0.4) == pytest.approx(1.0)
    assert d.cdf(0.31, 0.4) == pytest.approx(0.0)
    assert d.sample() == pytest.approx(0.3)
    assert np.allclose(d.sample(5), 0.3)


def test_empirical_distribution_contract():
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.1, 0.4, size=5000)
    d = EmpiricalTubeRadiusDistribution(samples, seed=1)
    assert d.min() == pytest.approx(samples.min())
    assert d.max() == pytest.approx(samples.max())
    assert d.cdf(d.min(), d.max()) == pytest.approx(1.0)
    assert 0.0 < d.cdf(0.2, 0.3) < 1.0
    draws = np.asarray(d.sample(1000, rng=np.random.default_rng(2)))
    assert draws.min() >= d.min() and draws.max() <= d.max()
    # cdf half-way through a uniform sample set is ~0.5.
    assert d.cdf(d.min(), 0.25) == pytest.approx(0.5, abs=0.05)


def test_gaussian_fit_from_samples_recovers_params():
    rng = np.random.default_rng(0)
    s = rng.normal(5.0, 1.0, size=200000)        # stays positive => no bias
    d = TruncatedGaussianTubeDistribution.from_samples(s)
    assert d.mean == pytest.approx(5.0, rel=0.02)
    assert d.sigma == pytest.approx(1.0, rel=0.02)
    assert d.min() == pytest.approx(s.min()) and d.max() == pytest.approx(s.max())


def test_gaussian_fit_from_density_recovers_params_and_truncates():
    r = np.linspace(1.0, 9.0, 400)
    density = np.exp(-0.5 * ((r - 5.0) / 1.0) ** 2)     # unnormalised Gaussian pdf
    d = TruncatedGaussianTubeDistribution.from_density(r, density, r_min=1.0, r_max=5.0)
    assert d.mean == pytest.approx(5.0, rel=0.02)
    assert d.sigma == pytest.approx(1.0, rel=0.03)
    assert d.max() == 5.0                                 # resolution truncation honoured
    assert d.cdf(d.min(), d.max()) == pytest.approx(1.0)


def test_lognormal_fit_from_samples_recovers_logspace_params():
    rng = np.random.default_rng(1)
    s = rng.lognormal(mean=np.log(3.0), sigma=0.4, size=200000)
    d = TruncatedLognormalTubeDistribution.from_samples(s, r_min=0.3, r_max=30.0)
    assert np.exp(d.mean) == pytest.approx(3.0, rel=0.02)   # log-space location
    assert d.sigma == pytest.approx(0.4, rel=0.03)


def test_fit_constructors_reject_degenerate_input():
    with pytest.raises(ValueError):
        TruncatedGaussianTubeDistribution.from_samples([2.0])            # too few
    with pytest.raises(ValueError):
        TruncatedGaussianTubeDistribution.from_density([1.0, 2.0], [1.0])  # length mismatch


@pytest.mark.parametrize("dist", [
    TruncatedGaussianTubeDistribution(0.25, 0.10, 0.05, 0.45),
    TruncatedLognormalTubeDistribution(np.log(0.25), 0.5, 0.05, 0.45),
])
def test_truncated_distribution_contract(dist):
    assert dist.min() == 0.05 and dist.max() == 0.45
    # Full-range probability mass integrates to 1.
    assert dist.cdf(dist.min(), dist.max()) == pytest.approx(1.0, abs=1e-9)
    # Monotone: a sub-range carries less mass than the full range.
    assert 0.0 < dist.cdf(0.20, 0.30) < 1.0
    # Samples stay inside the truncation window.
    s = np.asarray(dist.sample(2000))
    assert s.min() >= 0.05 - 1e-9 and s.max() <= 0.45 + 1e-9


def test_darcy_permeability_scaling():
    # k ~ phi * <r^2> / 8: doubling radius quadruples k; scales linearly in phi.
    k_small = darcy_permeability(ConstantTubeDistribution(0.1), 0.5)
    k_big = darcy_permeability(ConstantTubeDistribution(0.2), 0.5)
    assert k_big / k_small == pytest.approx(4.0, rel=1e-6)
    k_lo = darcy_permeability(ConstantTubeDistribution(0.1), 0.25)
    assert k_small / k_lo == pytest.approx(2.0, rel=1e-6)
    assert darcy_permeability(ConstantTubeDistribution(0.1), 0.0) == 0.0


def test_subresolution_conductance_function():
    f = subresolution_conductance_function(ConstantTubeDistribution(0.2))
    arr = np.array([0, 10, 50, 99, 100], dtype=np.uint8)
    out = np.asarray(f(arr))
    assert out.shape == arr.shape          # vectorized, shape-preserving
    assert out[0] == 0.0                    # 0 -> 0 (never pollutes solid/open)
    assert np.all(out >= 0.0)
    assert np.all(np.diff(out[:4]) >= 0.0)  # non-decreasing in porosity percent


# --------------------------------------------------------------------------- #
# MultiscaleVolumeManager on the downscaled Bentheimer image
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", list(_DISTRIBUTIONS))
def test_build_conductivity_map(kind):
    dist = _DISTRIBUTIONS[kind]
    mvm = MultiscaleVolumeManager(
        POROSITY, scale=1.0,
        distributions={REGION_SUBRES_FIRST: dist})

    cond = mvm.volume
    assert cond.shape == POROSITY.shape
    assert cond.dtype == np.float64
    assert np.all(np.isfinite(cond))
    assert np.all(cond >= 0.0)
    # Solid voxels never conduct.
    assert np.all(cond[POROSITY == 0.0] == 0.0)
    # Subresolution voxels contribute positive conductance somewhere.
    subres = (POROSITY > 0.0) & (POROSITY < 1.0)
    assert np.any(cond[subres] > 0.0)
    assert cond.max() > 0.0


def test_labelmap_autogenerated_to_spec():
    mvm = MultiscaleVolumeManager(
        POROSITY, scale=1.0,
        distributions={REGION_SUBRES_FIRST: _DISTRIBUTIONS["constant"]})
    labels = mvm.labelmap
    assert set(int(v) for v in np.unique(labels)) <= {
        REGION_SOLID, REGION_RESOLVED, REGION_SUBRES_FIRST}
    assert np.all(labels[POROSITY == 0.0] == REGION_SOLID)
    assert np.all(labels[POROSITY == 1.0] == REGION_RESOLVED)
    assert np.all(labels[(POROSITY > 0.0) & (POROSITY < 1.0)] == REGION_SUBRES_FIRST)


def test_nonzeros_consistency_after_ordering_fix():
    # The ordering fix requires the condensed unknown set (computed by the base
    # __init__) to match the FINAL conductivity field, not the raw porosity.
    mvm = MultiscaleVolumeManager(
        POROSITY, scale=1.0,
        distributions={REGION_SUBRES_FIRST: _DISTRIBUTIONS["constant"]})
    assert mvm.nonzeros == int((mvm.volume > 0.0).sum())
    # ravel scatters exactly `nonzeros` values back (gated on volume > 0).
    sol = mvm.ravel_sparse_solution(
        np.arange(1, mvm.nonzeros + 1, dtype=np.float64))
    assert int((sol > 0.0).sum()) == mvm.nonzeros


def test_porosity_scale_normalization_equivalence():
    # A [0..100] input must be treated identically to [0..1]. The subresolution
    # conductance now uses the exact (float) porosity, so the [0..100] path picks
    # up the ~1e-16 noise of the /100 round-trip -- hence allclose, not exact.
    dist = _DISTRIBUTIONS["constant"]
    mvm01 = MultiscaleVolumeManager(
        POROSITY, scale=1.0, distributions={REGION_SUBRES_FIRST: dist})
    mvm100 = MultiscaleVolumeManager(
        POROSITY * 100.0, scale=1.0, distributions={REGION_SUBRES_FIRST: dist})
    np.testing.assert_allclose(mvm01.porosity_map, mvm100.porosity_map, atol=1e-12)
    np.testing.assert_array_equal(mvm01.labelmap, mvm100.labelmap)
    np.testing.assert_allclose(mvm01.volume, mvm100.volume, rtol=1e-9)


@pytest.mark.parametrize("model", ["homogeneous", "sampled"])
def test_conductance_and_K_share_subresolution_model(model):
    # Both the fast-Laplacian conductance field (self.volume) and the Brinkman K
    # field (get_permeability_field) must route subresolution voxels through the
    # SAME model + seed. On a fully-connected uniform-porosity block (no open,
    # no solid, nothing filtered) the two fields must coincide voxel-for-voxel.
    por = _uniform_porosity((16, 16, 16), 0.5)
    dist = TruncatedGaussianTubeDistribution(0.25, 0.10, 0.05, 0.45)
    mvm = MultiscaleVolumeManager(
        por, scale=1.0, distributions={REGION_SUBRES_FIRST: dist},
        subresolution_model=model, subresolution_seed=5)
    np.testing.assert_allclose(mvm.volume, mvm.get_permeability_field(), rtol=1e-9)


def test_missing_distribution_raises():
    with pytest.raises(ValueError):
        MultiscaleVolumeManager(POROSITY, scale=1.0, distributions={})


# --------------------------------------------------------------------------- #
# get_permeability_field (homogeneous bundle-of-tubes K)
# --------------------------------------------------------------------------- #
def test_darcy_permeability_is_flow_weighted_r4_over_r2():
    # Homogeneous bundle: K(phi) = (1/8) * phi * <r^4>/<r^2> (area/flow-weighted,
    # the r^4 Hagen-Poiseuille weighting). Cross-check the cdf-based moment
    # integral against an INDEPENDENT Monte-Carlo estimate from the sampling path.
    dist = TruncatedLognormalTubeDistribution(np.log(0.25), 0.6, 0.05, 0.6, seed=7)
    r = np.asarray(dist.sample(300000))
    expected = (1.0 / 8.0) * (np.mean(r ** 4) / np.mean(r ** 2))  # phi = 1
    assert darcy_permeability(dist, 1.0) == pytest.approx(expected, rel=0.03)
    # For a constant radius the ratio collapses to r^2.
    assert darcy_permeability(ConstantTubeDistribution(0.2), 1.0) == pytest.approx(
        (1.0 / 8.0) * 0.2 ** 2, rel=1e-9)


@pytest.mark.parametrize("kind", list(_DISTRIBUTIONS))
def test_get_permeability_field_regions(kind):
    dist = _DISTRIBUTIONS[kind]
    mvm = MultiscaleVolumeManager(
        POROSITY, scale=1.0, distributions={REGION_SUBRES_FIRST: dist})
    K = mvm.get_permeability_field()
    assert K.shape == POROSITY.shape
    assert np.all(np.isinf(K[POROSITY >= 1.0]))   # open -> infinite K (drag -> 0)
    assert np.all(K[POROSITY == 0.0] == 0.0)       # solid -> 0
    subres = (POROSITY > 0.0) & (POROSITY < 1.0)
    assert np.all(np.isfinite(K[subres]))
    assert np.all(K[subres] > 0.0)


def test_get_permeability_field_linear_in_porosity():
    # Homogeneous model: K = porosity * (1/8)<r^4>/<r^2>, exactly linear in the
    # local porosity with a per-region distribution factor.
    dist = _DISTRIBUTIONS["constant"]
    mvm = MultiscaleVolumeManager(
        POROSITY, scale=1.0, distributions={REGION_SUBRES_FIRST: dist})
    K = mvm.get_permeability_field()
    subres = (POROSITY > 0.0) & (POROSITY < 1.0)
    factor = darcy_permeability(dist, 1.0)
    np.testing.assert_allclose(K[subres], POROSITY[subres] * factor, rtol=1e-12)


def test_get_permeability_field_multiregion_uses_per_region_distribution():
    por = np.full((3, 3, 4), 0.4)
    labels = np.full((3, 3, 4), REGION_SUBRES_FIRST, dtype=np.int32)
    labels[..., 2:] = REGION_SUBRES_FIRST + 1          # a second subres region
    d2 = ConstantTubeDistribution(0.1)
    d3 = ConstantTubeDistribution(0.3)
    mvm = MultiscaleVolumeManager(
        por, scale=1.0, labelmap=labels,
        distributions={REGION_SUBRES_FIRST: d2, REGION_SUBRES_FIRST + 1: d3})
    K = mvm.get_permeability_field()
    np.testing.assert_allclose(
        K[labels == REGION_SUBRES_FIRST], 0.4 * darcy_permeability(d2, 1.0), rtol=1e-12)
    np.testing.assert_allclose(
        K[labels == REGION_SUBRES_FIRST + 1], 0.4 * darcy_permeability(d3, 1.0), rtol=1e-12)


def test_unknown_subresolution_model_raises():
    with pytest.raises(ValueError):
        MultiscaleVolumeManager(
            POROSITY, scale=1.0,
            distributions={REGION_SUBRES_FIRST: _DISTRIBUTIONS["constant"]},
            subresolution_model="does_not_exist")


# --------------------------------------------------------------------------- #
# Finite-bundle sampling subresolution model (near-resolution)
# --------------------------------------------------------------------------- #
def _uniform_porosity(shape, phi):
    return np.full(shape, phi, dtype=np.float64)


def test_sample_accepts_external_rng():
    d = TruncatedGaussianTubeDistribution(0.25, 0.10, 0.05, 0.45)
    a = np.asarray(d.sample(50, rng=np.random.default_rng(0)))
    b = np.asarray(d.sample(50, rng=np.random.default_rng(0)))
    np.testing.assert_array_equal(a, b)            # rng makes draws reproducible
    assert a.min() >= 0.05 - 1e-9 and a.max() <= 0.45 + 1e-9


def test_sampled_model_reproducible_given_seed():
    por = _uniform_porosity((16, 16, 16), 0.5)
    dist = TruncatedGaussianTubeDistribution(0.25, 0.10, 0.05, 0.45)
    kw = dict(scale=1.0, distributions={REGION_SUBRES_FIRST: dist},
              subresolution_model="sampled", subresolution_seed=3)
    K1 = MultiscaleVolumeManager(por, **kw).get_permeability_field()
    K2 = MultiscaleVolumeManager(por, **kw).get_permeability_field()
    np.testing.assert_array_equal(K1, K2)


def test_sampled_model_seed_changes_realization():
    por = _uniform_porosity((16, 16, 16), 0.5)
    dist = TruncatedGaussianTubeDistribution(0.25, 0.10, 0.05, 0.45)
    base = dict(scale=1.0, distributions={REGION_SUBRES_FIRST: dist},
                subresolution_model="sampled")
    K0 = MultiscaleVolumeManager(por, subresolution_seed=0, **base).get_permeability_field()
    K1 = MultiscaleVolumeManager(por, subresolution_seed=1, **base).get_permeability_field()
    assert not np.array_equal(K0, K1)


def test_sampled_model_introduces_heterogeneity():
    # Uniform porosity + a non-degenerate distribution + few tubes/voxel: the
    # sampled K varies voxel-to-voxel, whereas the homogeneous model is constant.
    por = _uniform_porosity((20, 20, 20), 0.5)
    dist = TruncatedGaussianTubeDistribution(0.25, 0.10, 0.05, 0.45)
    common = dict(scale=1.0, distributions={REGION_SUBRES_FIRST: dist})
    K_s = MultiscaleVolumeManager(
        por, subresolution_model="sampled", subresolution_seed=0, **common
    ).get_permeability_field()
    K_h = MultiscaleVolumeManager(
        por, subresolution_model="homogeneous", **common
    ).get_permeability_field()
    # Homogeneous is constant over uniform porosity (CV ~ 0 to fp precision);
    # the sampled field is genuinely heterogeneous (few tubes/voxel).
    assert K_h.std() / K_h.mean() < 1e-12
    assert K_s.std() / K_s.mean() > 0.05


def test_sampled_model_mean_matches_homogeneous():
    # Calibration: E[K] of the finite bundle equals the homogeneous K
    # phi*(1/8)<r^4>/<r^2> (up to integer-tube rounding). Checked as a spatial
    # mean over a large uniform-porosity region.
    por = _uniform_porosity((40, 40, 40), 0.5)
    dist = TruncatedGaussianTubeDistribution(0.05, 0.015, 0.01, 0.12)
    K_s = MultiscaleVolumeManager(
        por, scale=1.0, distributions={REGION_SUBRES_FIRST: dist},
        subresolution_model="sampled", subresolution_seed=0
    ).get_permeability_field()
    assert K_s.mean() == pytest.approx(0.5 * darcy_permeability(dist, 1.0), rel=0.1)


def test_sampled_model_converges_to_homogeneous_for_tiny_tubes():
    # Tubes << voxel -> N exceeds the sampling cap -> exact homogeneous value.
    por = _uniform_porosity((8, 8, 8), 0.5)
    dist = ConstantTubeDistribution(0.01)
    common = dict(scale=1.0, distributions={REGION_SUBRES_FIRST: dist})
    K_s = MultiscaleVolumeManager(
        por, subresolution_model="sampled", subresolution_seed=0, **common
    ).get_permeability_field()
    K_h = MultiscaleVolumeManager(
        por, subresolution_model="homogeneous", **common
    ).get_permeability_field()
    np.testing.assert_allclose(K_s, K_h, rtol=1e-12)
