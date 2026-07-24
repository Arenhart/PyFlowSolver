"""Bundle-of-tubes model for subresolution porosity.

A subresolution voxel (one imaged below the pore scale, neither fully solid nor
fully open) is modelled as a *bundle of parallel capillary tubes* whose radii
follow a per-region distribution. From that single distribution we derive:

- **single-phase Darcy permeability** `k` -- the local conductivity a
  subresolution voxel contributes to the fast-Laplacian / Brinkman drag
  (implemented, single-phase);
- **two-phase capillary pressure** `Pc(S)` and **relative permeability**
  `kr(S)` as a function of saturation, via tube-by-tube invasion
  (documented + stubbed -- future two-phase step).

Design notes
------------
The region radius statistics live behind the ``TubeRadiusDistribution``
interface (an abstract base class) so any parametric or empirical distribution
can be plugged in, provided it can report its ``min``/``max``, draw a random
``sample``, and evaluate the ``cdf`` over a radius range. The bundle-of-tubes
functions below consume *only* that interface, so they are agnostic to the
concrete distribution. Three concrete distributions ship here:
``ConstantTubeDistribution``, ``TruncatedGaussianTubeDistribution`` and
``TruncatedLognormalTubeDistribution``.

Convention: radii and ``pore_scale`` are lengths in the same unit; permeability
/ conductance comes out in that unit squared (length^2), matching
``fastLaplacian.fast_laplacian_volume_generator`` and the harmonic-mean face
conductivities in ``VolumeManager``.

STATUS: single-phase (``darcy_permeability`` /
``subresolution_conductance_function``) and all three distributions are
implemented; ``capillary_pressure`` and ``relative_permeability`` are stubs
pending the two-phase step (see brinkman.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import ndtr, ndtri  # standard-normal CDF and its inverse

# Hagen-Poiseuille bundle constant: laminar flow in a circular tube gives a
# conductance ~ r^4 / (8 mu); factored so k has units length^2 (see docstring).
_HP_CONSTANT = 1.0 / 8.0


class TubeRadiusDistribution(ABC):
    """Distribution of capillary-tube radii within one subresolution region.

    The four methods below are the entire contract the bundle-of-tubes model
    relies on. Concrete subclasses may be analytic (see below) or empirical
    (e.g. a mercury-injection capillary-pressure curve resampled into radius
    bins). All radii are lengths in the same unit as the voxel ``scale``.
    """

    @abstractmethod
    def min(self) -> float:
        """Smallest tube radius with non-negligible probability."""

    @abstractmethod
    def max(self) -> float:
        """Largest tube radius with non-negligible probability."""

    @abstractmethod
    def sample(self, n: Optional[int] = None, rng=None) -> Union[float, np.ndarray]:
        """Draw random radii.

        Parameters
        ----------
        n : int or None
            If ``None``, return a single float. Otherwise return an ``(n,)``
            ndarray of draws.
        rng : numpy Generator or None
            Optional external generator. When given it is used instead of the
            distribution's internal RNG, so callers can make a whole field of
            draws reproducible from a single seed (independent of the
            distribution's own state).
        """

    @abstractmethod
    def cdf(self, r_low: float, r_high: float) -> float:
        """Probability that a tube radius falls in ``[r_low, r_high]``.

        Satisfies ``cdf(min, max) == 1`` and is monotone. Used both for the
        permeability moments (single-phase) and for the saturation <-> invaded
        radius mapping (two-phase).
        """


class ConstantTubeDistribution(TubeRadiusDistribution):
    """Degenerate (single-radius) bundle: every tube has radius ``radius``."""

    def __init__(self, radius: float):
        if radius <= 0.0:
            raise ValueError("radius must be > 0")
        self.radius = float(radius)

    def min(self) -> float:
        return self.radius

    def max(self) -> float:
        return self.radius

    def sample(self, n: Optional[int] = None, rng=None) -> Union[float, np.ndarray]:
        if n is None:
            return self.radius
        return np.full(int(n), self.radius, dtype=np.float64)

    def cdf(self, r_low: float, r_high: float) -> float:
        return 1.0 if (r_low <= self.radius <= r_high) else 0.0


class _TruncatedNormalBase(TubeRadiusDistribution):
    """Shared machinery for normal/log-normal radii truncated to [r_min, r_max].

    Subclasses define ``_z(r)`` mapping a radius to the standardised normal
    variate whose CDF is ``ndtr``. Everything else (truncation renormalisation,
    inverse-CDF sampling, min/max) is shared.
    """

    def __init__(self, mean: float, sigma: float,
                 r_min: float, r_max: float, seed: Optional[int] = None):
        if not (r_min > 0.0 and r_max > r_min):
            raise ValueError("require 0 < r_min < r_max")
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self._rng = np.random.default_rng(seed)
        # Standardised-normal CDF at the truncation bounds (mass renormaliser).
        self._phi_lo = float(ndtr(self._z(self.r_min)))
        self._phi_hi = float(ndtr(self._z(self.r_max)))
        self._mass = self._phi_hi - self._phi_lo

    def _z(self, r):  # radius -> standardised variate; overridden per subclass
        raise NotImplementedError

    def _r(self, phi):  # standardised CDF value -> radius; overridden per subclass
        raise NotImplementedError

    def min(self) -> float:
        return self.r_min

    def max(self) -> float:
        return self.r_max

    def cdf(self, r_low: float, r_high: float) -> float:
        lo = max(float(r_low), self.r_min)
        hi = min(float(r_high), self.r_max)
        if hi <= lo:
            return 0.0
        num = float(ndtr(self._z(hi))) - float(ndtr(self._z(lo)))
        return num / self._mass

    def sample(self, n: Optional[int] = None, rng=None) -> Union[float, np.ndarray]:
        gen = self._rng if rng is None else rng
        size = 1 if n is None else int(n)
        u = gen.random(size)
        # Inverse-CDF within the truncation window.
        phi = self._phi_lo + u * self._mass
        radii = self._r(phi)
        radii = np.clip(radii, self.r_min, self.r_max)
        return float(radii[0]) if n is None else radii

    # -- data-driven constructors ------------------------------------------ #
    # Fit the distribution's parameters to measured data (e.g. mercury-injection
    # throat radii), then truncate. `mean`/`sigma` are the moments in this class's
    # fit space (linear for the Gaussian, log for the log-normal); the fit space
    # is selected by `_fit_space`, so both subclasses share these constructors.

    @staticmethod
    def _fit_space(r):
        return r                                       # linear (log-normal overrides)

    @classmethod
    def _bounds(cls, r, r_min, r_max):
        lo = float(np.min(r)) if r_min is None else float(r_min)
        hi = float(np.max(r)) if r_max is None else float(r_max)
        return lo, hi

    @classmethod
    def from_samples(cls, radii, r_min=None, r_max=None, seed=None):
        """Fit to raw radius samples (unweighted). Truncation defaults to the data
        range; pass ``r_max`` = the image resolution F to drop already-resolved
        pores (a subresolution radius must be < F)."""
        r = np.asarray(radii, dtype=np.float64).ravel()
        r = r[np.isfinite(r) & (r > 0.0)]
        if r.size < 2:
            raise ValueError("need >= 2 positive radius samples to fit")
        x = cls._fit_space(r)
        mean, sigma = float(x.mean()), float(x.std())   # population (MLE) moments
        if sigma <= 0.0:
            raise ValueError("samples have zero spread; use ConstantTubeDistribution")
        lo, hi = cls._bounds(r, r_min, r_max)
        return cls(mean, sigma, lo, hi, seed=seed)

    @classmethod
    def from_density(cls, radii, density, r_min=None, r_max=None, seed=None):
        """Fit to a (radius, density) distribution curve, e.g. a mercury-injection
        pore-throat-size distribution. ``density`` values are relative weights
        (normalised internally); parameters are the weighted moments."""
        r = np.asarray(radii, dtype=np.float64).ravel()
        w = np.asarray(density, dtype=np.float64).ravel()
        if r.shape != w.shape:
            raise ValueError("radii and density must have the same length")
        m = np.isfinite(r) & np.isfinite(w) & (r > 0.0) & (w >= 0.0)
        r, w = r[m], w[m]
        if r.size < 2 or w.sum() <= 0.0:
            raise ValueError("need >= 2 positive-weight (radius, density) points")
        w = w / w.sum()
        x = cls._fit_space(r)
        mean = float(np.sum(w * x))
        sigma = float(np.sqrt(np.sum(w * (x - mean) ** 2)))
        if sigma <= 0.0:
            raise ValueError("density has zero spread; use ConstantTubeDistribution")
        lo, hi = cls._bounds(r, r_min, r_max)
        return cls(mean, sigma, lo, hi, seed=seed)


class TruncatedGaussianTubeDistribution(_TruncatedNormalBase):
    """Normal radii ``N(mean, sigma)`` truncated to ``[r_min, r_max]``.

    Build from data with ``from_samples`` / ``from_density`` (fits mean, sigma).
    """

    def _z(self, r):
        return (np.asarray(r, dtype=np.float64) - self.mean) / self.sigma

    def _r(self, phi):
        return self.mean + self.sigma * ndtri(phi)


class TruncatedLognormalTubeDistribution(_TruncatedNormalBase):
    """Log-normal radii (``ln r ~ N(mean, sigma)``) truncated to ``[r_min, r_max]``.

    ``mean`` and ``sigma`` parameterise the underlying normal in log-space, so
    e.g. ``mean=np.log(0.25)`` centres the distribution near a 0.25 radius.
    ``from_samples`` / ``from_density`` fit these log-space moments from data.
    """

    @staticmethod
    def _fit_space(r):
        return np.log(np.asarray(r, dtype=np.float64))    # fit in log-space

    def _z(self, r):
        return (np.log(np.asarray(r, dtype=np.float64)) - self.mean) / self.sigma

    def _r(self, phi):
        return np.exp(self.mean + self.sigma * ndtri(phi))


class EmpiricalTubeRadiusDistribution(TubeRadiusDistribution):
    """Radius distribution defined by an empirical sample set (e.g. the footprint
    / local-thickness values of a high-resolution image). ``min``/``max`` are the
    sample extremes, ``sample`` resamples with replacement, and ``cdf`` is the
    empirical (counting) CDF over the sorted samples."""

    def __init__(self, samples, seed: Optional[int] = None):
        s = np.asarray(samples, dtype=np.float64).ravel()
        s = s[np.isfinite(s) & (s > 0.0)]
        if s.size == 0:
            raise ValueError("EmpiricalTubeRadiusDistribution needs >0 positive samples")
        self._samples = np.sort(s)
        self._rng = np.random.default_rng(seed)

    def min(self) -> float:
        return float(self._samples[0])

    def max(self) -> float:
        return float(self._samples[-1])

    def sample(self, n: Optional[int] = None, rng=None) -> Union[float, np.ndarray]:
        gen = self._rng if rng is None else rng
        if n is None:
            return float(gen.choice(self._samples))
        return gen.choice(self._samples, size=int(n))

    def cdf(self, r_low: float, r_high: float) -> float:
        lo = int(np.searchsorted(self._samples, r_low, side="left"))
        hi = int(np.searchsorted(self._samples, r_high, side="right"))
        return (hi - lo) / self._samples.size


# --------------------------------------------------------------------------- #
# Single-phase bundle-of-tubes permeability
# --------------------------------------------------------------------------- #

def _radius_moment(distribution: TubeRadiusDistribution, p: int,
                   n_bins: int = 64) -> float:
    """Number-averaged moment ``<r^p> = E[r^p | r in [min, max]]`` via the cdf.

    Midpoint quadrature over ``n_bins`` radius bins weighted by the distribution
    bin masses (``cdf``), normalised so the result is a proper conditional
    expectation. Degenerate (constant) distributions return ``r^p`` directly.
    """
    r_lo, r_hi = distribution.min(), distribution.max()
    if r_hi - r_lo <= 1e-15:
        return float(r_lo) ** p
    edges = np.linspace(r_lo, r_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.array([distribution.cdf(edges[i], edges[i + 1])
                        for i in range(n_bins)])
    total = float(weights.sum())
    if total <= 0.0:
        return (0.5 * (r_lo + r_hi)) ** p
    return float(np.sum(weights * centers ** p) / total)


def darcy_permeability(distribution: TubeRadiusDistribution,
                       porosity: float,
                       n_bins: int = 64) -> float:
    """Single-fluid (homogeneous-bundle) Darcy permeability of a subresolution
    voxel, in length^2.

    A bundle of parallel Hagen-Poiseuille tubes filling fraction ``porosity`` of
    the cross-section, radii drawn from the (number) distribution ``f(r)``,
    has effective permeability

        K = porosity * (1/8) * <r^4> / <r^2>

    where ``<.>`` are number-averaged moments over ``[min, max]``. The
    ``<r^4>/<r^2>`` ratio is the area/flow-weighted mean of the per-tube
    permeability ``r^2/8`` (each tube weighted by its cross-section ``pi r^2``
    and probability), so the ``r^4`` flow weighting pulls the effective radius
    toward the larger tubes. Derivation: total flow ``Q ~ sum r_i^4`` and
    ``porosity ~ sum r_i^2`` over the tubes filling the voxel; eliminating the
    tube count gives the ratio above.

    The moments are integrated numerically from the distribution's ``cdf`` (bin
    masses), so the function is agnostic to the concrete distribution. Valid when
    the tubes are much smaller than the voxel (voxel is a representative volume);
    the near-resolution regime is left to other (future) subresolution models --
    see ``subresolution_permeability_field`` and brinkman.md.

    Parameters
    ----------
    distribution : TubeRadiusDistribution
        Radius statistics for this region.
    porosity : float
        Local porosity fraction in ``[0, 1]``.
    n_bins : int
        Radius-bin count for the moment quadrature.

    Returns
    -------
    float
        Permeability in length^2 (same unit as radius^2).
    """
    m2 = _radius_moment(distribution, 2, n_bins)
    m4 = _radius_moment(distribution, 4, n_bins)
    moment_ratio = (m4 / m2) if m2 > 0.0 else 0.0     # <r^4>/<r^2>
    return _HP_CONSTANT * float(porosity) * moment_ratio


# --------------------------------------------------------------------------- #
# Pluggable per-voxel permeability models (subresolution K field)
# --------------------------------------------------------------------------- #
# A subresolution model maps (distribution, porosity_fraction_array) -> K array.
# "homogeneous" is the tubes-<<-voxel limit (deterministic). Near-resolution
# models (finite-bundle sampling, critical-path/EMT, correlated realizations)
# will register here later WITHOUT touching the assembly path -- they may need
# extra inputs (voxel size, seed), threaded via **kwargs. See brinkman.md.

def _homogeneous_permeability_field(distribution, porosity_fraction, **kwargs):
    """K = porosity * (1/8) <r^4>/<r^2>, evaluated per voxel (vectorized).

    The distribution factor ``(1/8)<r^4>/<r^2>`` is porosity-independent, so it
    is computed once and scaled by the local porosity fraction of each voxel.
    Correct in the tubes-<<-voxel (REV) limit.
    """
    factor = darcy_permeability(distribution, 1.0)
    return np.asarray(porosity_fraction, dtype=np.float64) * factor


def _voxel_cross_section_area(scale) -> float:
    """In-plane area perpendicular to the (z-driven) flow: dx*dy."""
    if scale is None:
        return 1.0
    s = np.asarray(scale, dtype=np.float64).ravel()
    return float(s[0] * s[1]) if s.size >= 2 else float(s[0] * s[0])


def _sampled_permeability_field(distribution, porosity_fraction,
                                scale=None, seed=0, max_tubes=1000,
                                n_bins=64, **kwargs):
    """Finite-bundle sampling model for the near-resolution regime.

    Each voxel holds only the number of tubes that physically fit,

        N = round( A * phi / (pi * <r^2>) ),   A = voxel cross-section (dx*dy)

    drawn from the distribution, and its permeability is the Hagen-Poiseuille
    bundle value

        K = (pi/8) * sum_i r_i^4 / A .

    Because E[sum r_i^4] = N * <r^4> and N = A*phi/(pi<r^2>), the expectation is
    E[K] = (phi/8) <r^4>/<r^2> -- exactly the homogeneous value -- so this
    CONVERGES to homogeneous as N grows (tubes << voxel) and injects real
    voxel-to-voxel HETEROGENEITY when N is small (features near resolution). The
    downstream Darcy/Brinkman solve then resolves connectivity/bottlenecking
    across the heterogeneous field via the existing harmonic-mean faces.

    Reproducible: all draws come from a single ``seed``-ed generator, consumed in
    a fixed voxel order. When ``N > max_tubes`` the bundle has effectively
    converged, so the exact deterministic homogeneous value is used instead
    (bounds cost and removes pointless noise). Stochastic, so seed-controlled.
    """
    phi = np.asarray(porosity_fraction, dtype=np.float64)
    area = _voxel_cross_section_area(scale)
    r2_mean = _radius_moment(distribution, 2, n_bins)
    homogeneous_factor = darcy_permeability(distribution, 1.0, n_bins)
    rng = np.random.default_rng(seed)
    coef = np.pi / 8.0
    mean_tube_area = np.pi * r2_mean

    flat = phi.ravel()
    out = np.zeros(flat.size, dtype=np.float64)
    for idx in range(flat.size):
        p = flat[idx]
        if p <= 0.0:
            continue
        n_tubes = int(round(area * p / mean_tube_area))
        if n_tubes < 1:
            n_tubes = 1
        if n_tubes > max_tubes:                       # converged -> deterministic
            out[idx] = p * homogeneous_factor
            continue
        radii = np.asarray(distribution.sample(n_tubes, rng=rng), dtype=np.float64)
        out[idx] = coef * float(np.sum(radii ** 4)) / area
    return out.reshape(phi.shape)


SUBRESOLUTION_MODELS = {
    "homogeneous": _homogeneous_permeability_field,
    "sampled": _sampled_permeability_field,
}


def subresolution_permeability_field(distribution: TubeRadiusDistribution,
                                     porosity_fraction,
                                     model: str = "homogeneous",
                                     **kwargs) -> np.ndarray:
    """Per-voxel subresolution permeability K (length^2) for a region.

    Dispatches to a registered model in ``SUBRESOLUTION_MODELS``. ``model`` names
    the strategy (default ``"homogeneous"``); ``porosity_fraction`` is the
    per-voxel porosity in ``[0, 1]`` for the region's voxels.
    """
    try:
        fn = SUBRESOLUTION_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown subresolution model {model!r}; "
            f"available: {sorted(SUBRESOLUTION_MODELS)}"
        )
    return fn(distribution, porosity_fraction, **kwargs)


def subresolution_conductance_function(distribution: TubeRadiusDistribution):
    """Build a *vectorized* ``subresolution_function`` for ``fastLaplacian``.

    ``fast_laplacian_volume_generator`` calls its ``subresolution_function`` hook
    ONCE on the whole integer porosity-percent array (values 0..99, with 0 for
    non-subresolution voxels), NOT element by element. The returned closure must
    therefore:

    - accept an integer/float ndarray of porosity *percent* (0..99),
    - return a same-shaped float ndarray of conductance (length^2),
    - map input 0 -> output 0 (so it never pollutes solid/open voxels when the
      generator adds it to the Arns field).

    We precompute ``darcy_permeability`` on the 1..99 percent lookup table and
    index into it, which is both vectorized and exact per the bundle model.
    """
    # Lookup table: index = porosity percent (0..100). lut[0] = 0 by construction.
    lut = np.zeros(101, dtype=np.float64)
    for percent in range(1, 100):
        lut[percent] = darcy_permeability(distribution, percent / 100.0)
    # 100% (fully open) is handled by the Arns/EDT path, not here; leave 0.

    def _f(porosity_percent_array):
        idx = np.clip(np.rint(porosity_percent_array).astype(np.int64), 0, 100)
        return lut[idx]

    return _f


# --------------------------------------------------------------------------- #
# Two-phase bundle-of-tubes  (STUBS -- future step, see brinkman.md)
# --------------------------------------------------------------------------- #

def capillary_pressure(distribution: TubeRadiusDistribution,
                       saturation: float,
                       surface_tension: float = 1.0,
                       contact_angle: float = 0.0) -> float:
    """Capillary pressure at a given wetting-phase saturation (STUB).

    Planned model: order tubes by radius; at wetting saturation ``S`` the
    non-wetting phase occupies the largest tubes down to the radius ``r*`` where
    the cumulative (volume-weighted) tube fraction equals ``1 - S`` -- found by
    inverting ``distribution.cdf``. Then Young-Laplace:

        Pc(S) = 2 * surface_tension * cos(contact_angle) / r*(S)
    """
    raise NotImplementedError("capillary_pressure: two-phase step (see brinkman.md)")


def relative_permeability(distribution: TubeRadiusDistribution,
                          saturation: float) -> Tuple[float, float]:
    """Wetting/non-wetting relative permeability at saturation (STUB).

    Planned model: with the invasion radius ``r*(S)`` from ``capillary_pressure``,
    the wetting phase conducts through tubes with ``r <= r*`` and the non-wetting
    phase through ``r > r*``; each kr is that sub-bundle's Hagen-Poiseuille
    conductance divided by the full-bundle conductance (the ``darcy_permeability``
    moment integral restricted to the sub-range).

    Returns ``(krw, krnw)``.
    """
    raise NotImplementedError("relative_permeability: two-phase step (see brinkman.md)")
