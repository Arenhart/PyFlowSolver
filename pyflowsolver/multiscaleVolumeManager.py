"""MultiscaleVolumeManager -- Darcy assembly over a graded porosity map.

``VolumeManager`` binarizes its input (every non-empty voxel becomes a fully
open pore) so it cannot represent **subresolution porosity** -- voxels that are
partially open. ``MultiscaleVolumeManager`` preserves the graded porosity map
and routes it through ``fastLaplacian.fast_laplacian_volume_generator`` with a
per-region bundle-of-tubes ``subresolution_function``, producing a continuous
conductivity field. Everything downstream (harmonic-mean face assembly,
condensation, flux/permeability integration) is inherited unchanged -- the base
class already treats ``self.volume`` as an arbitrary float conductivity field.

Input conventions
------------------
- ``porosity_map``: float/int ndarray. Accepts ``[0..1]`` or ``[0..100]``
  (auto-detected by ``max > 1``); stored internally as ``[0..1]`` float in
  ``self.porosity_map``. Converted to the ``0/1-99/100`` uint percent convention
  only at the ``fastLaplacian`` boundary.
- ``labelmap``: optional int region map. ``0`` = solid, ``1`` = resolved (fully
  open) porosity, ``2+`` = distinct subresolution regions. If omitted, one is
  auto-generated: ``0`` where porosity is 0, ``1`` where porosity is maximal
  (fully open), ``2`` for all intermediate porosities (a single subresolution
  region).
- ``distributions``: ``{region_label: TubeRadiusDistribution}`` -- one entry per
  ``2+`` region present in ``labelmap``. Provides the tube-radius statistics the
  bundle-of-tubes model turns into local permeability (and, later, Pc/kr).

STATUS: skeleton. Signatures, the ordering fix, and the conversion routing are
in place; permeability/label internals are marked TODO.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.fastLaplacian import fast_laplacian_volume_generator
from pyflowsolver.tubeBundle import (
    TubeRadiusDistribution,
    subresolution_permeability_field,
    SUBRESOLUTION_MODELS,
)

# Region-label convention (distinct from constants.SOLID/PORE which label BCs).
REGION_SOLID = 0
REGION_RESOLVED = 1
REGION_SUBRES_FIRST = 2  # first subresolution region; further regions are 3, 4, ...

# Sentinel "infinite" permeability for fully-open voxels in the K field: the
# Brinkman drag nu/K -> 0 there, recovering Stokes. A large finite value keeps
# arithmetic well-defined; the solver treats it as no-drag.
OPEN_PERMEABILITY = np.inf


class MultiscaleVolumeManager(VolumeManager):

    def __init__(self, porosity_map, scale=1,
                 labelmap: Optional[np.ndarray] = None,
                 distributions: Optional[Dict[int, TubeRadiusDistribution]] = None,
                 boundary_volume=None,
                 enhanced_model: bool = True,
                 subresolution_model: str = "homogeneous",
                 subresolution_seed: int = 0):
        """
        porosity_map: 3D ndarray of porosity, in [0..1] or [0..100] (auto-detected).
        scale: voxel size, scalar or (dx, dy, dz).
        labelmap: optional region map (0 solid / 1 resolved / 2+ subresolution).
        distributions: {region_label: TubeRadiusDistribution} for every 2+ region.
        boundary_volume: irregular INLET/OUTLET geometry (see VolumeManager).
        enhanced_model: use the enhanced (footprint) Arns model for open voxels.
        subresolution_model: strategy for the per-voxel subresolution permeability
            (see tubeBundle.SUBRESOLUTION_MODELS). Default "homogeneous" (the
            tubes-<<-voxel limit); "sampled" is the near-resolution finite-bundle
            model.
        subresolution_seed: RNG seed for stochastic subresolution models (e.g.
            "sampled"), so get_permeability_field is reproducible. Ignored by
            deterministic models.
        """
        porosity_map = np.asarray(porosity_map)
        # Normalize to [0..1] float internally.
        self.porosity_map = self._normalize_porosity(porosity_map)
        self.labelmap = (np.asarray(labelmap) if labelmap is not None
                         else self._auto_labelmap(self.porosity_map))
        self.distributions = dict(distributions) if distributions else {}
        self._enhanced_model = bool(enhanced_model)
        if subresolution_model not in SUBRESOLUTION_MODELS:
            raise ValueError(
                f"unknown subresolution_model {subresolution_model!r}; "
                f"available: {sorted(SUBRESOLUTION_MODELS)}"
            )
        self._subresolution_model = subresolution_model
        self._subresolution_seed = int(subresolution_seed)
        self._validate_regions()

        # ------------------------------------------------------------------ #
        # ORDERING FIX (critical): the base __init__ runs filter_connected_volume
        # / _calc_null_counts / self.nonzeros off `self.volume > 0` BEFORE any
        # conversion. If we passed the porosity map straight in, the unknown set
        # would be computed from porosity>0 and could then desync from the final
        # conductivity field (the Arns model can yield ~0 conductivity at some
        # open-voxel edges, silently turning an "unknown" into a solid and
        # breaking ravel_sparse_solution's index walk). So we build the
        # conductivity field FIRST and hand the ALREADY-CONVERTED field to the
        # base constructor, so all condensation bookkeeping is computed on it.
        # ------------------------------------------------------------------ #
        conductivity = self._build_conductivity_field(scale)
        super().__init__(conductivity, scale=scale, boundary_volume=boundary_volume)

        # Guard: no non-solid voxel may have become 0 conductivity, else the
        # unknown set diverges from the porosity topology.
        # TODO: assert (conductivity > 0).sum() matches the intended pore count
        #       and repair (nudge to a small floor) any Arns zeros if needed.

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_porosity(porosity_map: np.ndarray) -> np.ndarray:
        """Return porosity as float in [0..1]. Inputs with max > 1 are /100."""
        p = porosity_map.astype(np.float64)
        if p.max() > 1.0:
            p = p / 100.0
        # TODO: validate range [0..1] and warn on values slightly outside.
        return p

    @staticmethod
    def _auto_labelmap(porosity01: np.ndarray) -> np.ndarray:
        """Build a default region map: 0 solid / 1 fully-open / 2 subresolution.

        Per spec, "fully open" means porosity == the scale maximum (1.0 on the
        internal [0..1] scale), NOT the data maximum -- so a sample with no
        exactly-open voxel simply has no region-1 voxels, and everything between
        solid and fully open becomes one subresolution region.
        """
        labels = np.zeros(porosity01.shape, dtype=np.int32)
        open_mask = porosity01 >= 1.0
        intermediate = (porosity01 > 0.0) & ~open_mask
        labels[open_mask] = REGION_RESOLVED
        labels[intermediate] = REGION_SUBRES_FIRST
        return labels

    def _validate_regions(self):
        """Every 2+ region present in labelmap must have a distribution."""
        present = set(int(v) for v in np.unique(self.labelmap))
        subres = {r for r in present if r >= REGION_SUBRES_FIRST}
        missing = subres - set(self.distributions)
        if missing:
            raise ValueError(
                f"subresolution regions {sorted(missing)} have no TubeRadiusDistribution "
                f"in `distributions` (provided: {sorted(self.distributions)})"
            )

    # ------------------------------------------------------------------ #
    # Conductivity construction
    # ------------------------------------------------------------------ #
    def _build_conductivity_field(self, scale) -> np.ndarray:
        """Assemble the per-voxel Darcy conductivity field (self.volume):

        - fully-open voxels (region 1): EDT/footprint Arns conductance;
        - subresolution voxels (2+): the selected ``subresolution_model``'s K
          for that region's distribution (using exact porosity, per region);
        - solid: 0.

        The two contributions live on disjoint voxel sets and are summed. This is
        the ONE place the open-voxel Arns model and the subresolution model meet,
        and it shares ``_subresolution_permeability_field`` with
        ``get_permeability_field`` -- so the fast-Laplacian conductance and the
        Brinkman K field are guaranteed to use the same subresolution model.
        """
        scale_arr = self._scale_tuple(scale)
        # Arns/EDT conductance for fully-open voxels only. Feeding a 0/100 map
        # (no 1-99 values) means the generator's own subresolution path is inert,
        # so this returns the open-voxel Arns field and 0 everywhere else.
        open_only = np.where(self.labelmap == REGION_RESOLVED,
                             100, 0).astype(np.uint8)
        arns = fast_laplacian_volume_generator(
            open_only,
            scale_arr,
            subresolution_function=None,
            closed_border=False,
            enhanced_model=self._enhanced_model,
        )
        subres = self._subresolution_permeability_field(scale_arr)
        return np.asarray(arns, dtype=np.float64) + subres

    def _subresolution_permeability_field(self, scale) -> np.ndarray:
        """Per-voxel subresolution permeability K, 0 outside subresolution regions.

        Each subresolution region (2+) uses its own distribution and the selected
        ``subresolution_model`` (see tubeBundle.SUBRESOLUTION_MODELS), evaluated on
        that region's exact per-voxel porosity. Shared by the conductivity field
        (which adds Arns on the open voxels) and by ``get_permeability_field``
        (which sets the open voxels to infinity) -- so a single model choice /
        seed drives both, and new models plug in here for both consumers at once.
        """
        K = np.zeros(self.porosity_map.shape, dtype=np.float64)
        for region, dist in self.distributions.items():
            mask = self.labelmap == region
            if not mask.any():
                continue
            K[mask] = subresolution_permeability_field(
                dist, self.porosity_map[mask], model=self._subresolution_model,
                scale=scale, seed=self._subresolution_seed)
        return K

    @staticmethod
    def _scale_tuple(scale):
        s = np.asarray(scale, dtype=np.float64).ravel()
        return np.repeat(s, 3) if s.size == 1 else s[:3]

    # ------------------------------------------------------------------ #
    # Override the (binarizing) base conversion so callers using the standard
    # entry point get the multiscale field instead of "Not implemented yet".
    # ------------------------------------------------------------------ #
    def convert_pore_volume_to_laplacian_conductivity(self, porosity_map=True,
                                                       enhanced_model=None):
        """No-op-friendly override: the conductivity field is already built in
        __init__ (see the ordering fix). Kept so code calling the base method on
        a MultiscaleVolumeManager stays correct instead of re-binarizing.
        """
        # self.volume already holds the multiscale conductivity from __init__.
        # TODO: allow rebuilding with a different enhanced_model here if desired.
        return

    # ------------------------------------------------------------------ #
    # Bridge to BrinkmanSolver
    # ------------------------------------------------------------------ #
    def get_permeability_field(self) -> np.ndarray:
        """Per-voxel permeability K (length^2) for the Brinkman drag nu/K.

        - fully-open voxels (region 1): ``OPEN_PERMEABILITY`` (drag -> 0, Stokes);
        - subresolution voxels (2+): bundle-of-tubes ``darcy_permeability``;
        - solid voxels (0): 0.

        This is the object BrinkmanSolver consumes. It shares the subresolution
        model with the conductance field ``self.volume`` (both go through
        ``_subresolution_permeability_field``); the only difference is that open
        voxels here become infinite K (drag -> 0) rather than an Arns conductance.
        Connectivity between voxels is NOT resolved here -- that is left to the
        global Darcy/Brinkman solve on the resulting heterogeneous K field.
        """
        K = self._subresolution_permeability_field(self.scale)
        K[self.labelmap == REGION_RESOLVED] = OPEN_PERMEABILITY
        return K
        #       K[mask] = ...
        return K
