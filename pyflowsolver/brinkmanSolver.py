"""BrinkmanSolver -- Darcy-Stokes-Brinkman flow on a MAC grid.

Thin subclass of ``StokesSolver`` that turns on the Brinkman drag term
``-(nu/K)*u`` in the momentum predictor (see ``StokesSolver(permeability=...)``
and brinkman.md). One solver then spans three regimes with a single spatially
varying permeability field ``K``:

- **open pores** ``K -> inf``  => drag 0, pure Stokes (recovers the base solver);
- **subresolution** finite ``K`` => Brinkman;
- **Darcy limit** small ``K``   => drag-dominated (use the implicit predictor).

Because the Darcy limit is a stiff reaction term, the default predictor here is
``"implicit"`` (unconditionally stable), unlike ``StokesSolver``'s explicit
default.

Input flexibility (confirmed design): accept EITHER a precomputed permeability
field (stays model-independent, like Stokes) OR a porosity map that is converted
to ``K`` internally via ``MultiscaleVolumeManager`` (lazy import, mirroring the
existing ``fast_laplacian_guess`` bridge in ``StokesSolver``).

STATUS: functional. The drag term, both input paths, and effective-permeability
reporting work and are tested (tests/unit/test_brinkman_solver.py). The
multiscale fast-Laplacian first guess is not wired yet, so ``fast_laplacian_guess``
defaults to False here (cold start); pass an ``initial_pressure`` to warm-start.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pyflowsolver.stokesSolver import StokesSolver


class BrinkmanSolver(StokesSolver):

    def __init__(self, volume=None, scale=1.0,
                 permeability=None, porosity_map=None,
                 labelmap: Optional[np.ndarray] = None,
                 distributions: Optional[Dict[int, object]] = None,
                 initial_pressure=None, initial_velocity=None,
                 backend="native", fast_laplacian_guess=False, **params):
        """
        Exactly one of `permeability` or `porosity_map` must be given.

        permeability: precomputed per-voxel K field (length^2), shaped like the
            volume. `volume` (the pore mask) is then required too. Fully
            model-independent path.
        porosity_map: graded porosity ([0..1] or [0..100]); K and the pore mask
            are built internally via MultiscaleVolumeManager (lazy import).
        labelmap, distributions: forwarded to MultiscaleVolumeManager when
            building from a porosity map (see multiscaleVolumeManager.py).

        Other arguments match StokesSolver. `predictor` defaults to "implicit"
        here (the Darcy limit is stiff); override via params if desired.
        `fast_laplacian_guess` defaults to False (the multiscale first guess is
        not wired yet); pass `initial_pressure` to warm-start.
        """
        if (permeability is None) == (porosity_map is None):
            raise ValueError("provide exactly one of `permeability` or `porosity_map`")

        # Brinkman-appropriate default: implicit predictor (stiff reaction term).
        params.setdefault("predictor", "implicit")

        if porosity_map is not None:
            volume, permeability = self._build_from_porosity(
                porosity_map, scale, labelmap, distributions)
        elif volume is None:
            raise ValueError("`permeability` path also requires `volume` (the pore mask)")

        super().__init__(
            volume, scale=scale,
            initial_pressure=initial_pressure, initial_velocity=initial_velocity,
            backend=backend, fast_laplacian_guess=fast_laplacian_guess,
            permeability=permeability, **params,
        )

    # ------------------------------------------------------------------ #
    # Porosity-map -> (pore mask, K field) bridge
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_from_porosity(porosity_map, scale, labelmap, distributions):
        """Build the pore `volume` mask and the K field from a porosity map.

        Lazily imports MultiscaleVolumeManager (which pulls fastLaplacian/pyedt),
        so a plain `import brinkmanSolver` stays free of that dependency.
        """
        from pyflowsolver.multiscaleVolumeManager import MultiscaleVolumeManager
        mvm = MultiscaleVolumeManager(
            porosity_map, scale=scale,
            labelmap=labelmap, distributions=distributions)
        K = mvm.get_permeability_field()
        # Pore mask: any non-solid voxel (open or subresolution) participates.
        volume = (K > 0).astype(np.float64)
        # TODO: keep a reference to `mvm` so the multiscale Darcy first guess
        #       (below) can reuse its assembled system instead of rebuilding.
        return volume, K

    # ------------------------------------------------------------------ #
    # First guess: multiscale fast-Laplacian pressure (drag-aware seed)
    # ------------------------------------------------------------------ #
    def _compute_fast_laplacian_guess(self):
        """Warm-start from a MULTISCALE fast-Laplacian (Darcy) pressure.

        Overrides StokesSolver's binarized poremap guess: routes through
        MultiscaleVolumeManager so subresolution voxels contribute their real
        conductivity, giving a physically consistent pressure for the seed. The
        inherited "from_pressure" velocity seed then reuses the (drag-aware)
        diffusion operator, so the seed automatically satisfies the Brinkman
        momentum balance rather than the pure-Stokes one.

        STATUS: TODO -- wire MultiscaleVolumeManager.get_sparse_system_jit() +
        MultigridSolver here, mirroring the base implementation but multiscale.
        """
        raise NotImplementedError(
            "BrinkmanSolver multiscale first guess not implemented yet; pass "
            "fast_laplacian_guess=False or an explicit initial_pressure (see brinkman.md)."
        )

    # ------------------------------------------------------------------ #
    # Effective permeability reporting (optional output)
    # ------------------------------------------------------------------ #
    def effective_permeability(self):
        """Sample-scale Darcy permeability K_eff (length^2) from the solution.

        Applies Darcy's law to the whole sample: with volumetric through-flow Q
        (averaged over the inlet/outlet z-planes of the MAC `w` field), total
        cross-section `A`, length `L`, imposed drop `dP = 1` (inlet p=1, outlet
        p=0) and dynamic viscosity `mu = nu*rho`,

            K_eff = (Q/A) * mu * L / dP .

        In the Darcy limit (uniform K, thin Brinkman boundary layers) this
        returns the input K; in the open (Stokes) limit it returns the duct's
        geometric permeability. Call after `solve()`.
        """
        if self.w is None:
            raise RuntimeError("call solve() before effective_permeability()")
        dx, dy, dz = (float(s) for s in self.scale[:3])
        w, h, d = self.volume.shape
        mu = self.params["viscosity"] * self.params["density"]
        face_area = dx * dy
        q_in = float(self.w[:, :, 0].sum()) * face_area
        q_out = float(self.w[:, :, d].sum()) * face_area
        Q = 0.5 * (q_in + q_out)
        A = (w * dx) * (h * dy)
        L = d * dz
        dP = 1.0
        return (Q / A) * mu * L / dP
