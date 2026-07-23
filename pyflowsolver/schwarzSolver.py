"""Distributed-ready iterative Schwarz solver for the pressure-Poisson system.

Motivation (see stress_test_poisson.py): the multigrid Poisson solve is fast but
RAM-bound -- the global CSR matrix + AMG hierarchy dominate memory and cap the
tractable volume on a single node. This solver breaks that ceiling by never
materialising the global matrix: the geometry is split into z-slabs, and **each
slab assembles and solves only its own subdomain**, so peak RAM per node scales
as (global / n_slabs). Add slabs (nodes) to solve arbitrarily large volumes, at
the cost of more wall-clock time (Schwarz rounds + halo communication).

It is the convergent evolution of `distributedDarcySolver.py`: instead of one
frozen-boundary block solve, it runs **additive Schwarz** -- each round every
slab solves its local Laplacian using its neighbours' latest boundary pressure
as a Dirichlet (halo) condition, then exchanges updated halo layers; iterating
drives the coupling to consistency and the field to the true global solution.

This module contains the numerical core, written so the same per-slab
assemble/solve runs in three modes (all giving the same solution):
  * ``solve_serial`` -- all slabs resident in one process (halo exchange in
    memory). Fast, but holds every slab at once (RAM ~ the whole problem); used
    as the reference and as the base the distributed mode parallelizes.
  * ``solve_streaming(resident=k)`` -- out-of-core on ONE machine: keep at most k
    slabs resident, rebuild the rest on demand each round (no disk). Peak RAM =
    floor + k*O(N/n_slabs), so k=1 reaches the distributed per-node RAM floor
    locally, trading speed (rebuilds) for memory.
  * distributed -- the same per-slab assemble/solve shipped to Dask/SLURM workers
    (see docker/), exchanging halo layers between rounds. (Remote wiring is a
    separate task.)

Conventions match VolumeManager's regular (z-driven) bare-Laplacian: unit
conductivity, z=0 inlet (p=1) / z=max outlet (p=0) as diagonal ghost terms, and
no-flux (dropped) connections at solid walls. Isotropic voxels.
"""

from collections import OrderedDict

import numpy as np

from pyflowsolver.multigridSolver import MultigridSolver, _csr_matvec
from pyflowsolver.pressurePoisson import (
    filter_percolating, global_fluid_index, slab_ranges, assemble_slab)



class SchwarzSolver:
    """Additive-Schwarz Poisson solver over geometry z-slabs (RAM-bounded)."""

    def __init__(self, volume, n_partitions=2, target_error=1e-6,
                 max_rounds=500, local_target=1e-9, use_coarse=True,
                 filter_disconnected=True):
        mask = (np.asarray(volume) > 0)
        # Match VolumeManager: drop non-percolating pores, which would otherwise
        # be singular (zero-diagonal) rows in the slab systems.
        self.mask = filter_percolating(mask) if filter_disconnected else mask
        self.n_partitions = n_partitions
        self.target_error = target_error
        self.max_rounds = max_rounds
        self.local_target = local_target
        # Two-level correction: a global coarse solve (one DOF per slab) each
        # round removes the slow inter-slab modes that one-level Schwarz leaves,
        # keeping the round count ~independent of the number of slabs. Assembled
        # entirely from local slab data (no global matrix), so it preserves the
        # RAM win. Disable to compare against plain one-level Schwarz.
        self.use_coarse = use_coarse
        self.gidx, self.N = global_fluid_index(self.mask)
        self.slabs = None
        self.solvers = None
        self.a_coarse = None
        self.b_norm = 1.0

    def build(self):
        """Assemble every slab's local system and multigrid hierarchy (once),
        plus the coarse operator A_c for the two-level correction."""
        self.slabs, self.solvers = [], []
        for (z_lo, z_hi) in slab_ranges(self.mask.shape[2], self.n_partitions):
            slab = assemble_slab(self.mask, self.gidx, z_lo, z_hi)
            self.slabs.append(slab)
            mg = MultigridSolver(backend="native", target_error=self.local_target)
            mg.set_linear_system(slab["a_sparse"],
                                 np.zeros(slab["local_n"], dtype=np.float64))
            mg.generate_preconditioner()
            self.solvers.append(mg)

        # Coarse operator A_c = P^T A P with P = per-slab piecewise-constant
        # (indicator). Tridiagonal along z: diagonal = each slab's local
        # matrix-entry sum; off-diagonal(k, k+1) = number of interface couplings
        # (each off-diagonal A entry is +1). Symmetric by construction.
        k = self.n_partitions
        A_c = np.zeros((k, k), dtype=np.float64)
        for i, slab in enumerate(self.slabs):
            A_c[i, i] = slab["diag_block_sum"]
            if i + 1 < k:
                coupling = float(slab["halo_above"])   # == slab[i+1]["halo_below"]
                A_c[i, i + 1] = coupling
                A_c[i + 1, i] = coupling
        self.a_coarse = A_c

        sq = sum(float(np.dot(s["b_base"], s["b_base"])) for s in self.slabs)
        self.b_norm = np.sqrt(sq) or 1.0

    def _global_residual(self, p):
        """Global residual r = b - A p, returned per slab (owned rows), plus its
        norm. Uses only local matvecs + halo values -- distributed-friendly."""
        r_norm_sq = 0.0
        r_slabs = []
        for slab, mg in zip(self.slabs, self.solvers):
            a = slab["a_sparse"]
            p_owned = np.ascontiguousarray(p[slab["owned_grows"]])
            ap = np.empty(slab["local_n"], dtype=np.float64)
            _csr_matvec(a["val"], a["col_idx"], a["row_ptr"], p_owned, ap)
            r = slab["b_base"] - ap
            if slab["halo_rows"].size:
                np.add.at(r, slab["halo_rows"], -p[slab["halo_gcols"]])
            r_slabs.append(r)
            r_norm_sq += float(np.dot(r, r))
        return r_slabs, np.sqrt(r_norm_sq)

    def solve_serial(self, initial=None):
        """Run (two-level) additive Schwarz to convergence in-process, exchanging
        halos in RAM -- the same algorithm the distributed workers run.

        Each round: an additive Schwarz sweep (per-slab local Dirichlet solve via
        native MG) followed, when `use_coarse`, by a global coarse correction
        (one DOF per slab) that removes the slow inter-slab modes. Stops on the
        relative global residual ||b - A p|| / ||b||, computed from local matvecs
        + halo values (no global matrix). Returns (p_global, rounds, residual).
        """
        if self.slabs is None:
            self.build()
        p = np.zeros(self.N) if initial is None else initial.astype(np.float64).copy()

        residual = np.inf
        rnd = 0
        for rnd in range(1, self.max_rounds + 1):
            # --- fine level: additive Schwarz sweep (local Dirichlet solves) ---
            p_new = p.copy()
            for slab, mg in zip(self.slabs, self.solvers):
                b = slab["b_base"].copy()
                # halo Dirichlet from the *previous* iterate (additive Schwarz)
                if slab["halo_rows"].size:
                    np.add.at(b, slab["halo_rows"], -p[slab["halo_gcols"]])
                mg.b_array = b
                x0 = np.ascontiguousarray(p[slab["owned_grows"]])
                x, _, _ = mg.solve_pcg(X0=x0.copy())
                p_new[slab["owned_grows"]] = x
            p = p_new

            # --- coarse level: global correction from the residual ------------
            r_slabs, residual = self._global_residual(p)
            if self.use_coarse and self.n_partitions > 1:
                r_c = np.array([float(np.sum(r)) for r in r_slabs])   # P^T r
                e_c = np.linalg.solve(self.a_coarse, r_c)             # A_c e = r_c
                for i, slab in enumerate(self.slabs):                 # p += P e_c
                    p[slab["owned_grows"]] += e_c[i]
                _, residual = self._global_residual(p)

            if residual / self.b_norm < self.target_error:
                break
        self.x = p
        self.rounds = rnd
        self.residual = residual / self.b_norm
        return p, rnd, self.residual

    def solve_streaming(self, resident=1, initial=None):
        """Out-of-core additive two-level Schwarz on a single machine.

        Same slab algorithm as `solve_serial`, but holds at most `resident` slabs
        (matrix + AMG hierarchy) in RAM at once, **rebuilding the others on demand
        each round** (no disk). Peak RAM is therefore

            floor  +  resident * O(N / n_partitions)

        where the floor is the full-grid arrays (mask, gidx, global p). With
        resident=1 this reaches the distributed solver's per-node RAM floor on one
        box, trading speed (n_partitions rebuilds per round) for memory. The knob
        spans the whole range: resident >= n_partitions caches everything and is
        equivalent to `solve_serial` (fast, high RAM).

        Converges to the same solution as `solve_serial`; returns
        (p_global, rounds, relative_residual). The two-level scheme is
        *multiplicative* (coarse correction uses the post-sweep residual, like
        `solve_serial`) -- this needs a second rebuild pass per round for that
        residual, but is robustly convergent (an additive coarse-on-top-of-exact-
        fine-solve overshoots and diverges).
        """
        ranges = slab_ranges(self.mask.shape[2], self.n_partitions)
        resident = max(1, min(int(resident), self.n_partitions))

        cache = OrderedDict()                       # i -> (slab, mg), LRU
        scalars = [None] * self.n_partitions        # (diag_block_sum, halo_above)
        b_sq = [None] * self.n_partitions           # per-slab ||b_base||^2

        def get(i):
            hit = cache.get(i)
            if hit is not None:
                cache.move_to_end(i)
                return hit
            z_lo, z_hi = ranges[i]
            slab = assemble_slab(self.mask, self.gidx, z_lo, z_hi)
            mg = MultigridSolver(backend="native", target_error=self.local_target)
            mg.set_linear_system(slab["a_sparse"],
                                 np.zeros(slab["local_n"], dtype=np.float64))
            mg.generate_preconditioner()
            if scalars[i] is None:                  # capture coarse inputs once
                scalars[i] = (slab["diag_block_sum"], slab["halo_above"])
                b_sq[i] = float(np.dot(slab["b_base"], slab["b_base"]))
            cache[i] = (slab, mg)
            if len(cache) > resident:
                cache.popitem(last=False)           # evict least-recently-used
            return slab, mg

        def slab_rhs(slab, p_):                      # b_base + halo Dirichlet(p_)
            b = slab["b_base"].copy()
            if slab["halo_rows"].size:
                np.add.at(b, slab["halo_rows"], -p_[slab["halo_gcols"]])
            return b

        p = np.zeros(self.N) if initial is None else initial.astype(np.float64).copy()
        a_coarse = None
        b_norm = None
        residual = np.inf
        rnd = 0
        for rnd in range(1, self.max_rounds + 1):
            # --- Pass A: fine sweep (exact local solves, halo from old p) ------
            p_new = p.copy()
            for i in range(self.n_partitions):
                slab, mg = get(i)
                mg.b_array = slab_rhs(slab, p)
                x0 = np.ascontiguousarray(p[slab["owned_grows"]])
                x, _, _ = mg.solve_pcg(X0=x0.copy())
                p_new[slab["owned_grows"]] = x
            p = p_new
            if b_norm is None:                      # all slabs seen after pass A
                b_norm = np.sqrt(sum(b_sq)) or 1.0
            if a_coarse is None and all(s is not None for s in scalars):
                a_coarse = self._coarse_operator(scalars)

            # --- Pass B: post-sweep residual (rebuild) + coarse correction -----
            r_c = np.zeros(self.n_partitions)       # P^T r (per-slab residual sum)
            r_norm_sq = 0.0
            for i in range(self.n_partitions):
                slab, mg = get(i)
                a = slab["a_sparse"]
                x_owned = np.ascontiguousarray(p[slab["owned_grows"]])
                ap = np.empty(slab["local_n"], dtype=np.float64)
                _csr_matvec(a["val"], a["col_idx"], a["row_ptr"], x_owned, ap)
                r_i = slab_rhs(slab, p) - ap
                r_c[i] = float(np.sum(r_i))
                r_norm_sq += float(np.dot(r_i, r_i))
            residual = np.sqrt(r_norm_sq) / b_norm

            # Coarse correction: scatter e_c[i] onto slab i's cells via gidx
            # (no need to retain owned-row maps for evicted slabs).
            if self.use_coarse and self.n_partitions > 1 and a_coarse is not None:
                e_c = np.linalg.solve(a_coarse, r_c)
                for i in range(self.n_partitions):
                    z_lo, z_hi = ranges[i]
                    sl = self.gidx[:, :, z_lo:z_hi]
                    p[sl[sl >= 0]] += e_c[i]

            if residual < self.target_error:
                break

        self.x = p
        self.rounds = rnd
        self.residual = residual
        return p, rnd, residual

    def _coarse_operator(self, scalars):
        """Tridiagonal coarse operator A_c (one DOF/slab) from per-slab scalars:
        diagonal = slab local matrix-entry sum; off-diagonal = interface coupling
        count (== the neighbour's halo_below)."""
        k = self.n_partitions
        A_c = np.zeros((k, k), dtype=np.float64)
        for i, (diag_block_sum, halo_above) in enumerate(scalars):
            A_c[i, i] = diag_block_sum
            if i + 1 < k:
                A_c[i, i + 1] = halo_above
                A_c[i + 1, i] = halo_above
        return A_c
