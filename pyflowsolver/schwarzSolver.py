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

This module contains the numerical core, written so it runs identically in two
modes:
  * ``solve_serial`` -- all slabs in one process (halo exchange in memory); used
    for validation against the single-domain solve and as a fallback.
  * distributed -- the same per-slab assemble/solve is shipped to Dask/SLURM
    workers (see docker/), exchanging halo layers between rounds.

Conventions match VolumeManager's regular (z-driven) bare-Laplacian: unit
conductivity, z=0 inlet (p=1) / z=max outlet (p=0) as diagonal ghost terms, and
no-flux (dropped) connections at solid walls. Isotropic voxels.
"""

import numpy as np

from pyflowsolver.multigridSolver import MultigridSolver, _csr_matvec


def global_fluid_index(mask):
    """Map fluid voxels to global condensed row indices (C order over x,y,z).

    Returns (gidx, n_fluid): gidx is an int64 array shaped like `mask` with the
    row index of each fluid voxel and -1 for solid, matching the row order
    VolumeManager produces.
    """
    flat = np.cumsum(mask.reshape(-1).astype(np.int64)) - 1
    gidx = np.where(mask.reshape(-1), flat, -1).astype(np.int64)
    return gidx.reshape(mask.shape), int(flat[-1] + 1) if mask.any() else 0


def slab_ranges(depth, n_partitions):
    """Contiguous z-ranges [(z_lo, z_hi), ...] partitioning [0, depth)."""
    bounds = [k * depth // n_partitions for k in range(n_partitions + 1)]
    return [(bounds[k], bounds[k + 1]) for k in range(n_partitions)]


def assemble_slab(mask, gidx, z_lo, z_hi):
    """Assemble the local Laplacian for the fluid cells of a z-slab.

    `mask` is the full geometry (bool/uint8); `gidx` the global fluid index. Only
    voxels with z in [z_lo, z_hi) are local unknowns. A fluid neighbour outside
    the slab (at z_lo-1 or z_hi) is a *halo* cell: its coupling is moved to the
    RHS (Dirichlet), to be filled each round from the neighbour slab's latest
    pressure. z=0 / z=max carry the inlet/outlet ghost exactly as VolumeManager.

    Returns a dict with the fixed local system and the halo coupling:
      a_sparse   : local CSR (project convention, row_ptr length local_n)
      b_base     : fixed RHS part (inlet ghost); halo part added each round
      owned_grows: global row index of each local row (for scatter back)
      halo_rows  : local rows that couple to a halo cell
      halo_gcols : global index of the halo cell for each halo coupling
      local_n    : number of local unknowns
    """
    W, H, D = mask.shape
    # Local index for owned fluid cells (C order within the slab).
    owned = []
    lidx = -np.ones((W, H, z_hi - z_lo), dtype=np.int64)
    for x in range(W):
        for y in range(H):
            for z in range(z_lo, z_hi):
                if mask[x, y, z]:
                    lidx[x, y, z - z_lo] = len(owned)
                    owned.append(gidx[x, y, z])
    local_n = len(owned)
    owned_grows = np.array(owned, dtype=np.int64)

    vals, cols, row_ptr = [], [], np.zeros(local_n, dtype=np.int64)
    b_base = np.zeros(local_n, dtype=np.float64)
    halo_rows, halo_gcols = [], []
    halo_below = 0          # couplings to the slab below (z < z_lo)
    halo_above = 0          # couplings to the slab above (z >= z_hi)

    neigh = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    for x in range(W):
        for y in range(H):
            for z in range(z_lo, z_hi):
                if not mask[x, y, z]:
                    continue
                row = lidx[x, y, z - z_lo]
                row_ptr[row] = len(vals)
                total_c = 0.0
                # inlet / outlet ghost (global z boundary), exactly as VolumeManager
                if z == 0:
                    total_c += 2.0
                    b_base[row] -= 2.0
                elif z == D - 1:
                    total_c += 2.0
                diag_pos = len(vals)
                vals.append(0.0)            # reserve diagonal slot
                cols.append(row)
                for dx, dy, dz in neigh:
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if not (0 <= xx < W and 0 <= yy < H and 0 <= zz < D):
                        continue            # domain wall: no-flux, no contribution
                    if not mask[xx, yy, zz]:
                        continue            # solid wall: no-flux
                    total_c += 1.0          # fluid face contributes to the diagonal
                    if z_lo <= zz < z_hi:
                        cols.append(lidx[xx, yy, zz - z_lo])   # owned neighbour
                        vals.append(1.0)
                    else:
                        halo_rows.append(row)                  # halo (Dirichlet)
                        halo_gcols.append(gidx[xx, yy, zz])
                        if zz < z_lo:
                            halo_below += 1
                        else:
                            halo_above += 1
                vals[diag_pos] = -total_c

    a_sparse = {
        "val": np.array(vals, dtype=np.float64),
        "col_idx": np.array(cols, dtype=np.int64),
        "row_ptr": row_ptr,
    }
    return {
        "a_sparse": a_sparse, "b_base": b_base, "owned_grows": owned_grows,
        "halo_rows": np.array(halo_rows, dtype=np.int64),
        "halo_gcols": np.array(halo_gcols, dtype=np.int64),
        "local_n": local_n,
        "diag_block_sum": float(np.sum(a_sparse["val"])),  # A_c[k,k] = P^T A P
        "halo_below": halo_below, "halo_above": halo_above,
    }


class SchwarzSolver:
    """Additive-Schwarz Poisson solver over geometry z-slabs (RAM-bounded)."""

    def __init__(self, volume, n_partitions=2, target_error=1e-6,
                 max_rounds=500, local_target=1e-9, use_coarse=True):
        self.mask = (np.asarray(volume) > 0)
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
