"""Pressure-Poisson (unit MAC Laplacian) geometry and assembly.

Model-neutral: this is the incompressibility operator used by the Stokes
projection (`stokesSolver.py`) and the distributed pressure solve
(`schwarzSolver.py`). It has **no dependency on the Darcy / fast-Laplacian model**
(`fastLaplacian`, `VolumeManager.get_conductivity`, conductance estimation) --
it assembles the *unit-conductivity* Laplacian directly from a fluid mask.

Convention (matches what `VolumeManager` produces for a unit-conductivity
regular volume, verified equal to ~1e-10): one row per fluid voxel; diagonal
-total_c, off-diagonal +1 to each fluid neighbour; z=0 is the inlet (Dirichlet
p=1, folded into the RHS as -2 on those rows) and z=max the outlet (p=0), both
added as a +2 diagonal ghost; solid walls are no-flux (dropped). Isotropic
voxels; regular (z-driven) geometry only.

Contents:
- `filter_percolating` / `global_fluid_index` / `slab_ranges` -- geometry helpers.
- `assemble_slab` -- local Laplacian for a z-slab (halo couplings moved to the
  RHS); used by the distributed Schwarz solver.
- `assemble_poisson` -- the whole-domain Laplacian (a slab spanning all of z, so
  no halo); used by the Stokes projection.
- `ravel` -- scatter a condensed pressure vector back onto the voxel grid.
"""

import numpy as np
import scipy.ndimage as ndi
from numba import njit


def filter_percolating(mask):
    """Keep only pore voxels in a cluster connected to BOTH the z=0 inlet and the
    z=max outlet. Isolated / dead-end pores would otherwise be singular
    (zero-diagonal) rows in the Laplacian.
    """
    labels, _ = ndi.label(mask)
    inlet = set(np.unique(labels[:, :, 0]))
    outlet = set(np.unique(labels[:, :, -1]))
    keep = (inlet & outlet) - {0}          # exclude the solid/background label
    if not keep:
        return np.zeros_like(mask)
    return np.isin(labels, list(keep))


def global_fluid_index(mask):
    """Map fluid voxels to condensed row indices (C order over x,y,z).

    Returns (gidx, n_fluid): gidx is an int64 array shaped like `mask` with the
    row index of each fluid voxel and -1 for solid.
    """
    flat = np.cumsum(mask.reshape(-1).astype(np.int64)) - 1
    gidx = np.where(mask.reshape(-1), flat, -1).astype(np.int64)
    return gidx.reshape(mask.shape), int(flat[-1] + 1) if mask.any() else 0


def slab_ranges(depth, n_partitions):
    """Contiguous z-ranges [(z_lo, z_hi), ...] partitioning [0, depth)."""
    bounds = [k * depth // n_partitions for k in range(n_partitions + 1)]
    return [(bounds[k], bounds[k + 1]) for k in range(n_partitions)]


@njit(cache=True)
def _assemble_slab_core(mask, gidx, z_lo, z_hi):
    """JIT core for `assemble_slab` (three passes: index, count, fill).

    Only z-direction neighbours can cross a slab boundary (x,y neighbours share
    the cell's z), so the halo bookkeeping is limited to the two z-faces.
    """
    W, H, D = mask.shape
    nzl = z_hi - z_lo
    lidx = -np.ones((W, H, nzl), dtype=np.int64)
    local_n = 0
    for x in range(W):
        for y in range(H):
            for z in range(z_lo, z_hi):
                if mask[x, y, z]:
                    lidx[x, y, z - z_lo] = local_n
                    local_n += 1

    # Pass 2: count nonzeros and halo couplings.
    nnz = 0
    n_halo = 0
    for x in range(W):
        for y in range(H):
            for z in range(z_lo, z_hi):
                if lidx[x, y, z - z_lo] < 0:
                    continue
                nnz += 1                                   # diagonal
                if x > 0 and mask[x - 1, y, z]:
                    nnz += 1
                if x < W - 1 and mask[x + 1, y, z]:
                    nnz += 1
                if y > 0 and mask[x, y - 1, z]:
                    nnz += 1
                if y < H - 1 and mask[x, y + 1, z]:
                    nnz += 1
                if z > 0 and mask[x, y, z - 1]:
                    if z - 1 >= z_lo:
                        nnz += 1
                    else:
                        n_halo += 1
                if z < D - 1 and mask[x, y, z + 1]:
                    if z + 1 < z_hi:
                        nnz += 1
                    else:
                        n_halo += 1

    val = np.empty(nnz, dtype=np.float64)
    col_idx = np.empty(nnz, dtype=np.int64)
    row_ptr = np.zeros(local_n, dtype=np.int64)
    owned_grows = np.empty(local_n, dtype=np.int64)
    b_base = np.zeros(local_n, dtype=np.float64)
    halo_rows = np.empty(n_halo, dtype=np.int64)
    halo_gcols = np.empty(n_halo, dtype=np.int64)
    halo_below = 0
    halo_above = 0

    # Pass 3: fill.
    pos = 0
    hpos = 0
    for x in range(W):
        for y in range(H):
            for z in range(z_lo, z_hi):
                r = lidx[x, y, z - z_lo]
                if r < 0:
                    continue
                owned_grows[r] = gidx[x, y, z]
                row_ptr[r] = pos
                total_c = 0.0
                if z == 0:
                    total_c += 2.0
                    b_base[r] -= 2.0
                elif z == D - 1:
                    total_c += 2.0
                diag_pos = pos
                val[pos] = 0.0
                col_idx[pos] = r
                pos += 1
                if x > 0 and mask[x - 1, y, z]:
                    total_c += 1.0; val[pos] = 1.0
                    col_idx[pos] = lidx[x - 1, y, z - z_lo]; pos += 1
                if x < W - 1 and mask[x + 1, y, z]:
                    total_c += 1.0; val[pos] = 1.0
                    col_idx[pos] = lidx[x + 1, y, z - z_lo]; pos += 1
                if y > 0 and mask[x, y - 1, z]:
                    total_c += 1.0; val[pos] = 1.0
                    col_idx[pos] = lidx[x, y - 1, z - z_lo]; pos += 1
                if y < H - 1 and mask[x, y + 1, z]:
                    total_c += 1.0; val[pos] = 1.0
                    col_idx[pos] = lidx[x, y + 1, z - z_lo]; pos += 1
                if z > 0 and mask[x, y, z - 1]:
                    total_c += 1.0
                    if z - 1 >= z_lo:
                        val[pos] = 1.0; col_idx[pos] = lidx[x, y, z - 1 - z_lo]; pos += 1
                    else:
                        halo_rows[hpos] = r; halo_gcols[hpos] = gidx[x, y, z - 1]
                        hpos += 1; halo_below += 1
                if z < D - 1 and mask[x, y, z + 1]:
                    total_c += 1.0
                    if z + 1 < z_hi:
                        val[pos] = 1.0; col_idx[pos] = lidx[x, y, z + 1 - z_lo]; pos += 1
                    else:
                        halo_rows[hpos] = r; halo_gcols[hpos] = gidx[x, y, z + 1]
                        hpos += 1; halo_above += 1
                val[diag_pos] = -total_c

    diag_block_sum = 0.0
    for i in range(nnz):
        diag_block_sum += val[i]
    return (val, col_idx, row_ptr, owned_grows, b_base, halo_rows, halo_gcols,
            diag_block_sum, halo_below, halo_above, local_n)


def assemble_slab(mask, gidx, z_lo, z_hi):
    """Local unit Laplacian for the fluid cells of a z-slab (see module docstring).

    A fluid neighbour outside the slab is a *halo* cell whose coupling is moved to
    the RHS (Dirichlet), to be filled from the neighbour slab's pressure. Returns
    a dict with the fixed local system (`a_sparse`, `b_base`), scatter map
    (`owned_grows`), halo coupling (`halo_rows`, `halo_gcols`), and coarse-operator
    inputs (`diag_block_sum`, `halo_below`, `halo_above`).
    """
    (val, col_idx, row_ptr, owned_grows, b_base, halo_rows, halo_gcols,
     diag_block_sum, halo_below, halo_above, local_n) = _assemble_slab_core(
        mask, gidx, int(z_lo), int(z_hi))
    return {
        "a_sparse": {"val": val, "col_idx": col_idx, "row_ptr": row_ptr},
        "b_base": b_base, "owned_grows": owned_grows,
        "halo_rows": halo_rows, "halo_gcols": halo_gcols,
        "local_n": int(local_n),
        "diag_block_sum": float(diag_block_sum),
        "halo_below": int(halo_below), "halo_above": int(halo_above),
    }


def assemble_poisson(mask):
    """Whole-domain unit pressure-Poisson for a (already-filtered) fluid mask.

    A single slab spanning all of z, so there is no halo. Returns a dict:
      gidx     : condensed row index per voxel (-1 solid)
      a_sparse : CSR matrix (project convention, row_ptr length N)
      b        : RHS carrying the inlet Dirichlet (z=0) contribution
      n        : number of fluid unknowns
    """
    gidx, n = global_fluid_index(mask)
    slab = assemble_slab(mask, gidx, 0, mask.shape[2])
    return {"gidx": gidx, "a_sparse": slab["a_sparse"], "b": slab["b_base"], "n": n}


def ravel(x, mask):
    """Scatter a condensed pressure vector onto the voxel grid (0 on solid)."""
    p = np.zeros(mask.shape, dtype=np.float64)
    p[mask] = x
    return p
