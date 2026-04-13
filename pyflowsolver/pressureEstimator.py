import numpy as np
from scipy import ndimage

from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET


def segment_pore_space(pore_volume, subsegment_size = 20):
    """Segment a 3D porosity map into coarse regions for pseudo-network creation.

    Takes the boundary_volume (uint8 3D array with SOLID/PORE) 
    and groups PORE voxels into contiguous segments.
    Inly works for regular boundaries.

    Parameters
    ----------
    pore_volume : np.ndarray
        uint8 3D array where each voxel is SOLID(0) or PORE(1)

    Returns
    -------
    segmented_volume : np.ndarray
        int32 3D array, same shape as pore_volume.
        0 = solid (no segment), positive integers = segment labels.
    n_segments : int
        Total number of segments created.
    """
    w, h, d = pore_volume.shape
    segmented_volume = np.zeros((w, h, d), dtype=np.int32)
    total_segments = 0
    for i in range(0, w, subsegment_size):
        for j in range(0, h, subsegment_size):
            for k in range(0, d, subsegment_size):
                subregion = pore_volume[i:i+subsegment_size, j:j+subsegment_size, k:k+subsegment_size]
                labeled_subregion, n_subsegments = ndimage.label(subregion)
                labeled_subregion[labeled_subregion > 0] += total_segments
                total_segments += n_subsegments
                segmented_volume[i:i+subsegment_size, j:j+subsegment_size, k:k+subsegment_size] = labeled_subregion
    return segmented_volume, total_segments


def create_pseudo_network(segmented_volume, conductivity_volume, scale):
    """Build a NetworkManager-compatible pore network from segmented regions.

    Each segment becomes a pore. Two pores are connected (by a throat) if
    their segments share a face-adjacent voxel boundary. The throat
    conductance is geometric mean of the pore voxels conductivity
    multiplied by the total voxel volume (considering both touching segments)
     and divided by the square of the distance between segment centroids.

    Inlets and outlets are added as new pores connected to segments that touch
    the volume boundary in z = 0 (inlet) and z = segmented_volume.shape[2] (outlet).
    Their conductivities are calculated similarly to other throats, but using the
    the distance from the segment centroid to the volume boundary.

    Parameters
    ----------
    segmented_volume : np.ndarray
        int32 3D array of segment labels (from segment_pore_space).
    conductivity_volume : np.ndarray
        float 3D array of per-voxel conductivities (from VolumeManager
        after convert_pore_volume_to_laplacian_conductivity).
    scale : tuple of float
        Voxel dimensions (dx, dy, dz).

    Returns
    -------
    conn : np.ndarray, shape (n_throats, 2)
        Pore connectivity array (pairs of 0-indexed pore IDs).
    cond : np.ndarray, shape (n_throats,)
        Throat conductances.
    inlets : np.ndarray, shape (n_pores,)
        Boolean array, True for the inlet pore.
    outlets : np.ndarray, shape (n_pores,)
        Boolean array, True for the outlet pore.
    """
    w, h, d = segmented_volume.shape
    dx, dy, dz = float(scale[0]), float(scale[1]), float(scale[2])
    voxel_volume = dx * dy * dz

    n_segments = int(segmented_volume.max())

    # Two virtual pores: inlet at z=0, outlet at z=d-1
    inlet_idx = n_segments
    outlet_idx = n_segments + 1
    n_pores = n_segments + 2

    # --- Per-segment stats in a single pass ---
    centroids = np.zeros((n_segments, 3), dtype=np.float64)
    counts = np.zeros(n_segments, dtype=np.int64)
    log_cond_prods = np.zeros(n_segments, dtype=np.float64)
    cond_inverse_sums = np.zeros(n_segments, dtype=np.float64)
    touches_inlet = np.zeros(n_segments, dtype=bool)
    touches_outlet = np.zeros(n_segments, dtype=bool)

    for x in range(w):
        for y in range(h):
            for z in range(d):
                lbl = segmented_volume[x, y, z]
                if lbl <= 0:
                    continue
                idx = lbl - 1
                centroids[idx, 0] += x * dx
                centroids[idx, 1] += y * dy
                centroids[idx, 2] += (z + 0.5) * dz
                counts[idx] += 1

                c = conductivity_volume[x, y, z]
                if c > 0:
                    log_cond_prods[idx] += np.log(c)
                    cond_inverse_sums[idx] += 1.0 / c

                if z == 0:
                    touches_inlet[idx] = True
                if z == d - 1:
                    touches_outlet[idx] = True

    nonzero = counts > 0
    centroids[nonzero] /= counts[nonzero, np.newaxis]

    # Geometric mean of conductivity per segment (for inlet/outlet throats)
    geom_mean_per_seg = np.zeros(n_segments, dtype=np.float64)
    geom_mean_per_seg[nonzero] = np.exp(log_cond_prods[nonzero] / counts[nonzero])

    # --- Find face-adjacent inter-segment connections ---
    throat_pairs = set()
    face_directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for di, dj, dk in face_directions:
        for x in range(w - di):
            for y in range(h - dj):
                for z in range(d - dk):
                    lbl1 = segmented_volume[x, y, z]
                    lbl2 = segmented_volume[x + di, y + dj, z + dk]

                    if lbl1 <= 0 or lbl2 <= 0 or lbl1 == lbl2:
                        continue

                    idx1 = lbl1 - 1
                    idx2 = lbl2 - 1
                    throat_pairs.add((min(idx1, idx2), max(idx1, idx2)))

    # --- Build throats ---
    conn_list = []
    cond_list = []

    # Inter-segment throats
    for idx1, idx2 in throat_pairs:
        total_count = counts[idx1] + counts[idx2]
        if total_count <= 0:
            continue
        combined_log_sum = log_cond_prods[idx1] + log_cond_prods[idx2]
        geom_mean = np.exp(combined_log_sum / total_count)
        harmonic_mean = total_count / (cond_inverse_sums[idx1] + cond_inverse_sums[idx2])
        total_vol = total_count * voxel_volume
        dist = np.linalg.norm(centroids[idx1] - centroids[idx2])
        if dist > 0 and geom_mean > 0:
            conn_list.append((idx1, idx2))
            cond_list.append(geom_mean * total_vol / (2 * dist ** 2))

    # Inlet throats: segments touching z=0 connect to the virtual inlet pore
    for idx in range(n_segments):
        if not touches_inlet[idx] or counts[idx] <= 0:
            continue
        gm = geom_mean_per_seg[idx]
        if gm <= 0:
            continue
        total_vol = counts[idx] * voxel_volume
        dist = centroids[idx, 2]  # physical distance from centroid to z=0
        if dist <= 0:
            dist = dz / 2.0
        conn_list.append((idx, inlet_idx))
        cond_list.append(gm * total_vol / (dist ** 2))

    # Outlet throats: segments touching z=d-1 connect to the virtual outlet pore
    z_max_phys = (d - 1) * dz
    for idx in range(n_segments):
        if not touches_outlet[idx] or counts[idx] <= 0:
            continue
        gm = geom_mean_per_seg[idx]
        if gm <= 0:
            continue
        total_vol = counts[idx] * voxel_volume
        dist = z_max_phys - centroids[idx, 2]
        if dist <= 0:
            dist = dz / 2.0
        conn_list.append((idx, outlet_idx))
        cond_list.append(gm * total_vol / (dist ** 2))

    # --- Pack into arrays ---
    n_throats = len(conn_list)
    if n_throats > 0:
        conn = np.array(conn_list, dtype=np.int32)
        cond = np.array(cond_list, dtype=np.float64)
    else:
        conn = np.empty((0, 2), dtype=np.int32)
        cond = np.empty(0, dtype=np.float64)

    inlets = np.zeros(n_pores, dtype=bool)
    outlets = np.zeros(n_pores, dtype=bool)
    inlets[inlet_idx] = True
    outlets[outlet_idx] = True

    return conn, cond, inlets, outlets


def solve_pseudo_network(conn, cond, inlets, outlets):
    """Solve the pseudo-network for pressure using NetworkManager.

    Parameters
    ----------
    conn : np.ndarray, shape (n_throats, 2)
        Pore connectivity.
    cond : np.ndarray, shape (n_throats,)
        Throat conductances.
    inlets : np.ndarray, shape (n_segments,)
        Boolean array marking inlet pores.
    outlets : np.ndarray, shape (n_segments,)
        Boolean array marking outlet pores.

    Returns
    -------
    segment_pressures : np.ndarray, shape (n_pores,)
        Solved pressure for each pore (inlet=1, outlet=0).
    """
    from pyflowsolver.networkManager import NetworkManager
    from pyflowsolver.darcySolver import DarcySolver

    nm = NetworkManager(conn, cond, inlets, outlets)
    nm.generate_sparse_system()
    sparse_A, b = nm.get_sparse_system()

    solver = DarcySolver()
    solver.set_linear_system(sparse_A, b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    x, _, _ = solver.solve_pcg()

    pressures = nm.get_pressure_list(x, pressure_drop=1.0)
    return pressures


def create_pressure_volume(segmented_volume, segment_pressures, scale, radius, sigma):
    """Map solved segment pressures back to the full 3D volume.

    Three stages:
    1. Each pore voxel receives the pressure of its segment.
    2. Voxels in segments touching z=0 (inlet) or z=max (outlet) are
       linearly interpolated along z from the segment centroid pressure
       to 1.0 at z=0 or 0.0 at z=max.
    3. A masked Gaussian blur is applied to smooth inter-segment
       boundaries. Only pore voxels contribute to the kernel and
       interpolated voxels are not modified.

    Parameters
    ----------
    segmented_volume : np.ndarray
        int32 3D array of segment labels (0 = solid, >0 = segment).
    segment_pressures : np.ndarray
        Pressure per pore from solve_pseudo_network.
    scale : tuple of float
        Voxel dimensions (dx, dy, dz).

    Returns
    -------
    pressure_volume : np.ndarray
        float64 3D array, same shape as segmented_volume.
        Estimated pressure at each voxel.
    """
    from scipy.ndimage import gaussian_filter

    w, h, d = segmented_volume.shape
    dx, dy, dz = float(scale[0]), float(scale[1]), float(scale[2])

    n_segments = int(segmented_volume.max())
    z_max_phys = (d - 0.5) * dz

    # --- Per-segment centroid z and boundary flags ---
    centroids_z = np.zeros(n_segments, dtype=np.float64)
    counts = np.zeros(n_segments, dtype=np.int64)
    touches_inlet = np.zeros(n_segments, dtype=bool)
    touches_outlet = np.zeros(n_segments, dtype=bool)

    for x in range(w):
        for y in range(h):
            for z in range(d):
                lbl = segmented_volume[x, y, z]
                if lbl <= 0:
                    continue
                idx = lbl - 1
                centroids_z[idx] += (z + 0.5) * dz
                counts[idx] += 1
                if z == 0:
                    touches_inlet[idx] = True
                if z == d - 1:
                    touches_outlet[idx] = True

    nonzero = counts > 0
    centroids_z[nonzero] /= counts[nonzero]

    # --- Step 1 & 2: fill pressures and interpolate near boundaries ---
    pressure_volume = np.zeros((w, h, d), dtype=np.float64)
    interpolated = np.zeros((w, h, d), dtype=bool)

    for x in range(w):
        for y in range(h):
            for z in range(d):
                lbl = segmented_volume[x, y, z]
                if lbl <= 0:
                    continue
                idx = lbl - 1
                seg_p = segment_pressures[idx]
                phys_z = (z + 0.5) * dz

                if touches_inlet[idx] and phys_z <= centroids_z[idx]:
                    # Interpolate from seg_p at centroid to 1.0 at z=0
                    cz = centroids_z[idx]
                    t = (cz - phys_z) / cz if cz > 0 else 1.0
                    pressure_volume[x, y, z] = seg_p + t * (1.0 - seg_p)
                    interpolated[x, y, z] = True
                elif touches_outlet[idx] and phys_z >= centroids_z[idx]:
                    # Interpolate from seg_p at centroid to 0.0 at z=max
                    denom = z_max_phys - centroids_z[idx]
                    t = (phys_z - centroids_z[idx]) / denom if denom > 0 else 1.0
                    pressure_volume[x, y, z] = seg_p * (1.0 - t)
                    interpolated[x, y, z] = True
                else:
                    pressure_volume[x, y, z] = seg_p

    # --- Step 3: masked Gaussian blur, preserving interpolated voxels ---
    pore_mask = (segmented_volume > 0).astype(np.float64)

    # Normalised blur: only pore voxels contribute to the kernel
    blurred_values = gaussian_filter(
        pressure_volume * pore_mask, 
        sigma=sigma, 
        radius=radius,
        mode="nearest",
    )
    blurred_weights = gaussian_filter(pore_mask, sigma=sigma, radius=radius, mode="nearest")
    blurred_weights[blurred_weights == 0] = 1.0
    blurred = blurred_values / blurred_weights

    # Apply blur only to non-interpolated pore voxels
    apply_mask = (segmented_volume > 0) & ~interpolated
    pressure_volume[apply_mask] = blurred[apply_mask]

    return pressure_volume


def estimate_pressure_distribution(
        pore_volume, 
        conductivity_volume, 
        scale, 
        subsegment_size=20,
        sigma=2,
        radius=5,
        ):
    """Convenience function: run all three steps to get a pressure estimate volume.

    Segments the pore space, builds and solves a pseudo-network, and maps
    pressures back to the 3D volume.

    Parameters
    ----------
    pore_volume : np.ndarray
        uint8 3D array (0 = solid, 1 = pore).
    conductivity_volume : np.ndarray
        float 3D array of per-voxel conductivities.
    scale : tuple of float
        Voxel dimensions (dx, dy, dz).

    Returns
    -------
    pressure_volume : np.ndarray
        float64 3D array with estimated pressure at each voxel.
    """
    segmented_volume, n_segments = segment_pore_space(pore_volume, subsegment_size=subsegment_size)
    conn, cond, inlets, outlets = create_pseudo_network(
        segmented_volume, conductivity_volume, scale,
    )
    segment_pressures = solve_pseudo_network(conn, cond, inlets, outlets)
    pressure_volume = create_pressure_volume(
        segmented_volume, 
        segment_pressures, 
        scale,
        sigma=sigma,
        radius=radius,
    )
    return pressure_volume
