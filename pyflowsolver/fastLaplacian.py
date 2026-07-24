
from pyedt import edt
import numpy as np
from numba import njit, prange

from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET

def fast_laplacian_volume_generator(
        porosity_volume, 
        pore_scale,
        subresolution_function=None,
        closed_border=True,
        enhanced_model=False,
        ):
    """
    porosity_map: must be an uint 3D ndarray, 0 represents solid, 100 pore, and 1-99
    are subresolution voxels with this indicated porosity.
    pore_scale: must be a float ndarray with 3 values, for voxel length across x, y, and
    z axes. 
    If scale is given in mm, the resulting permeability will be in mm^2. To convert it
    to mD, multiply by 1.0132e9
    Anisotropic volumes are not implemented. The result will be valid but not physically
    coherent in case of anisotropy.
    subresolution_function: a function that takes an int between 1 and 99 and returns a 
    float.
    """

    if subresolution_function is None:
        estimated_subresolution_conductance = (np.min(pore_scale)/10) ** 2
        subresolution_function = lambda x: (x/100) * estimated_subresolution_conductance

    darcy_pores = np.logical_and((porosity_volume > 0), (porosity_volume < 100))
    stokes_pores = (porosity_volume == 100)
    darcy_pores = porosity_volume * darcy_pores
    darcy_pores = subresolution_function(darcy_pores)
    #conductance_array = np.zeros_like(porosity_volume, dtype=np.float32)
    w, h, d = stokes_pores.shape
    if closed_border is True:
        stokes_pores_with_border = np.zeros((w+2, h+2, d), dtype=stokes_pores.dtype)
        stokes_pores_with_border[1:-1, 1:-1, :] = stokes_pores
        conductance_array = edt(
            stokes_pores_with_border, 
            scale=tuple(pore_scale),
            force_method="cpu",
            )
        conductance_array = conductance_array[1:-1, 1:-1, :]
    else: #closed border is false
        conductance_array = edt(
            stokes_pores, 
            scale=tuple(pore_scale),
            force_method="cpu",
            )
    alfa = np.min(pore_scale)/2
    if enhanced_model:
        footprint_array = np.zeros_like(conductance_array, dtype=np.float32)
        edt_array = conductance_array
        _calculate_footprint(edt_array, footprint_array, spacing=tuple(pore_scale))
        conductance_array = (
            (footprint_array + alfa) ** 2
            - (footprint_array + alfa - edt_array) ** 2
        ) / 4
        if np.any(conductance_array < 0):
            raise ValueError(
                "Enhanced Arns model produced negative conductivity "
                f"(min={conductance_array.min():.6g}); footprint should always "
                "be >= edt, so this signals a footprint/EDT inconsistency."
            )

    else:
        conductance_array = (conductance_array - alfa)**2

    conductance_array *= stokes_pores
    conductance_array += darcy_pores

    return conductance_array

@njit(parallel=True)
def _calculate_footprint(edt_array, output_array, spacing):
    """Local-thickness / footprint field.

        output[v] = max{ edt[u] : ||u - v|| <= edt[u] }   for pore voxels (edt[v] > 0)

    i.e. the largest radius of any inscribed ball (centred at any voxel u) that
    still covers v. Since a voxel covers itself (u == v, distance 0), the result
    is always >= edt[v].
    """
    nx, ny, nz = edt_array.shape
    sx, sy, sz = spacing[0], spacing[1], spacing[2]
    r_max = np.max(edt_array)
    r_x = int(np.ceil(r_max / sx))
    r_y = int(np.ceil(r_max / sy))
    r_z = int(np.ceil(r_max / sz))
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if edt_array[i, j, k] <= 0:
                    output_array[i, j, k] = 0.0
                    continue
                best = 0.0
                for d1 in range(-r_x, r_x + 1):
                    ii = i + d1
                    if ii < 0 or ii >= nx:
                        continue
                    for d2 in range(-r_y, r_y + 1):
                        jj = j + d2
                        if jj < 0 or jj >= ny:
                            continue
                        for d3 in range(-r_z, r_z + 1):
                            kk = k + d3
                            if kk < 0 or kk >= nz:
                                continue
                            r = edt_array[ii, jj, kk]
                            # Only a larger covering radius can improve `best`;
                            # test that before the (costlier) distance check.
                            if r > best:
                                dist = np.sqrt((d1 * sx) ** 2
                                               + (d2 * sy) ** 2
                                               + (d3 * sz) ** 2)
                                if dist <= r:
                                    best = r
                output_array[i, j, k] = best
