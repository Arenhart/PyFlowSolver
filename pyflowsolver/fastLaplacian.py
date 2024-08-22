
from pyedt import edt
import numpy as np

def fast_laplacian_volume_generator(porosity_volume, pore_scale, subresolution_function=None):
    """
    porosity_map: must be an uint 3D ndarray, 0 represents solid, 100 pore, and 1-99
    are subresolution voxels with this indicated porosity.
    pore_scale: must be a float ndarray with 3 values, for voxel length across x, y, and
    z axes. Anisotropic are not implemented. The result will be valid but not physically
    coherent.
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

    conductance_array = np.zeros_like(porosity_volume, dtype=np.float32)
    conductance_array = edt(stokes_pores, scale=pore_scale)
    alfa = np.min(pore_scale)/2
    conductance_array = (conductance_array - alfa)**2
    conductance_array *= stokes_pores

    conductance_array += darcy_pores

    return conductance_array