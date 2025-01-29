import numpy as np
import scipy as sc
import porespy as ps

from pyedt import edt

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver
from tests.unit.resources.templates import irregular_boundary_template
from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET

image = ps.generators.blobs(shape=(25, 25, 25), porosity=0.38, seed=42)
labeled_image, _ = sc.ndimage.label(image)
image = labeled_image == 1

volume_manager = VolumeManager(image)

dense_A, dense_b = volume_manager.get_linear_system()
solution_template = np.linalg.solve(dense_A, dense_b)
raveled_template = volume_manager.ravel_dense_solution(solution_template)

def test_darcy_solver():
    sparse_A, dense_b = volume_manager.get_sparse_system_jit()
    sparse_A_ref, dense_b_ref = volume_manager.get_sparse_system()
    np.testing.assert_allclose(sparse_A["val"], sparse_A_ref["val"], rtol=1e-08)
    np.testing.assert_allclose(sparse_A["row_ptr"], sparse_A_ref["row_ptr"], rtol=1e-08)
    np.testing.assert_allclose(sparse_A["col_idx"], sparse_A_ref["col_idx"], rtol=1e-08)
    np.testing.assert_allclose(dense_b, dense_b_ref, rtol=1e-08)
    solver = DarcySolver()
    solver.set_linear_system(sparse_A, dense_b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    solution, _, _ = solver.solve_pcg()
    raveled_solution = volume_manager.ravel_sparse_solution(solution)
    np.testing.assert_allclose(raveled_solution, raveled_template, rtol=1e-08)


def test_irregular_boundary_solver():

    boundary_volume = np.zeros((5,5,5), dtype=np.uint8)
    boundary_volume[1:-1, :, 1:-1] = PORE
    boundary_volume[1:-1,0,1:-1] = INLET
    boundary_volume[1,1,1] = INLET
    boundary_volume[1,0,1] = PORE
    boundary_volume[1:-1, -1, 1:-1] = OUTLET

    con_vol = (boundary_volume>=1)*100

    scale = (0.02,) * 3

    volume_manager = VolumeManager(
        con_vol, 
        scale=scale, 
        boundary_volume=boundary_volume,
        )
    volume_manager.convert_pore_volume_to_laplacian_conductivity()
    sparse_array, dense_b = volume_manager.get_sparse_system_jit()
    solver = DarcySolver()
    solver.set_linear_system(sparse_array, dense_b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    solution, _, _ = solver.solve_pcg()
    raveled_solution = volume_manager.ravel_sparse_solution(solution)
    np.testing.assert_allclose(
        raveled_solution, 
        irregular_boundary_template, 
        rtol=1e-06,
        )

def test_irregular_boundary_square_duct():
    boundary_volume = np.zeros((202, 41, 202), dtype=np.uint8)
    #boundary_volume = np.zeros((5, 5, 5), dtype=np.uint8)
    boundary_volume[1:-1, :, 1:-1,] = 1
    boundary_volume[1:-1, 0, 1:-1] = 2
    boundary_volume[1:-1, -1, 1:-1] = 3
    porosity_volume = (boundary_volume>=1)*100
    scale = 0.02

    volume_manager = VolumeManager(
        porosity_volume, 
        boundary_volume=boundary_volume,
        scale=scale,
        )
    volume_manager.convert_pore_volume_to_laplacian_conductivity()

    sparse_array, dense_b = volume_manager.get_sparse_system_jit()
    solver = DarcySolver()
    solver.set_linear_system(sparse_array, dense_b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    solution, _, _ = solver.solve_pcg()
    pressure_volume = volume_manager.ravel_sparse_solution(solution)

    conductivity = volume_manager.get_conductivity(
        pressure_volume,
        )
    reference_conductivity = 12.7313
    assert(np.isclose(conductivity, reference_conductivity, rtol=0.1))


#def test_stokes_solver():
#    pass






