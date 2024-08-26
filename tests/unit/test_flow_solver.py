import numpy as np
import scipy as sc
import porespy as ps

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver

image = ps.generators.blobs(shape=(25, 25, 25), porosity=0.38, seed=42)
labeled_image, _ = sc.ndimage.label(image)
image = labeled_image == 1

volume_manager = VolumeManager(image)

dense_A, dense_b = volume_manager.get_linear_system()
solution_template = np.linalg.solve(dense_A, dense_b)
raveled_template = volume_manager.ravel_dense_solution(solution_template)

def test_darcy_solver():
    sparse_A, sparse_b = volume_manager.get_sparse_system()
    solver = DarcySolver()
    solution = solver.solve(sparse_A, sparse_b)
    raveled_solution = volume_manager.ravel_sparse_solution(solution)
    np.testing.assert_allclose(raveled_solution, raveled_template, rtol=1e-06)

def test_darcy_solver_jit():
    sparse_A, sparse_b = volume_manager.get_sparse_system_jit()
    sparse_A_ref, sparse_b_ref = volume_manager.get_sparse_system()
    np.testing.assert_allclose(sparse_A.val, sparse_A_ref.val, rtol=1e-08)
    np.testing.assert_allclose(sparse_A.row_ptr, sparse_A_ref.row_ptr, rtol=1e-08)
    np.testing.assert_allclose(sparse_A.col_idx, sparse_A_ref.col_idx, rtol=1e-08)
    np.testing.assert_allclose(sparse_b, sparse_b_ref, rtol=1e-08)
    solver = DarcySolver()
    solution = solver.solve_jit(sparse_A, sparse_b)
    raveled_solution = volume_manager.ravel_sparse_solution(solution)
    np.testing.assert_allclose(raveled_solution, raveled_template, rtol=1e-08)

def test_darcy_solver_jit_parallel():
    sparse_A, sparse_b = volume_manager.get_sparse_system_jit()
    solver = DarcySolver()
    solution = solver.solve_jit(sparse_A, sparse_b, parallel=True)
    raveled_solution = volume_manager.ravel_sparse_solution(solution)
    np.testing.assert_allclose(raveled_solution, raveled_template, rtol=1e-08)
    
def test_stokes_solver():
    pass






