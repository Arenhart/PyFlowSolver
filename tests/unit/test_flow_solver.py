import numpy as np
import scipy as sc
import porespy as ps

from src.volumeManager import VolumeManager
from src.darcySolver import DarcySolver

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






