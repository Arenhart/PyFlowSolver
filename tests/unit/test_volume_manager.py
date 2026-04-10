import numpy as np

from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.sparseArray import SparseArray
from tests.unit.resources.templates import (
    A_template, 
    b_template,
    sparse_array_template,
    condensed_b_template,
)


volume = np.ones((4,4,4), dtype=int)
volume[1:3, 1:3, 1] = 0
volume[1:3, 1:3, 2] = 2
volume_manager = VolumeManager(volume)

def test_get_linear_system():
    A, b = volume_manager.get_linear_system()
    np.testing.assert_array_equal(A, A_template)
    np.testing.assert_array_equal(b, b_template)

def test_get_sparse_system():
    sparse_array, condensed_b = volume_manager.get_sparse_system_jit()
    test_array = SparseArray(sparse_array["val"], sparse_array["col_idx"], sparse_array["row_ptr"])
    np.testing.assert_allclose(test_array.val, sparse_array_template.val, rtol=1e-06)
    np.testing.assert_array_equal(test_array.col_idx, sparse_array_template.col_idx)
    np.testing.assert_array_equal(test_array.row_ptr, sparse_array_template.row_ptr)
    np.testing.assert_array_equal(condensed_b, condensed_b_template)





